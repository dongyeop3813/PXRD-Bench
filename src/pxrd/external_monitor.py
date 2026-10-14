"""Optional frozen external validation alongside the native training monitor."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from src.pxrd.eval_runtime import run_pxrd_match_eval


@contextmanager
def preserve_training_rng():
    """External reporting must not change the subsequent optimization stream."""
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_external_monitors(args):
    manifest = getattr(args, 'external_validation_manifest', None)
    if not manifest:
        return []
    if not args.csp or args.monitor_every <= 0:
        raise ValueError('External validation requires active CSP monitoring')
    if _sha(manifest) != getattr(args, 'external_validation_sha256', None):
        raise ValueError('External validation manifest SHA256 mismatch')
    entries = json.loads(Path(manifest).read_text())['datasets']
    names = [entry['name'] for entry in entries]
    if not names or len(names) != len(set(names)) or not set(names) <= {'rruff', 'opxrd'}:
        raise ValueError('Expected unique frozen RRUFF/opXRD validation manifests')
    from src.pxrd.eval_task_runtime import _build_dataset
    from src.training.data_module import _dataset_material_ids

    monitors = []
    with preserve_training_rng():
        for entry in entries:
            if entry['logical_split'] != 'validation':
                raise ValueError('External training monitors must use validation only')
            for path, digest in entry['files_sha256'].items():
                if _sha(path) != digest:
                    raise ValueError(f'External dataset checksum mismatch: {path}')
            external_args = copy.copy(args)
            for key in ('dataset_class', 'data_root', 'split'):
                setattr(external_args, key, entry[key])
            external_args.dataset_name = entry['name']
            external_args.real_pxrd_peak_preprocessing = 'official'
            external_args.experimental_background_subtraction = True
            external_args.simulated_profile_augmentation_seed = None
            external_args.structure_cache = None
            dataset, _, _ = _build_dataset(external_args, vars(args))
            indices = entry['target_indices']
            expected_count = {'rruff': 48, 'opxrd': 24}[entry['name']]
            if len(indices) != expected_count or len(set(indices)) != len(indices):
                raise ValueError('External validation target count or uniqueness mismatch')
            if any(not isinstance(i, int) or not 0 <= i < len(dataset) for i in indices):
                raise ValueError('External validation index out of bounds')
            ids = list(_dataset_material_ids(dataset))
            if [ids[i] for i in indices] != entry['sample_ids']:
                raise ValueError('External validation sample IDs/order mismatch')
            monitors.append((entry['name'], dataset, torch.tensor(indices), entry['sample_ids']))
    return monitors


def run_external_monitors(monitors, *, model, type_encoding, args, device,
                          step, artifacts_dir, use_ema):
    """Called inside the existing EMA context; return metrics for the same log row."""
    if not monitors:
        return {}
    external_args = copy.copy(args)
    external_args.matcher_profile = 'strict'
    external_args.match_gate = 'matcher_only'
    external_args.monitor_num_candidates = 1
    external_args.monitor_topk = [1]
    combined = {}
    with preserve_training_rng():
        for name, dataset, indices, sample_ids in monitors:
            metrics, targets = run_pxrd_match_eval(
                model=model, eval_ds=dataset, target_indices=indices,
                type_encoding=type_encoding, args=external_args, device=device,
                return_per_target=True,
            )
            if len(targets) != len(indices):
                raise ValueError('Incomplete external validation monitor results')
            metrics['pxrd_match_rate@1'] = sum(row['match@1'] for row in targets) / len(targets)
            for row, sample_id in zip(targets, sample_ids, strict=True):
                row['sample_id'] = sample_id
            section = f'{name}_val_{"ema" if use_ema else "raw"}'
            combined.update({f'{section}/{key}': value for key, value in metrics.items()})
            record = dict(step=int(step), dataset=name, logical_split='validation',
                          monitor_seed=int(args.monitor_seed), num_candidates=1,
                          topk=[1], num_targets=len(indices), use_ema=bool(use_ema),
                          matcher_profile='strict', match_gate='matcher_only',
                          background_subtraction=True,
                          pxrd_peak_min_prominence=float(getattr(args, "pxrd_peak_min_prominence", 0.0)),
                          metrics=metrics, targets=targets)
            with (Path(artifacts_dir) / 'external_periodic_eval_per_target.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
    return combined
