"""Train Bragglite with known composition or element-set conditioning."""

from __future__ import annotations
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import hydra
import torch
from omegaconf import DictConfig
from src.bragglite.factory import build_bragglite
from src.data.mp20_tokens import VZ
from src.models.type_encoding import build_type_encoding
from src.pxrd.eval_runtime import run_pxrd_match_eval, select_eval_targets
from src.pxrd.joint_eval_runtime import evaluate_joint_model
from src.pxrd.external_monitor import load_external_monitors, run_external_monitors
from src.training.checkpointing import build_train_checkpoint
from src.training.data_module import (
    build_dataloaders,
    make_token_dataset,
    setup_datasets,
)
from src.training.runner import TrainingRunContext, run_training
from src.training.runtime import (
    TrainingPaths,
    configure_device,
    copy_hydra_metadata,
    maybe_compile_model,
)
from src.training.distributed import initialize_distributed, wrap_distributed_model
from src.training.optim import build_optimizer_and_scheduler
from src.utils.checkpoint import _BestCandidate, maybe_update_best_ckpt
from src.utils.ema import EMA
from src.utils.seeding import seed_everything
from src.utils.provenance import git_commit
from src.utils.wandb_utils import init_wandb, log_metrics, log_summary
from src.pxrd.model_runtime import maybe_use_ema
from src.eval_util.checkpoint_runtime import (
    load_checkpoint,
    normalize_compiled_state_prefixes,
)
from src.eval_util.metric_names import (
    refset_tag,
    monitoring_section,
    metric_key,
    PXRD,
    MATCH_RATE,
)


def _monitor_target_seed(base_seed: int, step: int, resample: bool) -> int:
    return int(base_seed) + int(step) if resample else int(base_seed)


def _build_training_type_encoding(args):
    legacy_runtime_pca = bool(getattr(args, "legacy_runtime_pca", False))
    torch_num_threads = getattr(args, "torch_num_threads", None)
    if legacy_runtime_pca and torch_num_threads is None:
        raise SystemExit(
            "config error: legacy_runtime_pca requires run.torch_num_threads."
        )
    if torch_num_threads is not None:
        if int(torch_num_threads) <= 0:
            raise SystemExit("config error: run.torch_num_threads must be positive.")
        torch.set_num_threads(int(torch_num_threads))
    encoding = build_type_encoding(
        args.type_encoding, vz=VZ, legacy_runtime_pca=legacy_runtime_pca
    )
    digest = hashlib.sha256(
        encoding.z_to_enc.contiguous().numpy().tobytes()
    ).hexdigest()
    expected = getattr(args, "type_encoding_sha256_expected", None)
    if expected and digest != str(expected):
        raise SystemExit(
            f"config error: type encoding SHA mismatch: expected {expected}, got {digest}."
        )
    args.type_encoding_sha256 = digest
    print(
        f"[type-encoding] sha256={digest} torch_num_threads={torch.get_num_threads()} legacy_runtime_pca={legacy_runtime_pca}"
    )
    return encoding


def _build_eval_dataset(args, nmax, split):
    """RealPXRD eval split with d-I peaks attached and NO augmentation."""
    return make_token_dataset(
        args=args,
        nmax=nmax,
        root=args.data_root,
        split=split,
        augment_translate=False,
        augment_pxrd=False,
        has_split=True,
    )


def _build_pxrd_selection_candidate(
    *, step, epoch, selection_split, selection_refset, primary_k, metrics
):
    """Build a best-checkpoint candidate from validation PXRD metrics only."""
    if selection_split != "val":
        raise ValueError(
            f"PXRD checkpoint selection requires the validation split; got {selection_split!r}."
        )
    selection_value = metrics.get(f"pxrd_match_rate@{primary_k}")
    if selection_value is None or not math.isfinite(selection_value):
        return None
    return _BestCandidate(
        step=step,
        epoch=epoch,
        mode="csp",
        selection_source="primary",
        selector_name=f"{selection_refset}/{metric_key(PXRD, MATCH_RATE, primary_k)}",
        selector_value=float(selection_value),
        raw_metrics={
            f"{selection_refset}/{key}": value for key, value in metrics.items()
        },
    )


def _canonical_monitor_metrics(
    metrics: dict[str, float | None], *, element_set: bool
) -> dict[str, float | None]:
    """Map task-core keys to the public metric naming protocol."""
    if not element_set:
        return dict(metrics)
    canonical = {}
    for key, value in metrics.items():
        if key.startswith("pxrd_joint_"):
            key = "pxrd_elem_" + key.removeprefix("pxrd_joint_")
        elif key.startswith("pxrd_profile_"):
            key = "pxrd_elem_profile_" + key.removeprefix("pxrd_profile_")
        canonical[key] = value
    return canonical


_PRIOR_ARCH_KEYS = (
    "d_model",
    "n_heads",
    "n_layers",
    "type_encoding",
    "coord_n_freqs",
    "coord_embed_mode",
    "coord_head_mode",
    "coord_rff_dim",
    "coord_rff_sigma",
    "lattice_embed_mode",
    "lattice_repr",
    "lattice_rff_dim",
    "lattice_rff_sigma",
    "use_distance_bias",
    "use_edge_bias",
    "edge_bias_n_freqs",
    "edge_bias_hidden_dim",
    "edge_bias_n_rbf",
    "edge_bias_rbf_max",
    "pbc_radius",
    "use_noise_gate",
    "gem_per_layer",
)


def _init_from_prior_checkpoint(model, args, device) -> None:
    """Warm-start the conditioned model from an unconditional CSP prior.

    Loads every trunk/embedder/head weight from the prior (EMA weights by
    default — the eval-quality ones); only the PXRD encoder + fusion modules
    stay freshly initialized. With ``init_zero_pxrd_proj`` the encoder's
    ``out_proj`` is zeroed. Sum/mlp-sum fusion then reproduces the prior at
    step 0; in-context fusion remains a supported non-identical warm start
    because it inserts fresh context tokens into the trunk.
    """
    ckpt = load_checkpoint(args.init_checkpoint, map_location=device)
    normalize_compiled_state_prefixes(ckpt)
    prior_args = dict(ckpt.get("model_args", {}))
    if bool(prior_args.get("pxrd_cond", False)):
        raise SystemExit(
            "config error: pxrd.init_checkpoint is already PXRD-conditioned; expected an unconditional CSP prior."
        )
    mismatched = [
        (k, prior_args.get(k), getattr(args, k, None))
        for k in _PRIOR_ARCH_KEYS
        if str(prior_args.get(k)) != str(getattr(args, k, None))
    ]
    if mismatched:
        lines = ", ".join((f"{k}: prior={p!r} vs cfg={c!r}" for k, p, c in mismatched))
        raise SystemExit(f"config error: prior/config architecture mismatch — {lines}")
    state = dict(ckpt["model_state_dict"])
    used_ema = False
    if bool(args.init_use_ema) and "ema_state_dict" in ckpt:
        state.update(ckpt["ema_state_dict"])
        used_ema = True
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(
            f"config error: prior has keys absent from the model: {sorted(unexpected)[:5]}"
        )
    stray = [
        k
        for k in missing
        if not k.startswith(("pxrd_encoder.",)) and k not in {"pxrd_context_segment"}
    ]
    if stray:
        raise SystemExit(
            f"config error: prior is missing non-conditioning weights: {sorted(stray)[:5]}"
        )
    print(
        f"[init] loaded CSP prior {args.init_checkpoint} (step {ckpt.get('step')}, ema={used_ema}); {len(missing)} PXRD-module tensors stay fresh."
    )
    if bool(args.init_zero_pxrd_proj) and model.pxrd_encoder is not None:
        if not hasattr(model.pxrd_encoder, "out_proj"):
            print(
                f"[init] pxrd encoder has no out_proj to zero (pxrd_fusion={getattr(args, 'pxrd_fusion', None)}); skipping init_zero_pxrd_proj."
            )
            return
        torch.nn.init.zeros_(model.pxrd_encoder.out_proj.weight)
        torch.nn.init.zeros_(model.pxrd_encoder.out_proj.bias)
        print(
            "[init] zeroed pxrd_encoder.out_proj — sum-style paths may match the prior; in-context warm starts are supported but not bit-identical."
        )


def _load_resume_checkpoint(model, args, device):
    """Restore a compatible PXRD-conditioned model for further training."""
    ckpt = load_checkpoint(args.resume_checkpoint, map_location="cpu")
    normalize_compiled_state_prefixes(ckpt)
    resume_args = dict(ckpt.get("model_args", {}))
    if not bool(resume_args.get("pxrd_cond", False)):
        raise SystemExit("config error: resume checkpoint is not PXRD-conditioned.")
    if bool(resume_args.get("csp", True)) != bool(args.csp):
        raise SystemExit(
            "config error: resume checkpoint run.csp does not match current config."
        )
    checkpoint_encoding = str(
        ckpt.get("type_encoding", resume_args.get("type_encoding"))
    )
    if checkpoint_encoding != str(args.type_encoding):
        raise SystemExit("config error: resume checkpoint type encoding mismatch.")
    query_init = False
    if not query_init and "optimizer_state_dict" not in ckpt:
        raise SystemExit(
            "config error: resume checkpoint is missing optimizer_state_dict; use pxrd.transfer_checkpoint for model-only checkpoints."
        )
    if (
        not query_init
        and float(getattr(args, "ema_decay", 0.0)) > 0.0
        and ("ema_state_dict" not in ckpt)
    ):
        raise SystemExit(
            "config error: resume checkpoint is missing ema_state_dict while EMA is enabled; use pxrd.transfer_checkpoint for model-only checkpoints."
        )
    if query_init:
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        allowed_missing = {"atom_query_embed.weight"}
        if set(missing) != allowed_missing or unexpected:
            raise SystemExit(
                f"config error: query-init resume expected only a missing atom_query_embed.weight; missing={missing}, unexpected={unexpected}."
            )
        print(
            "[resume] initialized persistent atom queries while restoring all pre-existing model weights; optimizer/EMA state will be reset."
        )
    else:
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise SystemExit(
                f"config error: resume checkpoint architecture mismatch: {exc}"
            ) from exc
    resume_step = int(ckpt.get("step", 0))
    print(
        f"[resume] loaded conditioned checkpoint {args.resume_checkpoint} at global step {resume_step}."
    )
    return (ckpt, resume_step)


def _transfer_from_conditioned_checkpoint(model, args, device) -> None:
    """Load conditioned model weights while resetting all training state."""
    ckpt = load_checkpoint(args.transfer_checkpoint, map_location=device)
    normalize_compiled_state_prefixes(ckpt)
    source_args = dict(ckpt.get("model_args", {}))
    if not bool(source_args.get("pxrd_cond", False)):
        raise SystemExit(
            "config error: pxrd.transfer_checkpoint must be PXRD-conditioned."
        )
    if bool(source_args.get("csp", True)) != bool(args.csp):
        raise SystemExit(
            "config error: transfer checkpoint run.csp does not match current config."
        )
    checkpoint_encoding = str(
        ckpt.get("type_encoding", source_args.get("type_encoding"))
    )
    if checkpoint_encoding != str(args.type_encoding):
        raise SystemExit("config error: transfer checkpoint type encoding mismatch.")
    state = dict(ckpt["model_state_dict"])
    used_ema = False
    if bool(args.transfer_use_ema) and "ema_state_dict" in ckpt:
        state.update(ckpt["ema_state_dict"])
        used_ema = True
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise SystemExit(
            f"config error: transfer checkpoint architecture mismatch: {exc}"
        ) from exc
    print(
        f"[transfer] loaded conditioned checkpoint {args.transfer_checkpoint} (source step {ckpt.get('step')}, ema={used_ema}); optimizer, scheduler, EMA history, and global step start fresh."
    )


def run(args) -> None:
    pxrd_cond = bool(getattr(args, "pxrd_cond", False))
    if not pxrd_cond:
        raise SystemExit(
            "config error: train_bragglite requires PXRD conditioning (pxrd.pxrd_cond=true)."
        )
    if args.dataset_class not in (
        "real_pxrd",
        "mp20_tokens",
        "xrdiff_author_rruff_simulated",
        "shared_phase",
        "pxrd_bench",
    ):
        raise SystemExit(
            f"config error: PXRD-conditioned training needs data.dataset_class in {{real_pxrd, mp20_tokens, xrdiff_author_rruff_simulated, shared_phase, pxrd_bench}} (got {args.dataset_class}). real_pxrd reads d-I peaks from LMDB; mp20_tokens simulates them per structure via pymatgen XRDCalculator."
        )
    nmax = args.nmax
    distributed = initialize_distributed(args)
    output_error: list[str | None] = [None]
    if distributed.is_main:
        try:
            paths = TrainingPaths.create(args.output_dir)
            copy_hydra_metadata(paths)
        except OSError as exc:
            output_error[0] = str(exc)
    if distributed.enabled:
        torch.distributed.broadcast_object_list(output_error, src=0)
    if output_error[0] is not None:
        raise SystemExit(f"config error: {output_error[0]}")
    distributed.barrier()
    if not distributed.is_main:
        paths = TrainingPaths.from_root(args.output_dir)
    args.distributed_world_size = int(distributed.world_size)
    args.effective_local_batch_size = int(args.batch_size) * int(args.grad_accum_steps)
    args.global_batch_size = int(args.effective_local_batch_size) * int(
        distributed.world_size
    )
    process_seed = int(args.seed) + int(distributed.rank)
    seed_everything(process_seed, deterministic=args.deterministic)
    if distributed.is_main:
        print(f"[seed] base_seed={args.seed} deterministic={bool(args.deterministic)}")
        print(
            f"[batch] microbatch={args.batch_size} x accum={args.grad_accum_steps} x world_size={distributed.world_size} = global={args.global_batch_size}"
        )
    device = configure_device(
        args, local_rank=distributed.local_rank if distributed.enabled else None
    )
    data = setup_datasets(args=args, nmax=nmax)
    ds, val_ds = (data.ds, data.val_ds)
    has_split = data.has_split
    print(
        f"Dataset split sizes ({args.dataset_name}, nmax={nmax}):",
        f"train={len(ds)} val={len(val_ds)}",
    )
    loader, val_loader, data_iter, steps_per_epoch = build_dataloaders(
        args=args,
        ds=ds,
        val_ds=val_ds,
        device=device,
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    type_encoding = _build_training_type_encoding(args)
    checkpoint_model = build_bragglite(
        vars(args), type_dim=type_encoding.type_dim, device=device
    )
    num_params = sum((p.numel() for p in checkpoint_model.parameters()))
    if distributed.is_main:
        print(f"model parameters: {num_params}")
    checkpoint_modes = [
        name
        for name in ("init_checkpoint", "resume_checkpoint", "transfer_checkpoint")
        if getattr(args, name, None)
    ]
    if len(checkpoint_modes) > 1:
        raise SystemExit(
            "config error: use only one of pxrd.init_checkpoint, pxrd.resume_checkpoint, or pxrd.transfer_checkpoint."
        )
    if getattr(args, "init_checkpoint", None):
        _init_from_prior_checkpoint(checkpoint_model, args, device)
    if getattr(args, "transfer_checkpoint", None):
        _transfer_from_conditioned_checkpoint(checkpoint_model, args, device)
    resume_ckpt = None
    resume_training_state = None
    if getattr(args, "resume_checkpoint", None):
        resume_ckpt, _ = _load_resume_checkpoint(checkpoint_model, args, device)
    model = maybe_compile_model(checkpoint_model, args)
    model = wrap_distributed_model(model, context=distributed, args=args)
    resume_step = int(resume_ckpt.get("step", 0)) if resume_ckpt is not None else 0
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, args, start_step=resume_step
    )
    ema = (
        EMA(model, decay=args.ema_decay)
        if args.ema_decay > 0.0 and distributed.is_main
        else None
    )
    if resume_ckpt is not None and (not False):
        resume_training_state = resume_ckpt.get("training_state")
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        print("[resume] restored optimizer state.")
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        else:
            print(
                f"[resume] checkpoint has no scheduler state; aligned the fresh scheduler to global step {resume_step}."
            )
        if ema is not None:
            expected = set(ema.shadow)
            restored = set(resume_ckpt["ema_state_dict"])
            if expected != restored:
                raise SystemExit(
                    "config error: resume checkpoint EMA parameter set mismatch."
                )
            with torch.no_grad():
                for key, value in resume_ckpt["ema_state_dict"].items():
                    ema.shadow[key].copy_(value)
            print("[resume] restored EMA state.")
    elif resume_ckpt is not None:
        print(
            "[resume] query-init continuation uses fresh optimizer/scheduler/EMA state."
        )
    resume_ckpt = None
    bf16_dtype = torch.bfloat16 if args.bf16 else None
    output_dir = paths.root
    ckpt_dir = paths.checkpoints
    source_git_commit = git_commit(ROOT)
    artifacts_dir = paths.artifacts
    if data.train_subset_indices is not None and distributed.is_main:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        subset_path = artifacts_dir / "train_subset_indices.json"
        canonical_subset = int(getattr(args, "train_subset_size", 0) or 0) > 0
        subset_path.write_text(
            json.dumps(
                {
                    "split": "train",
                    "kind": "canonical" if canonical_subset else "overfit",
                    "index_space": "effective_train_after_exclusion",
                    "indices": [int(i) for i in data.train_subset_indices],
                    "size": len(data.train_subset_indices),
                    "seed": int(
                        getattr(
                            args,
                            "train_subset_seed"
                            if canonical_subset
                            else "overfit_subset_seed",
                            args.seed,
                        )
                    ),
                    "overfit_no_augment": bool(
                        getattr(args, "overfit_no_augment", False)
                    ),
                    "train_exclude_ids_path": getattr(
                        args, "train_exclude_ids_path", None
                    ),
                    "configured_batch_size": int(args.batch_size),
                },
                indent=2,
            )
        )
        print(f"[data] wrote train subset indices: {subset_path}")
    dataset_tag = (
        f"{args.dataset_class}_{args.real_pxrd_variant}"
        if args.dataset_class == "real_pxrd"
        else args.dataset_class
    )
    run_handle = None
    if distributed.is_main:
        run_handle = init_wandb(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=[args.dataset_name, dataset_tag, "bragglite"],
            enabled=not args.no_wandb,
            dir=str(output_dir),
        )
        log_summary(
            {
                "dataset/train_size": len(ds),
                "dataset/val_size": len(val_ds),
                "model/num_parameters": num_params,
                "training/world_size": distributed.world_size,
                "training/microbatch_size": int(args.batch_size),
                "training/grad_accum_steps": int(args.grad_accum_steps),
                "training/local_batch_size": int(args.effective_local_batch_size),
                "training/global_batch_size": int(args.global_batch_size),
            },
            enabled=not args.no_wandb,
        )
    weights = {
        "A": 0.0 if args.csp else float(args.loss_weights[0]),
        "F": float(args.loss_weights[1]),
        "Y": float(args.loss_weights[2]),
    }
    if not args.csp:
        print(
            "[cond] PXRD element-set mode: atom types, coords, and lattice are generated."
        )

    def build_ckpt(step_value: int):
        return build_train_checkpoint(
            model=checkpoint_model,
            args=args,
            type_encoding=type_encoding,
            ema=ema,
            step_value=step_value,
            git_commit_value=source_git_commit,
        )

    selection_eval_ds = None
    selection_target_indices = None
    selection_refset = "eval_selection"
    report_eval_ds = None
    report_target_indices = None
    report_refset = "eval_report"
    monitor_use_ema = args.monitor_mode == "ema"
    if args.monitor_every > 0:
        selection_split = args.monitor_selection_split
        report_split = args.monitor_split
        selection_refset = refset_tag(
            args.dataset_name, getattr(args, "real_pxrd_variant", None), selection_split
        )
        report_refset = refset_tag(
            args.dataset_name, getattr(args, "real_pxrd_variant", None), report_split
        )
        if args.csp or selection_split != "train":
            selection_eval_ds = _build_eval_dataset(args, nmax, selection_split)
            selection_target_indices = select_eval_targets(
                selection_eval_ds,
                args.monitor_num_samples,
                args.monitor_seed,
                target_ids_path=getattr(args, "monitor_ids_path", None),
            )
        else:
            if data.train_subset_indices is None:
                raise SystemExit(
                    "config error: joint periodic evaluation requires an explicit data.overfit_subset_size or data.overfit_subset_indices."
                )
            selection_eval_ds = data.ref_ds
            target_count = len(selection_eval_ds)
            if int(args.monitor_num_samples) > 0:
                target_count = min(target_count, int(args.monitor_num_samples))
            selection_target_indices = torch.arange(target_count, dtype=torch.long)
        print(
            f"[pxrd-eval] selection: {len(selection_target_indices)} targets from {selection_split} split; match@{args.monitor_topk} every {args.monitor_every} steps."
        )
        if report_split == selection_split:
            report_eval_ds = selection_eval_ds
            report_target_indices = selection_target_indices
        else:
            report_eval_ds = _build_eval_dataset(args, nmax, report_split)
            report_target_indices = select_eval_targets(
                report_eval_ds, args.monitor_num_samples, args.monitor_seed
            )
            print(
                f"[pxrd-eval] reporting only: {len(report_target_indices)} targets from {report_split} split; these metrics never select a checkpoint."
            )
    external_monitors = load_external_monitors(args) if distributed.is_main else []

    def sampling_evaluation_hook(step, best_ckpt_state):
        if args.monitor_every <= 0 or step % args.monitor_every != 0:
            return False
        if args.monitor_every > 0 and step % args.monitor_every == 0:
            target_seed = _monitor_target_seed(
                args.monitor_seed,
                step,
                bool(getattr(args, "monitor_resample_targets", False)),
            )
            current_selection_target_indices = select_eval_targets(
                selection_eval_ds,
                args.monitor_num_samples,
                target_seed,
                target_ids_path=getattr(args, "monitor_ids_path", None),
            )
            was_training = checkpoint_model.training
            checkpoint_model.eval()
            with maybe_use_ema(checkpoint_model, ema, monitor_use_ema):
                if args.csp:
                    selection_metrics, selection_per_target = run_pxrd_match_eval(
                        model=checkpoint_model,
                        eval_ds=selection_eval_ds,
                        target_indices=current_selection_target_indices,
                        type_encoding=type_encoding,
                        args=args,
                        device=device,
                        return_per_target=True,
                    )
                else:
                    targets = [
                        selection_eval_ds[int(i)]
                        for i in current_selection_target_indices
                    ]
                    topk = sorted({int(k) for k in args.monitor_topk})
                    selection_result = evaluate_joint_model(
                        model=checkpoint_model,
                        targets=targets,
                        type_encoding=type_encoding,
                        model_args=vars(args),
                        eval_args=args,
                        device=device,
                        seed=int(args.monitor_seed),
                        chunk_size=max(1, int(args.monitor_chunk_size or len(targets))),
                        num_candidates=max(int(args.monitor_num_candidates), topk[-1]),
                        topk=topk,
                        matcher_profile="strict",
                    )
                    selection_metrics = selection_result["metrics"]
                    selection_per_target = selection_result.get("records")
                external_metrics = run_external_monitors(
                    external_monitors,
                    model=checkpoint_model,
                    type_encoding=type_encoding,
                    args=args,
                    device=device,
                    step=step,
                    artifacts_dir=artifacts_dir,
                    use_ema=monitor_use_ema,
                )
            selection_metrics = _canonical_monitor_metrics(
                selection_metrics, element_set=not args.csp
            )
            monitor_tag = monitoring_section(
                args.dataset_name,
                getattr(args, "real_pxrd_variant", None),
                selection_split,
                weights="ema" if monitor_use_ema else "raw",
            )
            log_metrics(
                {
                    **{f"{monitor_tag}/{k}": v for k, v in selection_metrics.items()},
                    **external_metrics,
                    "epoch": math.ceil(step / steps_per_epoch),
                },
                step=step,
                enabled=not args.no_wandb,
            )
            primary_k = args.monitor_topk[0]
            if selection_per_target is not None:
                record = {
                    "step": int(step),
                    "monitor_seed": int(args.monitor_seed),
                    "target_seed": target_seed,
                    "use_ema": monitor_use_ema,
                    "split": selection_split,
                    "num_candidates": int(args.monitor_num_candidates),
                    "topk": [int(k) for k in args.monitor_topk],
                    "targets": selection_per_target,
                }
                with (artifacts_dir / "periodic_eval_per_target.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
            if not args.csp:
                match1 = selection_metrics.get(f"pxrd_elem_match_rate@{primary_k}")
                formula1 = float(
                    selection_metrics.get(f"pxrd_elem_formula_recall@{primary_k}", 0.0)
                )
                collision1 = float(
                    selection_metrics.get("pxrd_elem_collision_rate@1", 1.0)
                )
                record = {
                    "step": int(step),
                    "monitor_seed": int(args.monitor_seed),
                    "target_seed": target_seed,
                    "use_ema": monitor_use_ema,
                    "target_indices": [
                        int(i) for i in current_selection_target_indices
                    ],
                    "split": selection_split,
                    "metrics": selection_metrics,
                }
                with (artifacts_dir / "periodic_eval.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                if (
                    match1 is not None
                    and float(match1) >= 1.0
                    and (formula1 >= 1.0)
                    and (collision1 <= 0.0)
                    and (selection_split == "train")
                    and (not (ckpt_dir / "success.pt").exists())
                ):
                    torch.save(build_ckpt(step), ckpt_dir / "success.pt")
                    (ckpt_dir / "success_meta.json").write_text(
                        json.dumps(record, indent=2)
                    )
                    print(f"[overfit] strict success reached at global step {step}.")
            if args.best_ckpt:
                if args.csp:
                    cand = _build_pxrd_selection_candidate(
                        step=step,
                        epoch=math.ceil(step / steps_per_epoch),
                        selection_split=selection_split,
                        selection_refset=selection_refset,
                        primary_k=primary_k,
                        metrics=selection_metrics,
                    )
                else:
                    match1 = selection_metrics.get(f"pxrd_elem_match_rate@{primary_k}")
                    cand = None
                    if match1 is not None and math.isfinite(match1):
                        formula_tiebreak = float(
                            selection_metrics.get(
                                f"pxrd_elem_formula_recall@{primary_k}", 0.0
                            )
                        )
                        collision_tiebreak = float(
                            selection_metrics.get("pxrd_elem_collision_rate@1", 0.0)
                        )
                        cand = _BestCandidate(
                            step=step,
                            epoch=math.ceil(step / steps_per_epoch),
                            mode="dng",
                            selection_source="primary",
                            selector_name=f"{monitor_tag}/pxrd_elem_match_rate@{primary_k}",
                            selector_value=float(match1)
                            + 0.001 * formula_tiebreak
                            - 0.0001 * collision_tiebreak,
                            raw_metrics={
                                f"{monitor_tag}/{key}": value
                                for key, value in selection_metrics.items()
                            },
                        )
                if cand is not None:
                    maybe_update_best_ckpt(
                        state=best_ckpt_state,
                        candidate=cand,
                        maximize=True,
                        ckpt_dir=ckpt_dir,
                        build_ckpt_fn=build_ckpt,
                        enabled=True,
                    )
            if args.csp and report_refset != selection_refset:
                with maybe_use_ema(checkpoint_model, ema, monitor_use_ema):
                    report_metrics = run_pxrd_match_eval(
                        model=checkpoint_model,
                        eval_ds=report_eval_ds,
                        target_indices=report_target_indices,
                        type_encoding=type_encoding,
                        args=args,
                        device=device,
                    )
                report_metrics = _canonical_monitor_metrics(
                    report_metrics, element_set=False
                )
                report_tag = monitoring_section(
                    args.dataset_name,
                    getattr(args, "real_pxrd_variant", None),
                    report_split,
                    weights="ema" if monitor_use_ema else "raw",
                )
                log_metrics(
                    {
                        **{
                            f"{report_tag}/{key}": value
                            for key, value in report_metrics.items()
                        },
                        "epoch": math.ceil(step / steps_per_epoch),
                    },
                    step=step,
                    enabled=not args.no_wandb,
                )
            if was_training:
                checkpoint_model.train()
            return True
        return False

    try:
        run_training(
            TrainingRunContext(
                args=args,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                type_encoding=type_encoding,
                weights=weights,
                device=device,
                bf16_dtype=bf16_dtype,
                data_iter=data_iter,
                val_loader=val_loader,
                has_split=has_split,
                steps_per_epoch=steps_per_epoch,
                ckpt_dir=ckpt_dir,
                build_ckpt=build_ckpt,
                wandb_enabled=not args.no_wandb and distributed.is_main,
                run_handle=run_handle,
                progress_desc="train-pxrd",
                start_step=resume_step,
                distributed=distributed,
                eval_model=checkpoint_model,
                resume_state=resume_training_state,
            ),
            sampling_evaluation_hook=sampling_evaluation_hook,
            sampling_evaluation_due=lambda step: (
                args.monitor_every > 0 and step % args.monitor_every == 0
            ),
        )
    finally:
        distributed.close()


@hydra.main(version_base=None, config_path="../../configs", config_name="train_pxrd")
def main(cfg: DictConfig) -> None:
    from src.training.config import validate_cfg

    args = validate_cfg(cfg)
    run(args)


if __name__ == "__main__":
    main()
