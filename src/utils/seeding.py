import hashlib
import json
import os
import random
from contextlib import contextmanager

import numpy as np
import torch


def capture_rng_state() -> dict:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def stable_seed(*parts: object) -> int:
    """Return a process- and partition-independent PyTorch seed."""
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@contextmanager
def isolated_rng(seed: int | None = None):
    """Restore caller RNGs; do not initialize CUDA just to capture its state."""
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            torch.manual_seed(int(seed))
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Determinism can reduce performance, so keep it opt-in.
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
