"""Training checkpoint assembly.

Extracted from ``train_bragglite.py``. ``model_args = vars(args)`` preserves the
flat-key contract that the sample/eval/pxrd entrypoints rely on when they read
``ckpt["model_args"].get(<name>)`` (see ``src/bragglite/factory.py``).
"""

from __future__ import annotations

from pathlib import Path


def build_train_checkpoint(
    *,
    model,
    args,
    type_encoding,
    ema,
    step_value,
    git_commit_value: str | None = None,
):
    """Assemble the checkpoint dict (includes EMA state when available)."""
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_args": vars(args),
        "step": step_value,
        "type_encoding": type_encoding.name,
        "type_dim": type_encoding.type_dim,
        "resolved_config": vars(args),
        "training_run": str(Path(args.output_dir).resolve()),
        "git_commit": git_commit_value,
    }
    if ema is not None:
        ckpt["ema_state_dict"] = ema.state_dict()
        ckpt["ema_decay"] = args.ema_decay
    return ckpt
