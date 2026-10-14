"""Shared nested Hydra configuration adapter for evaluation entrypoints."""

from __future__ import annotations

import argparse
from typing import Any

from omegaconf import DictConfig, OmegaConf

SECTIONS = ("checkpoint", "data", "sampling", "scoring", "runtime", "run")


def evaluation_namespace(cfg: DictConfig) -> argparse.Namespace:
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("evaluation config must resolve to a mapping")
    payload.pop("hydra", None)
    flat: dict[str, Any] = {}
    for section in SECTIONS:
        values = payload.pop(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"evaluation section {section!r} must be a mapping")
        for key, value in values.items():
            if key in flat:
                raise ValueError(f"duplicate evaluation config leaf: {key}")
            flat[key] = value
    for key, value in payload.items():
        if key in flat:
            raise ValueError(f"duplicate evaluation config leaf: {key}")
        flat[key] = value
    return argparse.Namespace(**flat)
