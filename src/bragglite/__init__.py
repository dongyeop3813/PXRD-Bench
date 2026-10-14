"""Bragglite model exports, loaded only when a runtime asks for them."""

from __future__ import annotations

from typing import Any

__all__ = ["BraggliteModel", "MP20EDMModel", "mod1"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from .model import BraggliteModel, MP20EDMModel, mod1

    return {
        "BraggliteModel": BraggliteModel,
        "MP20EDMModel": MP20EDMModel,
        "mod1": mod1,
    }[name]
