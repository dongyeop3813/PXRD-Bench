"""Hydra entrypoint for PXRD evaluation with oracle element set and atom count."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.pxrd.eval_hydra_runtime import print_metrics, run_pxrd_hydra


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="eval_pxrd_element_set",
)
def main(cfg: DictConfig) -> None:
    print_metrics(run_pxrd_hydra("element_set", cfg))


if __name__ == "__main__":
    main()
