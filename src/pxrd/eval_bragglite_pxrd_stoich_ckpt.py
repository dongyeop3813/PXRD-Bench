"""Hydra entrypoint for PXRD evaluation with oracle stoichiometry."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.pxrd.eval_hydra_runtime import print_metrics, run_pxrd_hydra


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="eval_pxrd_stoich",
)
def main(cfg: DictConfig) -> None:
    print_metrics(run_pxrd_hydra("stoich", cfg))


if __name__ == "__main__":
    main()
