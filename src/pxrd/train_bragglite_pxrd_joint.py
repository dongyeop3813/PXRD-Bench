"""PXRD-conditioned joint atom-type/structure element-set training entrypoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.pxrd.train_bragglite import run
from src.training.config import validate_cfg


@hydra.main(version_base=None, config_path="../../configs", config_name="train_pxrd_element_set")
def main(cfg: DictConfig) -> None:
    args = validate_cfg(cfg)
    run(args)


if __name__ == "__main__":
    main()
