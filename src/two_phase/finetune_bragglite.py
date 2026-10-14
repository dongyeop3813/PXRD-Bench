"""Fine-tune Bragglite with oracle phase PXRD and shared mixture elements."""

import hydra
from omegaconf import DictConfig
from src.pxrd.train_bragglite import run
from src.training.config import validate_cfg


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="two_phase/finetune_shared_phase",
)
def main(config: DictConfig) -> None:
    run(validate_cfg(config))


if __name__ == "__main__":
    main()
