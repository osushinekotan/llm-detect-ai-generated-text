import hydra
from kaggle import KaggleApi
from omegaconf import DictConfig

from src.utils.kaggle_utils import Deploy


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    client = KaggleApi()
    client.authenticate()

    deploy = Deploy(cfg=cfg, client=client)

    # deploy trained model weights
    deploy.push_output(
        ignore_patterns=[
            ".git",
            "__pycache__",
            ".hydra",
            "checkpoints",
            "*.log",
            "csv",
            "wandb",
            "notebooks",
            "*.pkl",
            "*.csv",
            "*.ipynb",
        ]
    )

    deploy.push_code()
    deploy.push_huguingface_model()


if __name__ == "__main__":
    main()
