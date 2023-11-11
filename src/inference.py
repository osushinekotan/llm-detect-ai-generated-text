import logging
import os
import subprocess
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run experiment notebook."""

    env = os.environ.copy()
    env["OVERRIDES"] = cfg.experiment_name

    exp_notebooks_dir = Path(cfg.paths.notebooks_dir) / "inference"

    logger = logging.getLogger(__name__)
    logger.info(f"Overrides: {env['OVERRIDES']}, Notebook: {cfg.notebook}")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file_path = Path(tmpdir) / "temp_notebook.ipynb"

        command = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output",
            temp_file_path.as_posix(),
            (exp_notebooks_dir / f"{cfg.notebook}.ipynb").as_posix(),
        ]

        try:
            subprocess.run(command, env=env, check=True)
            logger.info("Finished running notebook successfully! 🎉")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run notebook 😭: {e}")
            raise


if __name__ == "__main__":
    main()
