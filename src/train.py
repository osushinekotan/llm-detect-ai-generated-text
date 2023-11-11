import logging
from pathlib import Path

import hydra
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run experiment notebook."""

    env = {"OVERRIDES": cfg.experiment_name}

    exp_notebooks_dir = Path(cfg.paths.notebooks_dir) / "experiment"
    logger.info(f"Overrides: {env['OVERRIDES']}, Notebook: {cfg.notebook}")
    compiled_notebook_path = Path(cfg.paths.output_dir) / f"{cfg.notebook}.ipynb"

    logger.info(f"{env['OVERRIDES']}, Notebook: {cfg.notebook}")

    with open(exp_notebooks_dir / f"{cfg.notebook}.ipynb") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=None, allow_errors=True)
    resources = {"metadata": {"path": exp_notebooks_dir.as_posix()}, "env": env}

    try:
        ep.preprocess(nb, resources)
        logger.info("Finished running notebook successfully! 🎉")
    except Exception as e:
        logger.error(f"Failed to run notebook 😭: {e}")
        raise

    with open(compiled_notebook_path, "wt") as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    main()
