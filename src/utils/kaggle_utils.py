import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from fnmatch import fnmatch
from functools import cached_property
from pathlib import Path

from kaggle import KaggleApi
from omegaconf import DictConfig
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def download_kaggle_competition_dataset(
    client: "KaggleApi",
    competition: str,
    out_dir: Path,
    force: bool = False,
) -> None:
    zipfile_path = out_dir / f"{competition}.zip"
    zipfile_path.parent.mkdir(exist_ok=True, parents=True)

    if not zipfile_path.is_file() or force:
        with tempfile.TemporaryDirectory() as tempdir:
            # NOTE : escape error when use gcsfuse
            client.competition_download_files(
                competition=competition,
                path=tempdir,
                quiet=False,
            )
            # TODO : fix error when use gcsfuse
            try:
                shutil.move(Path(tempdir) / f"{competition}.zip", zipfile_path)
            except OSError as e:
                logger.warning(f"{e} : please ignore error & warnings 😢")
        subprocess.run(["unzip", "-o", "-q", zipfile_path, "-d", out_dir])
    else:
        logger.info("Dataset already exists.")


def download_kaggle_datasets(
    client: "KaggleApi",
    datasets: list[str],
    out_dir: Path,
    force: bool = False,
) -> None:
    for dataset in datasets:
        zipfile_path = out_dir / dataset / f"{dataset.split('/')[1]}.zip"
        path = out_dir / dataset  # dataset: [owner]/[dataset-name]
        path.mkdir(exist_ok=True, parents=True)

        if not zipfile_path.is_file() or force:
            logger.info(f"Downloading dataset: {dataset}")
            with tempfile.TemporaryDirectory() as tempdir:
                tmp_path = Path(tempdir) / dataset
                client.dataset_download_files(
                    dataset=dataset,
                    quiet=False,
                    unzip=False,
                    path=tmp_path,
                    force=force,
                )
                # TODO : fix error when use gcsfuse
                try:
                    shutil.move(tmp_path / f"{dataset.split('/')[1]}.zip", zipfile_path)
                except OSError as e:
                    logger.warning(f"{e} : please ignore error & warnings 😢")

            subprocess.run(["unzip", "-o", "-q", zipfile_path, "-d", path])
        else:
            logger.info(f"Dataset ({dataset}) already exists.")


class Deploy:
    def __init__(
        self,
        cfg: DictConfig,
        client: "KaggleApi",
    ):
        self.cfg = cfg
        self.client = client

    def push_output(self, ignore_patterns: list[str] = [".git", "__pycache__"]) -> None:
        # model and predictions
        dataset_name = self.cfg.meta.alias + "-" + re.sub(r"[/_=]", "-", self.cfg.experiment_name)
        metadata = make_dataset_metadata(dataset_name=dataset_name)

        # if exist dataset, stop pushing
        if exist_dataset(
            dataset=f'{os.getenv("KAGGLE_USERNAME")}/{dataset_name}',
            existing_dataset=self.existing_dataset,
        ):
            logger.warning(f"{dataset_name} already exist!! Stop pushing. 🛑")
            return

        with tempfile.TemporaryDirectory() as tempdir:
            dst_dir = Path(tempdir) / dataset_name

            copytree(
                src=self.cfg.paths.output_dir,
                dst=dst_dir,
                ignore_patterns=ignore_patterns,
            )
            self._display_tree(dst_dir=dst_dir)

            with open(Path(dst_dir) / "dataset-metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
            self.client.dataset_create_new(
                folder=dst_dir,
                public=False,
                quiet=False,
                dir_mode="zip",
            )

    def push_huguingface_model(self) -> None:
        model_name = self.cfg.model_name
        dataset_name = re.sub(r"[/_]", "-", model_name)

        # if exist dataset, stop pushing
        if exist_dataset(
            dataset=f'{os.getenv("KAGGLE_USERNAME")}/{dataset_name}',
            existing_dataset=self.existing_dataset,
        ):
            logger.warning(f"{dataset_name} already exist!! Stop pushing. 🛑")
            return

        # pretrained tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        metadata = make_dataset_metadata(dataset_name=dataset_name)

        with tempfile.TemporaryDirectory() as tempdir:
            config.save_pretrained(tempdir)
            tokenizer.save_pretrained(tempdir)
            with open(Path(tempdir) / "dataset-metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            self.client.dataset_create_new(
                folder=tempdir,
                public=True,
                quiet=False,
                dir_mode="zip",
            )

    def push_code(self) -> None:
        dataset_name = "my-code-" + re.sub(r"[/_=]", "-", self.cfg.meta.competition)
        metadata = make_dataset_metadata(dataset_name=dataset_name)

        with tempfile.TemporaryDirectory() as tempdir:
            dst_dir = Path(tempdir) / dataset_name

            # for src directory
            dst_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(Path(self.cfg.paths.root_dir) / "README.md", dst_dir)

            copytree(
                src=(Path(self.cfg.paths.root_dir) / "src").as_posix(),
                dst=str(dst_dir / "src"),
                ignore_patterns=[".git", "__pycache__"],
            )
            copytree(
                src=(Path(self.cfg.paths.root_dir) / "configs").as_posix(),
                dst=str(dst_dir / "configs"),
                ignore_patterns=[".git", "__pycache__"],
            )
            copytree(
                src=(Path(self.cfg.paths.root_dir) / "notebooks").as_posix(),
                dst=str(dst_dir / "notebooks"),
                ignore_patterns=[".git", "__pycache__"],
            )
            self._display_tree(dst_dir=dst_dir)

            with open(dst_dir / "dataset-metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            # update dataset if dataset already exist
            if exist_dataset(
                dataset=f'{os.getenv("KAGGLE_USERNAME")}/{dataset_name}',
                existing_dataset=self.existing_dataset,
            ):
                logger.info("update code")
                self.client.dataset_create_version(
                    folder=dst_dir,
                    version_notes="latest",
                    quiet=False,
                    convert_to_csv=False,
                    delete_old_versions=True,
                    dir_mode="zip",
                )
            else:
                logger.info("create dataset of code")
                self.client.dataset_create_new(folder=dst_dir, public=False, quiet=False, dir_mode="zip")

    @cached_property
    def existing_dataset(self) -> list:
        return self.client.dataset_list(user=os.getenv("KAGGLE_USERNAME"))

    @staticmethod
    def _display_tree(dst_dir: Path) -> None:
        logger.info(f"dst_dir={dst_dir}\ntree")
        display_tree(dst_dir)


def exist_dataset(dataset: str, existing_dataset: list) -> bool:
    for ds in existing_dataset:
        if str(ds) == dataset:
            return True
    return False


def make_dataset_metadata(dataset_name: str) -> dict:
    dataset_metadata = {}
    dataset_metadata["id"] = f'{os.environ["KAGGLE_USERNAME"]}/{dataset_name}'
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]  # type: ignore
    dataset_metadata["title"] = dataset_name
    return dataset_metadata


def copytree(src: str, dst: str, ignore_patterns: list = []) -> None:
    if not os.path.exists(dst):
        os.makedirs(dst)

    for item in os.listdir(src):
        if any(fnmatch(item, pattern) for pattern in ignore_patterns):
            continue

        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, ignore_patterns)
        else:
            shutil.copy2(s, d)


def display_tree(directory: Path, file_prefix: str = "") -> None:
    entries = list(directory.iterdir())
    file_count = len(entries)

    for i, entry in enumerate(sorted(entries, key=lambda x: x.name)):
        if i == file_count - 1:
            prefix = "└── "
            next_prefix = file_prefix + "    "
        else:
            prefix = "├── "
            next_prefix = file_prefix + "│   "

        line = file_prefix + prefix + entry.name
        print(line)

        if entry.is_dir():
            display_tree(entry, next_prefix)
