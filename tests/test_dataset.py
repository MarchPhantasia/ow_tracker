from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import yaml

from src.head_tracker.dataset import normalize_names, prepare_yolo_data


def _write_tiny_dataset(root: Path) -> Path:
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images" / f"{split}.jpg").write_bytes(b"fake")
        (root / split / "labels" / f"{split}.txt").write_text(
            "1 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "nc: 2\n"
        "names: ['Ally', 'Enemy']\n",
        encoding="utf-8",
    )
    return data_yaml


def test_normalize_names_requires_ally_enemy_order():
    assert normalize_names(["Ally", "Enemy"]) == {0: "ally", 1: "enemy"}


def test_prepare_yolo_data_repairs_roboflow_relative_paths():
    root = Path("tmp") / "test-artifacts" / uuid4().hex
    data_yaml = _write_tiny_dataset(root / "datasets")

    prepared = prepare_yolo_data(data_yaml, root / "runs" / "dataset")
    generated = yaml.safe_load(prepared.data_yaml.read_text(encoding="utf-8"))

    assert Path(generated["train"]).is_dir()
    assert Path(generated["val"]).is_dir()
    assert Path(generated["test"]).is_dir()
    assert generated["names"] == ["ally", "enemy"]
    assert prepared.splits["train"].instances_by_class == {0: 0, 1: 1}


def test_prepare_yolo_data_use_all_data_writes_train_list():
    root = Path("tmp") / "test-artifacts" / uuid4().hex
    data_yaml = _write_tiny_dataset(root / "datasets")

    prepared = prepare_yolo_data(
        data_yaml,
        root / "runs" / "dataset",
        use_all_data=True,
    )
    generated = yaml.safe_load(prepared.data_yaml.read_text(encoding="utf-8"))
    train_list = Path(generated["train"])

    assert train_list.is_file()
    assert len(train_list.read_text(encoding="utf-8").splitlines()) == 3
    assert prepared.train_image_count == 3
