from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CANONICAL_NAMES = ("ally", "enemy")


@dataclass(frozen=True)
class SplitSummary:
    name: str
    images_dir: Path
    labels_dir: Path
    image_count: int
    label_count: int
    missing_label_count: int
    empty_label_count: int
    instances_by_class: dict[int, int]
    invalid_label_count: int


@dataclass(frozen=True)
class PreparedDataset:
    source_yaml: Path
    data_yaml: Path
    names: dict[int, str]
    splits: dict[str, SplitSummary]
    train_image_count: int


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def normalize_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, list):
        names = {idx: str(name).strip().lower() for idx, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(idx): str(name).strip().lower() for idx, name in raw_names.items()}
    else:
        raise ValueError("data.yaml must define names as a list or mapping")

    if len(names) != 2:
        raise ValueError(f"expected exactly 2 classes {CANONICAL_NAMES}, got {names}")
    if [names.get(0), names.get(1)] != list(CANONICAL_NAMES):
        raise ValueError(f"expected class order 0=ally, 1=enemy, got {names}")
    return names


def find_dataset_yaml(data: str | Path = "datasets/data.yaml") -> Path:
    path = Path(data)
    if path.is_file():
        return path
    if path.is_dir():
        direct = path / "data.yaml"
        if direct.is_file():
            return direct
        matches = sorted(path.glob("*/data.yaml"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"could not find data.yaml at {path}")


def prepare_yolo_data(
    source_yaml: str | Path = "datasets/data.yaml",
    output_dir: str | Path = "runs/datasets/ally_enemy",
    *,
    use_all_data: bool = False,
) -> PreparedDataset:
    source_yaml = find_dataset_yaml(source_yaml).resolve()
    raw = read_yaml(source_yaml)
    names = normalize_names(raw.get("names"))

    split_dirs = {
        split: _resolve_images_dir(source_yaml.parent, raw, split)
        for split in ("train", "val", "test")
    }
    summaries = {
        split: inspect_split(split, images_dir, names)
        for split, images_dir in split_dirs.items()
        if images_dir is not None
    }

    if "train" not in summaries:
        raise FileNotFoundError("dataset must contain a train/images split")
    if "val" not in summaries:
        raise FileNotFoundError("dataset must contain a valid/images or val/images split")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_value: str
    train_image_count = summaries["train"].image_count
    if use_all_data:
        train_images = []
        for split in ("train", "val", "test"):
            if split in split_dirs and split_dirs[split] is not None:
                train_images.extend(_list_images(split_dirs[split]))
        train_list = output_dir / "train_all.txt"
        train_list.write_text(
            "\n".join(path.resolve().as_posix() for path in train_images) + "\n",
            encoding="utf-8",
        )
        train_value = train_list.resolve().as_posix()
        train_image_count = len(train_images)
    else:
        train_value = summaries["train"].images_dir.resolve().as_posix()

    generated = {
        "train": train_value,
        "val": summaries["val"].images_dir.resolve().as_posix(),
        "nc": 2,
        "names": [names[0], names[1]],
    }
    if "test" in summaries:
        generated["test"] = summaries["test"].images_dir.resolve().as_posix()

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")

    return PreparedDataset(
        source_yaml=source_yaml,
        data_yaml=data_yaml.resolve(),
        names=names,
        splits=summaries,
        train_image_count=train_image_count,
    )


def inspect_dataset(source_yaml: str | Path = "datasets/data.yaml") -> PreparedDataset:
    return prepare_yolo_data(source_yaml, "runs/datasets/inspect", use_all_data=False)


def inspect_split(name: str, images_dir: Path, names: dict[int, str]) -> SplitSummary:
    labels_dir = images_dir.parent / "labels"
    images = _list_images(images_dir)
    labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []

    instances_by_class = {class_id: 0 for class_id in names}
    missing_label_count = 0
    empty_label_count = 0
    invalid_label_count = 0

    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_label_count += 1
            continue
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            empty_label_count += 1
            continue
        for line in text.splitlines():
            parsed = _parse_label_line(line, names)
            if parsed is None:
                invalid_label_count += 1
            else:
                instances_by_class[parsed] = instances_by_class.get(parsed, 0) + 1

    return SplitSummary(
        name=name,
        images_dir=images_dir.resolve(),
        labels_dir=labels_dir.resolve(),
        image_count=len(images),
        label_count=len(labels),
        missing_label_count=missing_label_count,
        empty_label_count=empty_label_count,
        instances_by_class=instances_by_class,
        invalid_label_count=invalid_label_count,
    )


def _resolve_images_dir(yaml_dir: Path, raw: dict[str, Any], split: str) -> Path | None:
    key = "val" if split == "val" else split
    fallback_name = "valid" if split == "val" else split
    declared = raw.get(key)

    candidates: list[Path] = []
    if isinstance(declared, str):
        declared_path = Path(declared)
        candidates.append(declared_path if declared_path.is_absolute() else yaml_dir / declared_path)
    candidates.extend(
        [
            yaml_dir / fallback_name / "images",
            yaml_dir / split / "images",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _parse_label_line(line: str, names: dict[int, str]) -> int | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    if class_id not in names:
        return None
    if any(value < 0.0 or value > 1.0 for value in coords):
        return None
    return class_id
