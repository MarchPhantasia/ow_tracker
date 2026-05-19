from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EpochRecommendation:
    epoch: int
    metric: str
    value: float
    results_csv: Path


def recommend_epoch(
    results_csv: str | Path,
    metric: str = "metrics/mAP50-95(B)",
) -> EpochRecommendation:
    path = Path(results_csv)
    if not path.is_file():
        raise FileNotFoundError(path)

    best_epoch: int | None = None
    best_value: float | None = None

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {key.strip(): value for key, value in row.items() if key is not None}
            if metric not in normalized:
                available = ", ".join(sorted(normalized))
                raise KeyError(f"metric {metric!r} not found. Available columns: {available}")
            try:
                epoch = int(float(normalized.get("epoch", "0")))
                value = float(normalized[metric])
            except ValueError:
                continue
            if best_value is None or value > best_value:
                best_epoch = epoch
                best_value = value

    if best_epoch is None or best_value is None:
        raise ValueError(f"no valid metric values found in {path}")

    return EpochRecommendation(
        epoch=best_epoch,
        metric=metric,
        value=best_value,
        results_csv=path,
    )


def copy_best_weight(run_dir: str | Path, output_path: str | Path = "models/best.pt") -> Path:
    run_dir = Path(run_dir)
    source = run_dir / "weights" / "best.pt"
    if not source.is_file():
        raise FileNotFoundError(source)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    return output_path
