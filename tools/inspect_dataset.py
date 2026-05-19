from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.head_tracker.dataset import prepare_yolo_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and normalize an ally/enemy YOLO dataset.")
    parser.add_argument("--data", default="datasets/data.yaml")
    parser.add_argument("--output-dir", default="runs/datasets/inspect")
    parser.add_argument("--use-all-data", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared = prepare_yolo_data(args.data, args.output_dir, use_all_data=args.use_all_data)
    print(f"Generated data yaml: {prepared.data_yaml}")
    print(f"Class names: {prepared.names}")
    print(f"Train images used: {prepared.train_image_count}")
    for split, summary in prepared.splits.items():
        counts = {
            prepared.names[class_id]: count
            for class_id, count in summary.instances_by_class.items()
        }
        print(
            f"{split}: images={summary.image_count}, labels={summary.label_count}, "
            f"missing_labels={summary.missing_label_count}, empty_labels={summary.empty_label_count}, "
            f"invalid_labels={summary.invalid_label_count}, instances={counts}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
