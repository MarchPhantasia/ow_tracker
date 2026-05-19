from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.head_tracker.dataset import prepare_yolo_data
from src.head_tracker.training import copy_best_weight, recommend_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ally/enemy YOLO detector.")
    parser.add_argument("--data", default="datasets/data.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="ally_enemy")
    parser.add_argument("--cache", default="False")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--use-all-data", action="store_true")
    parser.add_argument("--copy-best", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--copy-best-to", default="models/best.pt")
    parser.add_argument("--recommend-epoch-from")
    parser.add_argument("--select-metric", default="metrics/mAP50-95(B)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.recommend_epoch_from:
        rec = recommend_epoch(args.recommend_epoch_from, args.select_metric)
        print(
            f"Recommended epoch: {rec.epoch} "
            f"({rec.metric}={rec.value:.6f}) from {rec.results_csv}"
        )
        return 0

    from ultralytics import YOLO

    prepared = prepare_yolo_data(
        args.data,
        Path("runs") / "datasets" / args.name,
        use_all_data=args.use_all_data,
    )
    print(f"Using generated data config: {prepared.data_yaml}")
    print(f"Classes: {prepared.names}")
    for split, summary in prepared.splits.items():
        counts = {
            prepared.names[class_id]: count
            for class_id, count in summary.instances_by_class.items()
        }
        print(
            f"{split}: images={summary.image_count}, labels={summary.label_count}, "
            f"empty={summary.empty_label_count}, missing={summary.missing_label_count}, "
            f"invalid={summary.invalid_label_count}, instances={counts}"
        )

    model = YOLO(args.model)
    results = model.train(
        data=str(prepared.data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        cache=args.cache,
        patience=0 if args.use_all_data else args.patience,
        exist_ok=args.exist_ok,
    )

    run_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    copy_policy = args.copy_best
    should_copy = copy_policy == "always" or (copy_policy == "auto" and args.use_all_data)
    if should_copy:
        copied = copy_best_weight(run_dir, args.copy_best_to)
        print(f"Copied best weight to {copied}")
    else:
        print(f"Skipped copying best weight to {args.copy_best_to} (--copy-best {copy_policy})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
