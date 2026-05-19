from __future__ import annotations

import argparse
from pathlib import Path

from src.head_tracker.dataset import prepare_yolo_data
from src.head_tracker.inference import selected_class_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference. Defaults to enemy only.")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", default="datasets/data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="enemy")
    parser.add_argument("--include-ally", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from ultralytics import YOLO

    prepared = prepare_yolo_data(args.data, Path("runs") / "datasets" / "predict")
    classes = selected_class_ids(prepared.names, include_ally=args.include_ally)
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        classes=classes,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        show=args.show,
        exist_ok=args.exist_ok,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
