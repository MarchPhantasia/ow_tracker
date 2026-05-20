from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2

from src.head_tracker.dataset import IMAGE_EXTENSIONS
from src.head_tracker.runtime.config import load_runtime_config
from src.head_tracker.runtime.detector import YOLODetector
from src.head_tracker.runtime.selection import TargetSelector
from src.head_tracker.runtime.visualization import draw_overlay, write_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline runtime replay. Never moves the mouse.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default="datasets/test/images")
    parser.add_argument("--output", default="runs/replay")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_runtime_config(args.config)
    detector = YOLODetector(cfg.inference)
    selector = TargetSelector(cfg.selection)
    output = Path(args.output)
    start = perf_counter()
    source = Path(args.source)
    if source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
        processed = _process_video(source, output, detector, selector, args.limit, start)
    else:
        images = _list_images(source)
        if args.limit > 0:
            images = images[: args.limit]
        processed = _process_images(images, output, detector, selector, start)
    print(f"saved replay output to {output}")
    print(f"processed={processed}")
    return 0


def _process_images(
    images: list[Path],
    output: Path,
    detector: YOLODetector,
    selector: TargetSelector,
    start: float,
) -> int:
    for idx, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        overlay = _process_frame(image, detector, selector)
        write_image(output / image_path.name, overlay)
        if idx % 25 == 0:
            elapsed = max(1e-6, perf_counter() - start)
            print(f"processed={idx}/{len(images)} fps={idx / elapsed:.1f}")
    return len(images)


def _process_video(
    source: Path,
    output: Path,
    detector: YOLODetector,
    selector: TargetSelector,
    limit: int,
    start: float,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(source)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        overlay = _process_frame(frame, detector, selector)
        write_image(output / f"{source.stem}_{frame_idx:06d}.jpg", overlay)
        if frame_idx % 25 == 0:
            elapsed = max(1e-6, perf_counter() - start)
            print(f"processed={frame_idx} fps={frame_idx / elapsed:.1f}")
        if limit > 0 and frame_idx >= limit:
            break
    cap.release()
    return frame_idx


def _process_frame(image: object, detector: YOLODetector, selector: TargetSelector) -> object:
    detections = detector.detect(image)
    crosshair = (image.shape[1] * 0.5, image.shape[0] * 0.5)
    selected = selector.update(detections, crosshair)
    return draw_overlay(image, detections, selected, crosshair=crosshair)


def _list_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(source)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


if __name__ == "__main__":
    raise SystemExit(main())
