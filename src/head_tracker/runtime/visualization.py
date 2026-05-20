from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.head_tracker.runtime.types import Detection, SelectedTarget


def draw_overlay(
    image: object,
    detections: Iterable[Detection],
    selected: SelectedTarget | None,
    *,
    crosshair: tuple[float, float] | None = None,
) -> object:
    import cv2

    frame = image.copy()
    for detection in detections:
        color = (0, 0, 255) if detection.class_name == "enemy" else (255, 180, 0)
        p1 = (int(detection.x1), int(detection.y1))
        p2 = (int(detection.x2), int(detection.y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(frame, label, (p1[0], max(0, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if selected is not None:
        point = (int(selected.aim_point[0]), int(selected.aim_point[1]))
        cv2.drawMarker(frame, point, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    if crosshair is not None:
        cv2.drawMarker(frame, (int(crosshair[0]), int(crosshair[1])), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
    return frame


def write_image(path: str | Path, image: object) -> None:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
