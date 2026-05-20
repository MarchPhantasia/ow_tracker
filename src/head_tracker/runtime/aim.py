from __future__ import annotations

from src.head_tracker.runtime.types import Detection, Point


def aim_point(detection: Detection, y_ratio: float = 0.30) -> Point:
    ratio = min(1.0, max(0.0, float(y_ratio)))
    return (
        (detection.x1 + detection.x2) * 0.5,
        detection.y1 + detection.height * ratio,
    )
