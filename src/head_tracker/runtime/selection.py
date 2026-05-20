from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from src.head_tracker.runtime.aim import aim_point
from src.head_tracker.runtime.types import Detection, Point, SelectedTarget


@dataclass(frozen=True)
class SelectionConfig:
    enemy_class: str = "enemy"
    aim_y_ratio: float = 0.30
    min_confidence: float = 0.55
    max_acquisition_distance_px: float = 320.0
    min_box_area_px: float = 24.0
    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 1.60
    confirm_frames: int = 2
    max_lost_frames: int = 2
    association_radius_px: float = 120.0
    switch_margin_px: float = 40.0


class TargetSelector:
    def __init__(self, config: SelectionConfig):
        self._cfg = config
        self._locked: Detection | None = None
        self._lost_frames = 0
        self._candidate: Detection | None = None
        self._candidate_frames = 0

    def reset(self) -> None:
        self._locked = None
        self._lost_frames = 0
        self._candidate = None
        self._candidate_frames = 0

    def update(self, detections: list[Detection], crosshair: Point) -> SelectedTarget | None:
        valid = [d for d in detections if self._is_valid_enemy(d)]

        if self._locked is not None:
            associated = self._find_associated(valid, self._locked)
            if associated is None:
                if self._lost_frames < self._cfg.max_lost_frames:
                    self._lost_frames += 1
                    return self._selected(self._locked)
                self.reset()
            else:
                self._lost_frames = 0
                self._candidate = None
                self._candidate_frames = 0
                closest = min(valid, key=lambda d: _distance(self._target_point(d), crosshair))
                if closest is not associated:
                    associated_dist = _distance(self._target_point(associated), crosshair)
                    closest_dist = _distance(self._target_point(closest), crosshair)
                    if closest_dist < associated_dist - self._cfg.switch_margin_px:
                        self._locked = closest
                        return self._selected(closest)
                self._locked = associated
                return self._selected(associated)

        if not valid:
            self._candidate = None
            self._candidate_frames = 0
            return None

        candidate = min(valid, key=lambda d: _distance(self._target_point(d), crosshair))
        if _distance(self._target_point(candidate), crosshair) > self._cfg.max_acquisition_distance_px:
            self._candidate = None
            self._candidate_frames = 0
            return None

        if self._candidate is None or _distance(candidate.center, self._candidate.center) > self._cfg.association_radius_px:
            self._candidate = candidate
            self._candidate_frames = 1
        else:
            self._candidate = candidate
            self._candidate_frames += 1

        if self._candidate_frames >= max(1, self._cfg.confirm_frames):
            self._locked = candidate
            self._lost_frames = 0
            return self._selected(candidate)
        return None

    def _is_valid_enemy(self, detection: Detection) -> bool:
        if detection.class_name.strip().lower() != self._cfg.enemy_class.strip().lower():
            return False
        if detection.confidence < self._cfg.min_confidence:
            return False
        if detection.area < self._cfg.min_box_area_px:
            return False
        if detection.height <= 0.0:
            return False
        aspect = detection.width / detection.height
        return self._cfg.min_aspect_ratio <= aspect <= self._cfg.max_aspect_ratio

    def _find_associated(self, detections: list[Detection], locked: Detection) -> Detection | None:
        if not detections:
            return None
        nearest = min(detections, key=lambda d: _distance(d.center, locked.center))
        if _distance(nearest.center, locked.center) <= self._cfg.association_radius_px:
            return nearest
        return None

    def _selected(self, detection: Detection) -> SelectedTarget:
        return SelectedTarget(
            detection=detection,
            aim_point=self._target_point(detection),
        )

    def _target_point(self, detection: Detection) -> Point:
        return aim_point(detection, self._cfg.aim_y_ratio)


def _distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])
