from __future__ import annotations

from src.head_tracker.runtime.selection import SelectionConfig, TargetSelector
from src.head_tracker.runtime.types import Detection


def det(
    class_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    confidence: float = 0.8,
) -> Detection:
    class_id = 1 if class_name == "enemy" else 0
    return Detection(class_id, class_name, confidence, (x1, y1, x2, y2))


def test_selector_ignores_ally_and_confirms_enemy_for_two_frames():
    selector = TargetSelector(SelectionConfig(confirm_frames=2, min_confidence=0.55))

    first = selector.update([det("ally", 90, 90, 130, 170), det("enemy", 100, 100, 140, 180)], (120, 140))
    second = selector.update([det("enemy", 101, 100, 141, 180)], (120, 140))

    assert first is None
    assert second is not None
    assert second.detection.class_name == "enemy"


def test_selector_rejects_far_fresh_target():
    selector = TargetSelector(
        SelectionConfig(confirm_frames=1, max_acquisition_distance_px=120),
    )

    selected = selector.update([det("enemy", 500, 500, 560, 620)], (100, 100))

    assert selected is None


def test_selector_keeps_locked_target_through_short_miss_then_drops():
    selector = TargetSelector(
        SelectionConfig(confirm_frames=1, max_lost_frames=1),
    )
    locked = selector.update([det("enemy", 100, 100, 140, 180)], (120, 140))

    one_miss = selector.update([], (120, 140))
    second_miss = selector.update([], (120, 140))

    assert locked is not None
    assert one_miss is not None
    assert second_miss is None


def test_selector_stays_sticky_unless_new_target_is_clearly_better():
    selector = TargetSelector(
        SelectionConfig(confirm_frames=1, switch_margin_px=40, association_radius_px=120),
    )
    locked = selector.update([det("enemy", 300, 100, 360, 220)], (330, 160))
    still_locked = selector.update(
        [
            det("enemy", 304, 100, 364, 220),
            det("enemy", 270, 100, 330, 220),
        ],
        (300, 160),
    )
    switched = selector.update(
        [
            det("enemy", 308, 100, 368, 220),
            det("enemy", 120, 100, 180, 220),
        ],
        (150, 160),
    )

    assert locked is not None
    assert still_locked is not None
    assert still_locked.detection.center == (334.0, 160.0)
    assert switched is not None
    assert switched.detection.center == (150.0, 160.0)


def test_selector_acquires_by_aim_point_not_box_center():
    selector = TargetSelector(
        SelectionConfig(confirm_frames=1, aim_y_ratio=0.0, max_acquisition_distance_px=40),
    )
    crosshair = (120.0, 120.0)
    # center is far from crosshair, but aim point (top-center) is close enough
    target = det("enemy", 100, 100, 140, 300)

    selected = selector.update([target], crosshair)

    assert selected is not None
    assert selected.aim_point == (120.0, 100.0)
