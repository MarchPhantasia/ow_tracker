from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.head_tracker.runtime.aim import aim_point
from src.head_tracker.runtime.latency import LatencyLogger
from src.head_tracker.runtime.mouse_mover import MouseMoveConfig, RelativeMouseMover
from src.head_tracker.runtime.smoothing import PointKalmanFilter
from src.head_tracker.runtime.types import Detection


def test_aim_point_uses_upper_body_ratio():
    detection = Detection(1, "enemy", 0.9, (100, 50, 180, 250))

    assert aim_point(detection, 0.30) == (140.0, 110.0)


def test_kalman_predicts_forward_with_lead_time():
    filt = PointKalmanFilter(process_noise=1.0, measurement_noise=1.0, max_velocity_px_per_s=5000)
    filt.update((100.0, 100.0), dt=1 / 60, confidence=0.9)
    filt.update((120.0, 100.0), dt=1 / 60, confidence=0.9)

    predicted = filt.predict(lead_time_s=0.05)

    assert predicted[0] > 120.0
    assert abs(predicted[1] - 100.0) < 1.0


def test_mouse_mover_brakes_when_direction_reverses():
    mover = RelativeMouseMover(
        MouseMoveConfig(
            smoothing=1.0,
            sensitivity_scale=1.0,
            max_step_px=200,
            max_accel_input_px_per_sec=60,
            deadzone_px=0.0,
        )
    )

    for _ in range(10):
        mover.compute((130.0, 100.0), (100.0, 100.0), dt=1 / 60)

    dx, dy = mover.compute((95.0, 100.0), (100.0, 100.0), dt=1 / 60)

    assert dx <= 0
    assert dy == 0


def test_mouse_mover_deadzone_suppresses_micro_corrections():
    mover = RelativeMouseMover(
        MouseMoveConfig(
            smoothing=1.0,
            sensitivity_scale=1.0,
            max_step_px=200,
            max_accel_input_px_per_sec=10000,
            deadzone_px=3.0,
        )
    )

    assert mover.compute((101.0, 100.0), (100.0, 100.0), dt=1 / 60) == (0, 0)


def test_latency_logger_writes_csv():
    root = Path("tmp") / "test-artifacts" / uuid4().hex
    root.mkdir(parents=True)
    path = root / "latency.csv"
    logger = LatencyLogger(path)

    logger.write({"frame": 1, "capture_ms": 2.0, "inference_ms": 8.0})
    logger.close()

    text = Path(path).read_text(encoding="utf-8")
    assert "frame,capture_ms,inference_ms" in text
    assert "1,2.0,8.0" in text
