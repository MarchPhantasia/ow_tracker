from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from src.head_tracker.runtime.types import Point


@dataclass(frozen=True)
class FilterConfig:
    enabled: bool = True
    process_noise: float = 1.0
    measurement_noise: float = 1.0
    max_velocity_px_per_s: float = 3000.0
    lead_time_ms: float = 20.0


class PointKalmanFilter:
    """Small constant-velocity 2D filter for aim points.

    It is intentionally lightweight: the runtime needs stable velocity and lead
    behavior more than a fully configurable Kalman matrix implementation.
    """

    def __init__(
        self,
        *,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
        max_velocity_px_per_s: float = 3000.0,
    ):
        self._process_noise = max(0.0, float(process_noise))
        self._measurement_noise = max(1e-6, float(measurement_noise))
        self._max_velocity = max(1.0, float(max_velocity_px_per_s))
        self._pos: Point | None = None
        self._vel: Point = (0.0, 0.0)

    @property
    def initialized(self) -> bool:
        return self._pos is not None

    @property
    def velocity(self) -> Point:
        return self._vel

    def reset(self) -> None:
        self._pos = None
        self._vel = (0.0, 0.0)

    def update(self, measurement: Point, dt: float, confidence: float = 1.0) -> Point:
        dt = max(1e-4, float(dt))
        measurement = (float(measurement[0]), float(measurement[1]))
        if self._pos is None:
            self._pos = measurement
            self._vel = (0.0, 0.0)
            return measurement

        predicted = (
            self._pos[0] + self._vel[0] * dt,
            self._pos[1] + self._vel[1] * dt,
        )
        confidence = min(1.0, max(0.05, float(confidence)))
        noise_scale = self._measurement_noise / (self._measurement_noise + self._process_noise + confidence)
        alpha = min(1.0, max(0.10, 1.0 - noise_scale))

        new_pos = (
            predicted[0] + (measurement[0] - predicted[0]) * alpha,
            predicted[1] + (measurement[1] - predicted[1]) * alpha,
        )
        measured_vel = (
            (new_pos[0] - self._pos[0]) / dt,
            (new_pos[1] - self._pos[1]) / dt,
        )
        self._vel = _clamp_velocity(
            (
                self._vel[0] * (1.0 - alpha) + measured_vel[0] * alpha,
                self._vel[1] * (1.0 - alpha) + measured_vel[1] * alpha,
            ),
            self._max_velocity,
        )
        self._pos = new_pos
        return new_pos

    def predict(self, lead_time_s: float) -> Point:
        if self._pos is None:
            return (0.0, 0.0)
        lead = max(0.0, float(lead_time_s))
        return (
            self._pos[0] + self._vel[0] * lead,
            self._pos[1] + self._vel[1] * lead,
        )


def _clamp_velocity(velocity: Point, max_velocity: float) -> Point:
    speed = hypot(velocity[0], velocity[1])
    if speed <= max_velocity:
        return velocity
    scale = max_velocity / max(speed, 1e-6)
    return (velocity[0] * scale, velocity[1] * scale)
