from __future__ import annotations

import ctypes
from dataclasses import dataclass

from src.head_tracker.runtime.types import Point


@dataclass(frozen=True)
class MouseMoveConfig:
    smoothing: float = 0.85
    sensitivity_scale: float = 2.5
    max_step_px: int = 180
    max_accel_input_px_per_sec: float = 4200.0
    deadzone_px: float = 3.0


class RelativeMouseMover:
    def __init__(self, config: MouseMoveConfig):
        self._cfg = config
        self._residual = [0.0, 0.0]
        self._prev_velocity = [0.0, 0.0]

    def reset(self) -> None:
        self._residual = [0.0, 0.0]
        self._prev_velocity = [0.0, 0.0]

    def compute(self, target: Point, crosshair: Point, dt: float) -> tuple[int, int]:
        dt = max(1e-4, float(dt))
        base_delta = [
            (float(target[0]) - float(crosshair[0])) * self._cfg.smoothing,
            (float(target[1]) - float(crosshair[1])) * self._cfg.smoothing,
        ]

        for axis in (0, 1):
            if base_delta[axis] * self._prev_velocity[axis] < 0.0:
                self._prev_velocity[axis] = 0.0
                self._residual[axis] = 0.0

        delta = [
            base_delta[0] + self._residual[0],
            base_delta[1] + self._residual[1],
        ]
        if (delta[0] ** 2 + delta[1] ** 2) ** 0.5 <= self._cfg.deadzone_px:
            self.reset()
            return (0, 0)

        max_step = max(1, int(self._cfg.max_step_px))
        delta = [
            max(-max_step, min(max_step, delta[0])),
            max(-max_step, min(max_step, delta[1])),
        ]

        scale = max(1e-6, float(self._cfg.sensitivity_scale))
        desired = [delta[0] / scale, delta[1] / scale]
        max_change = max(0.0, float(self._cfg.max_accel_input_px_per_sec)) * dt
        velocity = [0.0, 0.0]
        for axis in (0, 1):
            change = desired[axis] - self._prev_velocity[axis]
            change = max(-max_change, min(max_change, change))
            velocity[axis] = self._prev_velocity[axis] + change
        self._prev_velocity = velocity

        dx = int(round(velocity[0]))
        dy = int(round(velocity[1]))
        self._residual = [
            delta[0] - dx * scale,
            delta[1] - dy * scale,
        ]
        return (dx, dy)


class SendInputMouse:
    def __init__(self):
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("SendInputMouse is only available on Windows")
        self._user32 = ctypes.windll.user32

    def move(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        extra = ctypes.c_ulong(0)
        input_struct = _INPUT(
            type=0,
            mi=_MOUSEINPUT(
                dx=int(dx),
                dy=int(dy),
                mouseData=0,
                dwFlags=0x0001,
                time=0,
                dwExtraInfo=ctypes.pointer(extra),
            ),
        )
        self._user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(input_struct))


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", _MOUSEINPUT),
    ]
