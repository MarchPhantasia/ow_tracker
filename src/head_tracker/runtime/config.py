from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src.head_tracker.runtime.mouse_mover import MouseMoveConfig
from src.head_tracker.runtime.selection import SelectionConfig
from src.head_tracker.runtime.smoothing import FilterConfig


@dataclass(frozen=True)
class CaptureConfig:
    backend: str = "dxcam"
    screen_width: int = 2560
    screen_height: int = 1440
    fov_width: int = 1920
    fov_height: int = 1080
    target_fps: int = 120

    @property
    def region(self) -> tuple[int, int, int, int]:
        left = max(0, (self.screen_width - self.fov_width) // 2)
        top = max(0, (self.screen_height - self.fov_height) // 2)
        return (left, top, left + self.fov_width, top + self.fov_height)


@dataclass(frozen=True)
class RuntimeInferenceConfig:
    weights: str = "models/best.pt"
    imgsz: int = 960
    conf: float = 0.55
    iou: float = 0.50
    device: str = "0"
    half: bool = True


@dataclass(frozen=True)
class HotkeyConfig:
    button: str = "x2"


@dataclass(frozen=True)
class DebugConfig:
    enable_visualizer: bool = True
    latency_csv: str = "runs/runtime/latency.csv"
    log_every_n_frames: int = 30


@dataclass(frozen=True)
class RuntimeConfig:
    capture: CaptureConfig = CaptureConfig()
    inference: RuntimeInferenceConfig = RuntimeInferenceConfig()
    selection: SelectionConfig = SelectionConfig()
    filter: FilterConfig = FilterConfig()
    mouse: MouseMoveConfig = MouseMoveConfig()
    hotkey: HotkeyConfig = HotkeyConfig()
    debug: DebugConfig = DebugConfig()


def load_runtime_config(path: str | Path = "config.yaml") -> RuntimeConfig:
    path = Path(path)
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    runtime = raw.get("runtime", {}) if isinstance(raw, dict) else {}
    if not isinstance(runtime, dict):
        raise ValueError("config.yaml runtime section must be a mapping")

    inference_values = dict(runtime.get("inference") or {})
    for key in ("weights", "imgsz", "conf", "iou", "device", "half"):
        if key in runtime and key not in inference_values:
            inference_values[key] = runtime[key]

    return RuntimeConfig(
        capture=_build(CaptureConfig(), runtime.get("capture")),
        inference=_build(RuntimeInferenceConfig(), inference_values),
        selection=_build(SelectionConfig(), runtime.get("selection")),
        filter=_build(FilterConfig(), runtime.get("filter")),
        mouse=_build(MouseMoveConfig(), runtime.get("mouse")),
        hotkey=_build(HotkeyConfig(), runtime.get("hotkey")),
        debug=_build(DebugConfig(), runtime.get("debug")),
    )


def _build(defaults: Any, values: Any) -> Any:
    if values is None:
        return defaults
    if not isinstance(values, dict):
        raise ValueError(f"expected mapping for {type(defaults).__name__}")
    allowed = defaults.__dataclass_fields__.keys()
    filtered = {key: value for key, value in values.items() if key in allowed}
    return replace(defaults, **filtered)
