from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.head_tracker.runtime.config import CaptureConfig, load_runtime_config


def test_runtime_config_defaults_to_2560_1440_center_capture():
    root = Path("tmp") / "test-artifacts" / uuid4().hex
    root.mkdir(parents=True)
    path = root / "config.yaml"
    path.write_text("runtime:\n  conf: 0.65\n", encoding="utf-8")

    cfg = load_runtime_config(path)

    assert cfg.capture.screen_width == 2560
    assert cfg.capture.screen_height == 1440
    assert cfg.capture.fov_width == 1920
    assert cfg.capture.fov_height == 1080
    assert cfg.inference.conf == 0.65
    assert cfg.selection.enemy_class == "enemy"
    assert cfg.selection.confirm_frames == 2


def test_capture_region_clamps_when_fov_exceeds_screen():
    capture = CaptureConfig(
        screen_width=1920,
        screen_height=1080,
        fov_width=2560,
        fov_height=1440,
    )

    assert capture.region == (0, 0, 1920, 1080)


def test_capture_region_size_matches_clamped_region():
    capture = CaptureConfig(
        screen_width=1920,
        screen_height=1080,
        fov_width=1280,
        fov_height=720,
    )

    assert capture.region == (320, 180, 1600, 900)
    assert capture.region_size == (1280, 720)
