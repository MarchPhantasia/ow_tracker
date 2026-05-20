from __future__ import annotations

from dataclasses import dataclass
from time import sleep

from src.head_tracker.runtime.config import CaptureConfig


@dataclass(frozen=True)
class CapturedFrame:
    image: object
    region: tuple[int, int, int, int]


class DxcamCapture:
    def __init__(self, config: CaptureConfig):
        self._cfg = config
        self._camera = None

    @property
    def region(self) -> tuple[int, int, int, int]:
        return self._cfg.region

    def start(self) -> None:
        try:
            import dxcam
        except ImportError as exc:
            raise RuntimeError("dxcam is required for realtime capture. Install requirements-runtime-windows.txt") from exc
        self._camera = dxcam.create(output_color="RGB")
        self._camera.start(region=self.region, target_fps=self._cfg.target_fps, video_mode=True)

    def read(self) -> CapturedFrame | None:
        if self._camera is None:
            raise RuntimeError("capture has not been started")
        frame = self._camera.get_latest_frame()
        if frame is None:
            sleep(0.001)
            return None
        return CapturedFrame(image=frame, region=self.region)

    def stop(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera = None

    def __enter__(self) -> DxcamCapture:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()
