from __future__ import annotations

import argparse
from time import perf_counter, sleep

import cv2

from src.head_tracker.runtime.capture import DxcamCapture
from src.head_tracker.runtime.config import load_runtime_config
from src.head_tracker.runtime.detector import YOLODetector
from src.head_tracker.runtime.hotkey import X2Hotkey
from src.head_tracker.runtime.latency import LatencyLogger
from src.head_tracker.runtime.mouse_mover import RelativeMouseMover, SendInputMouse
from src.head_tracker.runtime.selection import TargetSelector
from src.head_tracker.runtime.smoothing import PointKalmanFilter
from src.head_tracker.runtime.visualization import draw_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime ally/enemy runtime.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-mouse", action="store_true", help="Run detection/debug without moving the mouse.")
    parser.add_argument("--no-window", action="store_true", help="Disable OpenCV debug window.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_runtime_config(args.config)
    detector = YOLODetector(cfg.inference)
    selector = TargetSelector(cfg.selection)
    smoother = PointKalmanFilter(
        process_noise=cfg.filter.process_noise,
        measurement_noise=cfg.filter.measurement_noise,
        max_velocity_px_per_s=cfg.filter.max_velocity_px_per_s,
    )
    mover = RelativeMouseMover(cfg.mouse)
    mouse = None if args.no_mouse else SendInputMouse()
    hotkey = X2Hotkey(cfg.hotkey.button)
    crosshair = (cfg.capture.fov_width * 0.5, cfg.capture.fov_height * 0.5)
    frame_idx = 0
    last = perf_counter()

    print("Runtime started. Hold X2 to move. Press q in debug window to quit.")
    with DxcamCapture(cfg.capture) as capture, LatencyLogger(cfg.debug.latency_csv) as latency:
        while True:
            loop_start = perf_counter()
            captured = capture.read()
            if captured is None:
                continue

            after_capture = perf_counter()
            detections = detector.detect(captured.image)
            after_inference = perf_counter()
            selected = selector.update(detections, crosshair)
            after_select = perf_counter()

            now = perf_counter()
            dt = max(1e-4, now - last)
            last = now

            predicted = None
            if selected is not None:
                filtered = smoother.update(selected.aim_point, dt, selected.detection.confidence)
                predicted = smoother.predict(cfg.filter.lead_time_ms / 1000.0) if cfg.filter.enabled else filtered
            else:
                smoother.reset()
                mover.reset()

            pressed = hotkey.is_pressed()
            dxdy = (0, 0)
            if selected is not None and predicted is not None and pressed:
                dxdy = mover.compute(predicted, crosshair, dt)
                if mouse is not None:
                    mouse.move(*dxdy)
            elif not pressed:
                mover.reset()

            after_mouse = perf_counter()
            frame_idx += 1
            if frame_idx % max(1, cfg.debug.log_every_n_frames) == 0:
                total_ms = (after_mouse - loop_start) * 1000.0
                fps = 1000.0 / max(total_ms, 1e-6)
                print(
                    f"frame={frame_idx} fps={fps:.1f} "
                    f"inference_ms={(after_inference - after_capture) * 1000.0:.1f} "
                    f"selected={int(selected is not None)} pressed={int(pressed)} dxdy={dxdy}"
                )
                latency.write(
                    {
                        "frame": frame_idx,
                        "capture_ms": round((after_capture - loop_start) * 1000.0, 3),
                        "inference_ms": round((after_inference - after_capture) * 1000.0, 3),
                        "select_ms": round((after_select - after_inference) * 1000.0, 3),
                        "mouse_ms": round((after_mouse - after_select) * 1000.0, 3),
                        "total_ms": round(total_ms, 3),
                        "selected": int(selected is not None),
                        "pressed": int(pressed),
                        "dx": dxdy[0],
                        "dy": dxdy[1],
                    }
                )

            if cfg.debug.enable_visualizer and not args.no_window:
                overlay = draw_overlay(captured.image, detections, selected, crosshair=crosshair)
                cv2.imshow("ow_tracker runtime", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                sleep(0.001)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
