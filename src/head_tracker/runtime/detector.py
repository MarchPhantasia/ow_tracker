from __future__ import annotations

from pathlib import Path
from typing import Any

from src.head_tracker.runtime.config import RuntimeInferenceConfig
from src.head_tracker.runtime.types import Detection


class YOLODetector:
    def __init__(self, config: RuntimeInferenceConfig):
        from ultralytics import YOLO

        if not Path(config.weights).exists():
            raise FileNotFoundError(config.weights)
        self._cfg = config
        self._model = YOLO(config.weights)
        self._names = {
            int(class_id): str(name).strip().lower()
            for class_id, name in self._model.names.items()
        }

    def detect(self, image: Any) -> list[Detection]:
        results = self._model.predict(
            source=image,
            imgsz=self._cfg.imgsz,
            conf=self._cfg.conf,
            iou=self._cfg.iou,
            device=self._cfg.device,
            half=self._cfg.half,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None:
            return []

        detections: list[Detection] = []
        for box in boxes:
            class_id = int(box.cls.item())
            xyxy = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self._names.get(class_id, str(class_id)),
                    confidence=float(box.conf.item()),
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                )
            )
        return detections
