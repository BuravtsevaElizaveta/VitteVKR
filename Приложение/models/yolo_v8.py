# models/yolo_v8.py
"""
Мини-обёртка над Ultralytics YOLOv8.

Изменения:
• device строка приводится к lower-case;
• .to(device) вызывается только если device in {'cuda', 'mps'}.
"""

import logging
from ultralytics import YOLO  # type: ignore

logger = logging.getLogger(__name__)


class YOLOv8:
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = (device or "cpu").strip().lower()
        if self.device not in ("cpu", "cuda", "mps"):
            logger.warning("Неизвестное устройство «%s», используем CPU.", device)
            self.device = "cpu"
        self.model = self._load_model()

    # ---------------------------------------------------------------- private --
    def _load_model(self):
        try:
            logger.info("Загрузка YOLOv8 (%s) → %s", self.model_path, self.device)
            model = YOLO(self.model_path)
            if self.device in ("cuda", "mps"):
                model.to(self.device)
            logger.info("YOLOv8 готов.")
            return model
        except Exception as e:
            logger.exception("YOLOv8 load failed: %s", e)
            raise

    # ---------------------------------------------------------------- public --
    def predict(self, frame, conf=0.25):
        """
        Возвращает result.boxes.data (numpy) для первого кадра.
        """
        results = self.model(frame, conf=conf)
        result = results[0]
        if result.boxes is not None:
            return result.boxes.data.cpu().numpy()
        return []
