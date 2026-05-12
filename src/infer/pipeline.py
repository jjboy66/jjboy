from __future__ import annotations

import base64
from dataclasses import asdict
from io import BytesIO
from typing import Any, Dict, List

from PIL import Image

from src.models.detector import Detection, MockVisionGuardDetector, filter_by_score
from src.postprocess.geometry import nms, soft_nms, weighted_boxes_fusion
from src.postprocess.rules import Alert, RuleEngine
from src.postprocess.temporal import TemporalSmoother


class VisionGuardPipeline:
    def __init__(
        self,
        confidence_threshold: float = 0.35,
        suppression_strategy: str = "wbf",
    ) -> None:
        self.detector = MockVisionGuardDetector(confidence_floor=confidence_threshold)
        self.rule_engine = RuleEngine()
        self.smoother = TemporalSmoother()
        self.confidence_threshold = confidence_threshold
        self.suppression_strategy = suppression_strategy

    def decode_image_size(self, image_b64: str) -> tuple[int, int]:
        content = base64.b64decode(image_b64)
        with Image.open(BytesIO(content)) as img:
            return img.size

    def process_frame(
        self,
        image_b64: str,
        source_id: str,
        frame_index: int,
    ) -> Dict[str, Any]:
        width, height = self.decode_image_size(image_b64)
        detections = self.detector.detect(width, height, source_id=source_id, frame_index=frame_index)
        detections = filter_by_score(detections, threshold=self.confidence_threshold)

        if self.suppression_strategy == "wbf":
            detections = weighted_boxes_fusion(detections)
        elif self.suppression_strategy == "soft_nms":
            detections = soft_nms(detections, iou_threshold=0.5, sigma=0.5)
        else:
            detections = nms(detections, iou_threshold=0.45)

        detections = self.smoother.smooth(detections)

        alerts = self.rule_engine.evaluate(detections, source_id=source_id)
        return {
            "source_id": source_id,
            "frame_index": frame_index,
            "detections": [self._detection_to_dict(d) for d in detections],
            "alerts": [self._alert_to_dict(a) for a in alerts],
        }

    @staticmethod
    def _detection_to_dict(det: Detection) -> Dict[str, Any]:
        payload = asdict(det)
        payload["bbox"] = [round(v, 2) for v in det.bbox]
        payload["score"] = round(det.score, 4)
        return payload

    @staticmethod
    def _alert_to_dict(alert: Alert) -> Dict[str, Any]:
        payload = asdict(alert)
        payload["timestamp"] = alert.timestamp.isoformat()
        return payload
