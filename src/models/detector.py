from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


BBox = Tuple[float, float, float, float]


@dataclass(slots=True)
class Detection:
    label: str
    score: float
    bbox: BBox
    track_id: str | None = None
    attributes: Dict[str, str] = field(default_factory=dict)


class BaseDetector:
    def detect(self, width: int, height: int, source_id: str, frame_index: int) -> List[Detection]:
        raise NotImplementedError


class MockVisionGuardDetector(BaseDetector):
    """Deterministic fallback detector to keep end-to-end pipeline runnable."""

    def __init__(self, confidence_floor: float = 0.4) -> None:
        self.confidence_floor = confidence_floor

    def detect(self, width: int, height: int, source_id: str, frame_index: int) -> List[Detection]:
        box_w = max(width * 0.22, 30)
        box_h = max(height * 0.35, 40)
        shift = (frame_index % 9) * 2
        x1 = max(width * 0.45 - box_w / 2 + shift, 0)
        y1 = max(height * 0.55 - box_h / 2, 0)
        x2 = min(x1 + box_w, width - 1)
        y2 = min(y1 + box_h, height - 1)
        person_score = min(0.94, self.confidence_floor + 0.45)

        detections: List[Detection] = [
            Detection(
                label="person",
                score=person_score,
                bbox=(x1, y1, x2, y2),
                track_id=f"{source_id}:person:1",
            )
        ]

        if frame_index % 3 == 0:
            detections.append(
                Detection(
                    label="helmet",
                    score=0.71,
                    bbox=(x1 + 8, y1, min(x1 + 30, x2), min(y1 + 26, y2)),
                    track_id=f"{source_id}:helmet:1",
                )
            )
        return detections


def filter_by_score(detections: Sequence[Detection], threshold: float) -> List[Detection]:
    return [det for det in detections if det.score >= threshold]
