from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.models.detector import BBox, Detection


@dataclass
class TemporalSmoother:
    momentum: float = 0.65
    state: Dict[str, BBox] = field(default_factory=dict)

    def smooth(self, detections: List[Detection]) -> List[Detection]:
        smoothed: List[Detection] = []
        for det in detections:
            key = det.track_id or f"{det.label}:{round(det.bbox[0], 1)}:{round(det.bbox[1], 1)}"
            prev = self.state.get(key)
            if prev:
                blended = tuple(
                    self.momentum * prev[i] + (1 - self.momentum) * det.bbox[i] for i in range(4)
                )
                det = Detection(
                    label=det.label,
                    score=det.score,
                    bbox=blended,  # type: ignore[arg-type]
                    track_id=det.track_id,
                    attributes=det.attributes,
                )
            self.state[key] = det.bbox
            smoothed.append(det)
        return smoothed
