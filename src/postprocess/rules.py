from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Sequence

from src.models.detector import Detection
from src.postprocess.geometry import center, inside_rect


@dataclass(slots=True)
class Alert:
    alert_type: str
    severity: str
    source_id: str
    message: str
    timestamp: datetime
    track_id: str | None = None


@dataclass
class RuleEngine:
    intrusion_zone: tuple[float, float, float, float] = (50, 50, 590, 430)
    monitored_labels: tuple[str, ...] = ("person", "forklift")
    dwell_threshold_seconds: int = 20
    risky_labels: tuple[str, ...] = ("forklift",)
    _first_seen: Dict[str, datetime] = field(default_factory=dict)

    def evaluate(
        self,
        detections: Sequence[Detection],
        source_id: str,
        now: datetime | None = None,
    ) -> List[Alert]:
        now = now or datetime.now(timezone.utc)
        alerts: List[Alert] = []

        for det in detections:
            point = center(det.bbox)
            if det.label in self.monitored_labels and inside_rect(point, self.intrusion_zone):
                alerts.append(
                    Alert(
                        alert_type="zone_intrusion",
                        severity="medium",
                        source_id=source_id,
                        message=f"{det.label} entered monitored zone",
                        timestamp=now,
                        track_id=det.track_id,
                    )
                )

            if det.track_id:
                first = self._first_seen.setdefault(det.track_id, now)
                if now - first >= timedelta(seconds=self.dwell_threshold_seconds):
                    alerts.append(
                        Alert(
                            alert_type="dwell_timeout",
                            severity="high",
                            source_id=source_id,
                            message=f"{det.label} stayed over {self.dwell_threshold_seconds}s",
                            timestamp=now,
                            track_id=det.track_id,
                        )
                    )

            if det.label in self.risky_labels:
                alerts.append(
                    Alert(
                        alert_type="dangerous_behavior",
                        severity="high",
                        source_id=source_id,
                        message=f"Risk object detected: {det.label}",
                        timestamp=now,
                        track_id=det.track_id,
                    )
                )

        return alerts
