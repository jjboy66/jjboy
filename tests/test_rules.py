import unittest
from datetime import datetime, timedelta, timezone

from src.models.detector import Detection
from src.postprocess.rules import RuleEngine


class RuleEngineTest(unittest.TestCase):
    def test_intrusion_alert(self):
        engine = RuleEngine(intrusion_zone=(0, 0, 100, 100))
        dets = [Detection("person", 0.9, (10, 10, 20, 30), track_id="a")]
        alerts = engine.evaluate(dets, source_id="cam1", now=datetime.now(timezone.utc))
        self.assertTrue(any(a.alert_type == "zone_intrusion" for a in alerts))

    def test_dwell_alert(self):
        engine = RuleEngine(intrusion_zone=(0, 0, 100, 100), dwell_threshold_seconds=1)
        now = datetime.now(timezone.utc)
        det = Detection("person", 0.9, (10, 10, 20, 30), track_id="a")
        engine.evaluate([det], source_id="cam1", now=now)
        alerts = engine.evaluate([det], source_id="cam1", now=now + timedelta(seconds=2))
        self.assertTrue(any(a.alert_type == "dwell_timeout" for a in alerts))


if __name__ == "__main__":
    unittest.main()
