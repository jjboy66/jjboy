import unittest

from src.models.detector import Detection
from src.postprocess.geometry import iou, nms, weighted_boxes_fusion


class PostprocessTest(unittest.TestCase):
    def test_iou_basic(self):
        value = iou((0, 0, 10, 10), (5, 5, 15, 15))
        self.assertGreater(value, 0.1)
        self.assertLess(value, 0.2)

    def test_nms_filters_overlap(self):
        dets = [
            Detection("person", 0.9, (0, 0, 10, 10)),
            Detection("person", 0.8, (1, 1, 11, 11)),
        ]
        kept = nms(dets, iou_threshold=0.3)
        self.assertEqual(len(kept), 1)

    def test_wbf_merges(self):
        dets = [
            Detection("helmet", 0.7, (10, 10, 20, 20)),
            Detection("helmet", 0.6, (11, 11, 21, 21)),
        ]
        fused = weighted_boxes_fusion(dets, iou_threshold=0.3)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].label, "helmet")


if __name__ == "__main__":
    unittest.main()
