from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from src.models.detector import BBox, Detection


EPS = 1e-9


def area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(box_a: BBox, box_b: BBox) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = area((x1, y1, x2, y2))
    union = area(box_a) + area(box_b) - inter
    if union <= EPS:
        return 0.0
    return inter / union


def nms(detections: Sequence[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    ordered = sorted(detections, key=lambda d: d.score, reverse=True)
    keep: List[Detection] = []
    while ordered:
        candidate = ordered.pop(0)
        keep.append(candidate)
        ordered = [d for d in ordered if iou(d.bbox, candidate.bbox) < iou_threshold]
    return keep


def soft_nms(
    detections: Sequence[Detection],
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    min_score: float = 0.05,
) -> List[Detection]:
    mutable = [
        Detection(
            label=d.label,
            score=d.score,
            bbox=d.bbox,
            track_id=d.track_id,
            attributes=dict(d.attributes),
        )
        for d in detections
    ]
    keep: List[Detection] = []
    while mutable:
        mutable.sort(key=lambda d: d.score, reverse=True)
        best = mutable.pop(0)
        keep.append(best)
        next_round: List[Detection] = []
        for det in mutable:
            overlap = iou(best.bbox, det.bbox)
            if overlap > iou_threshold:
                decay = math.exp(-(overlap * overlap) / max(sigma, EPS))
                det.score *= decay
            if det.score >= min_score:
                next_round.append(det)
        mutable = next_round
    return keep


def weighted_boxes_fusion(
    detections: Sequence[Detection], iou_threshold: float = 0.55
) -> List[Detection]:
    clusters: List[List[Detection]] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        assigned = False
        for cluster in clusters:
            if iou(det.bbox, cluster[0].bbox) >= iou_threshold and det.label == cluster[0].label:
                cluster.append(det)
                assigned = True
                break
        if not assigned:
            clusters.append([det])

    fused: List[Detection] = []
    for cluster in clusters:
        weight_sum = sum(d.score for d in cluster) + EPS
        x1 = sum(d.bbox[0] * d.score for d in cluster) / weight_sum
        y1 = sum(d.bbox[1] * d.score for d in cluster) / weight_sum
        x2 = sum(d.bbox[2] * d.score for d in cluster) / weight_sum
        y2 = sum(d.bbox[3] * d.score for d in cluster) / weight_sum
        score = max(d.score for d in cluster)
        track_id = cluster[0].track_id
        fused.append(
            Detection(
                label=cluster[0].label,
                score=score,
                bbox=(x1, y1, x2, y2),
                track_id=track_id,
                attributes=dict(cluster[0].attributes),
            )
        )
    return fused


def center(box: BBox) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def inside_rect(point: Tuple[float, float], rect: BBox) -> bool:
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]
