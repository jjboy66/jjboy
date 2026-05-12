from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(slots=True)
class Annotation:
    image_id: str
    width: int
    height: int
    label: str
    bbox: list[float]
    scene_tags: list[str]
    hard_example: bool = False


def load_manifest(path: str) -> List[Annotation]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    annotations: List[Annotation] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        annotations.append(Annotation(**payload))
    return annotations


def save_manifest(path: str, items: Iterable[Annotation]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
