from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def convert(csv_path: str, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
        reader = csv.DictReader(f)
        for row in reader:
            payload = {
                "image_id": row["image_id"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "label": row["label"],
                "bbox": [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                "scene_tags": row.get("scene_tags", "").split("|") if row.get("scene_tags") else [],
                "hard_example": row.get("hard_example", "false").lower() == "true",
            }
            w.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python tools/data_convert.py <input.csv> <output.jsonl>")
    convert(sys.argv[1], sys.argv[2])
