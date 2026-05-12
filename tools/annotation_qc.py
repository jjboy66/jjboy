from __future__ import annotations

import json
import sys
from pathlib import Path


def run_qc(manifest_path: str) -> int:
    path = Path(manifest_path)
    if not path.exists():
        print("manifest not found")
        return 1

    issues = 0
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        x1, y1, x2, y2 = item["bbox"]
        if x2 <= x1 or y2 <= y1:
            print(f"line {idx}: invalid bbox {item['bbox']}")
            issues += 1
        if x1 < 0 or y1 < 0:
            print(f"line {idx}: negative bbox coordinates")
            issues += 1

    print(f"qc finished, issues={issues}")
    return 0 if issues == 0 else 2


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/annotation_qc.py <manifest.jsonl>")
    raise SystemExit(run_qc(sys.argv[1]))
