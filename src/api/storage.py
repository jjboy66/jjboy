from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class JsonLineStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, payload: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        data = [json.loads(line) for line in lines if line.strip()]
        return data[-limit:]
