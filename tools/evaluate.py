from __future__ import annotations

import json

from src.train.trainer import VisionGuardTrainer


if __name__ == "__main__":
    trainer = VisionGuardTrainer()
    metrics = trainer.evaluate()
    print(json.dumps({"evaluation": metrics}, ensure_ascii=False, indent=2))
