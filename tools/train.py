from __future__ import annotations

import json

from src.train.trainer import VisionGuardTrainer


if __name__ == "__main__":
    trainer = VisionGuardTrainer()
    cfg = trainer.load_config()
    result = trainer.run_training()
    payload = {
        "config": cfg,
        "metrics": {
            "mAP50_95": result.mAP50_95,
            "recall": result.recall,
            "small_object_ap": result.small_object_ap,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
