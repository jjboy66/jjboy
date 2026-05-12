from __future__ import annotations

from pathlib import Path


if __name__ == "__main__":
    out_dir = Path("data/processed/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx = out_dir / "visionguard_lite.onnx"
    trt = out_dir / "visionguard_lite.engine"
    onnx.write_text("placeholder onnx artifact", encoding="utf-8")
    trt.write_text("placeholder tensorrt artifact", encoding="utf-8")
    print(f"Exported artifacts: {onnx} {trt}")
