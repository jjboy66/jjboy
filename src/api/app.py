from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.api.storage import JsonLineStore
from src.infer.pipeline import VisionGuardPipeline


class DetectRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image content")
    source_id: str = Field(default="camera-1")
    frame_index: int = Field(default=0)


class FeedbackRequest(BaseModel):
    source_id: str
    frame_index: int
    note: str
    accepted: bool = True
    tags: List[str] = Field(default_factory=list)


class RetrainRequest(BaseModel):
    reason: str
    dataset_version: str


app = FastAPI(title="VisionGuard-Lite API", version="0.1.0")
pipeline = VisionGuardPipeline()
alerts_store = JsonLineStore("data/processed/alerts.jsonl")
feedback_store = JsonLineStore("data/annotations/feedback.jsonl")
retrain_store = JsonLineStore("data/processed/retrain_requests.jsonl")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "visionguard-lite"}


@app.post("/detect")
def detect(payload: DetectRequest) -> Dict[str, Any]:
    result = pipeline.process_frame(
        image_b64=payload.image_base64,
        source_id=payload.source_id,
        frame_index=payload.frame_index,
    )
    for alert in result["alerts"]:
        alerts_store.append(alert)
    return result


@app.get("/alerts")
def list_alerts(limit: int = 20) -> Dict[str, Any]:
    return {"items": alerts_store.recent(limit=limit)}


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest) -> Dict[str, Any]:
    record = payload.model_dump()
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    feedback_store.append(record)
    return {"status": "accepted", "record": record}


@app.post("/active-learning/retrain-request")
def request_retrain(payload: RetrainRequest) -> Dict[str, Any]:
    task = payload.model_dump()
    task["created_at"] = datetime.now(timezone.utc).isoformat()
    task["status"] = "queued"
    retrain_store.append(task)
    return {"status": "queued", "task": task}
