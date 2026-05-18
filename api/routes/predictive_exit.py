"""
PhantomEye — Predictive Exit Vector API Route
api/routes/predictive_exit.py

Endpoints:
  POST /api/predictive-exit/update   — feed frame detections, get predictions
  GET  /api/predictive-exit/status   — engine status + active track count
  POST /api/predictive-exit/reset    — clear all tracks
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import traceback

from core.predictive_exit import get_engine, reset_engine

router = APIRouter(prefix="/api/predictive-exit", tags=["Predictive Exit Vector"])


# --- Request/Response schemas --------------------------------------------- #

class BBoxModel(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionInput(BaseModel):
    person_id: int
    bbox: BBoxModel


class FrameInput(BaseModel):
    frame_id: Optional[int] = None
    frame_width: int = 640
    frame_height: int = 480
    fps: float = 25.0
    detections: List[DetectionInput]


# --- Endpoints ------------------------------------------------------------ #

@router.post("/update")
async def update_predictions(payload: FrameInput):
    try:
        engine = get_engine(
            frame_width=payload.frame_width,
            frame_height=payload.frame_height,
            fps=payload.fps,
        )

        detections = [
            {
                "person_id": d.person_id,
                "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
            }
            for d in payload.detections
        ]

        predictions = engine.update(detections, frame_id=payload.frame_id)

        alerts = [p for p in predictions if p.alert]

        return {
            "status": "ok",
            "frame_id": payload.frame_id,
            "active_tracks": len(engine.tracks),
            "predictions": [p.to_dict() for p in predictions],
            "alert_count": len(alerts),
            "alerts": [p.to_dict() for p in alerts],
        }

    except Exception as e:
        return {"status": "error", "detail": str(e), "trace": traceback.format_exc()}


@router.get("/status")
async def engine_status():
    try:
        engine = get_engine()
        track_summary = []
        for pid, track in engine.tracks.items():
            track_summary.append({
                "person_id": pid,
                "history_frames": len(track.positions),
            })

        return {
            "status": "ok",
            "engine": "PEV v1.0",
            "frame_size": f"{engine.W}x{engine.H}",
            "fps": engine.fps,
            "active_tracks": len(engine.tracks),
            "tracks": track_summary,
        }

    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.post("/reset")
async def reset():
    try:
        reset_engine()
        return {"status": "ok", "message": "PEV engine reset. All tracks cleared."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}