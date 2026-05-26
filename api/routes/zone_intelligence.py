"""
PhantomEye — Zone Intelligence API Route
api/routes/zone_intelligence.py
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import traceback

from core.zone_intelligence import (
    get_engine, reset_engine, ZoneType
)

router = APIRouter(prefix="/api/zone-intelligence", tags=["Zone Intelligence"])


# ── Schemas ───────────────────────────────────────────── #

class ZoneDefInput(BaseModel):
    name:         str
    zone_type:    str   # RESTRICTED | MONITORED | CAPACITY_LIMITED | SAFE
    x1:           int
    y1:           int
    x2:           int
    y2:           int
    max_capacity: int = 5


class BBoxInput(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionInput(BaseModel):
    person_id: int
    bbox:      BBoxInput


class FrameInput(BaseModel):
    frame_id:   Optional[int] = None
    detections: List[DetectionInput]


# ── Endpoints ─────────────────────────────────────────── #

@router.post("/zones/add")
async def add_zone(payload: ZoneDefInput):
    try:
        zone_type = ZoneType(payload.zone_type.upper())
        engine    = get_engine()
        zone      = engine.add_zone(
            name=payload.name, zone_type=zone_type,
            x1=payload.x1, y1=payload.y1,
            x2=payload.x2, y2=payload.y2,
            max_capacity=payload.max_capacity,
        )
        return {"status": "ok", "zone": zone.to_dict(), "total_zones": len(engine.zones)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/zones")
async def list_zones():
    try:
        engine = get_engine()
        return {
            "status":      "ok",
            "total_zones": len(engine.zones),
            "zones":       [z.to_dict() for z in engine.zones.values()],
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.delete("/zones/{zone_id}")
async def delete_zone(zone_id: int):
    try:
        engine = get_engine()
        engine.remove_zone(zone_id)
        return {"status": "ok", "message": f"Zone {zone_id} removed", "total_zones": len(engine.zones)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.post("/update")
async def update_frame(payload: FrameInput):
    try:
        engine     = get_engine()
        detections = [
            {"person_id": d.person_id, "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]}
            for d in payload.detections
        ]
        result = engine.update(detections, frame_id=payload.frame_id)
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "detail": str(e), "trace": traceback.format_exc()}


@router.get("/events")
async def get_events(n: int = 20):
    try:
        engine = get_engine()
        return {"status": "ok", "events": engine.get_recent_events(n)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/summary")
async def session_summary():
    try:
        engine = get_engine()
        return {"status": "ok", "summary": engine.session_summary()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.post("/reset")
async def reset():
    try:
        reset_engine()
        return {"status": "ok", "message": "ZIE engine reset. All zones and tracks cleared."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}