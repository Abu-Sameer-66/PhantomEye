import os
import time
import shutil
import uvicorn
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.detection import PersonDetector
from core.tracker import ByteTracker
from core.osint import OSINTAudit
from config import OUTPUTS_DIR, GALLERY_DIR, API_HOST, API_PORT

app = FastAPI(
    title="PhantomEye API",
    description="AI-powered surveillance intelligence — Person Re-ID, Behavioral Analytics, OSINT Defense",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = PersonDetector()
osint    = OSINTAudit()

import cv2
import numpy as np


@app.get("/")
def root():
    return {
        "system"  : "PhantomEye",
        "version" : "1.0.0",
        "status"  : "online",
        "modules" : ["detection", "tracking", "analytics", "osint"],
        "author"  : "Abu-Sameer-66",
    }


@app.get("/health")
def health():
    return {
        "status"        : "healthy",
        "gallery_size"  : len(osint.gallery),
        "timestamp"     : time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.post("/detect")
async def detect_persons(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "Only JPG/PNG images supported.")

    data  = await file.read()
    arr   = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Cannot decode image.")

    detections = detector.detect(image)

    return JSONResponse({
        "status"          : "success",
        "filename"        : file.filename,
        "total_persons"   : len(detections),
        "detections"      : [
            {
                "id"        : i + 1,
                "bbox"      : list(d["bbox"]),
                "confidence": d["confidence"],
            }
            for i, d in enumerate(detections)
        ],
    })


@app.post("/osint/audit")
async def osint_audit(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "Only JPG/PNG images supported.")

    data  = await file.read()
    arr   = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Cannot decode image.")

    query_id = Path(file.filename).stem
    result   = osint.audit(image, query_id=query_id)
    osint.save_report(result)

    return JSONResponse({
        "status": "success",
        "audit" : result,
    })


@app.post("/osint/add-to-gallery")
async def add_to_gallery(
    file     : UploadFile = File(...),
    person_id: str        = "unknown",
):
    data  = await file.read()
    arr   = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Cannot decode image.")

    success = osint.add_to_gallery(image, person_id)

    if not success:
        raise HTTPException(400, "No face detected in uploaded image.")

    return JSONResponse({
        "status"      : "success",
        "person_id"   : person_id,
        "gallery_size": len(osint.gallery),
        "message"     : f"Person '{person_id}' added to gallery.",
    })


@app.get("/osint/gallery")
def get_gallery():
    return JSONResponse({
        "status"      : "success",
        "gallery_size": len(osint.gallery),
        "persons"     : list(osint.gallery.keys()),
    })


@app.post("/track/video")
async def track_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(400, "Only MP4/AVI/MOV videos supported.")

    tmp_path = OUTPUTS_DIR / f"tmp_{int(time.time())}_{file.filename}"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap   = cv2.VideoCapture(str(tmp_path))
    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tracker    = ByteTracker()
    cap        = cv2.VideoCapture(str(tmp_path))
    frame_data = []
    frame_idx  = 0

    while frame_idx < min(total, fps * 10):
        ret, frame = cap.read()
        if not ret:
            break
        dets   = detector.detect(frame)
        active = tracker.update(dets)
        frame_data.append({
            "frame"          : frame_idx,
            "active_persons" : len(active),
            "track_ids"      : [t.track_id for t in active],
        })
        frame_idx += 1

    cap.release()
    tmp_path.unlink(missing_ok=True)

    return JSONResponse({
        "status"          : "success",
        "filename"        : file.filename,
        "resolution"      : f"{w}x{h}",
        "fps"             : fps,
        "frames_analyzed" : frame_idx,
        "unique_persons"  : tracker.next_id - 1,
        "frame_summary"   : frame_data[:10],
    })


@app.get("/outputs")
def list_outputs():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        {
            "name": f.name,
            "size": f"{f.stat().st_size // 1024} KB",
        }
        for f in OUTPUTS_DIR.iterdir()
        if f.is_file() and not f.name.startswith("tmp_")
    ]
    return JSONResponse({
        "status": "success",
        "total" : len(files),
        "files" : files,
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=port,
        reload=False,
    )