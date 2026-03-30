<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&ColorList=7B2FFF,0D0D2B,4A00E0,1A0533,00E5FF&height=280&section=header&text=PhantomEye&fontSize=80&fontColor=00E5FF&animation=fadeIn&fontAlignY=35&desc=AI-Powered%20Surveillance%20Intelligence%20System&descAlignY=60&descAlign=50" width="100%"/>
</div>

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=700&size=24&pause=900&color=B44FFF&center=true&vCenter=true&width=900&lines=Person+Re-ID+Across+Multiple+Cameras;Real-Time+Behavioral+Heatmap+Analytics;OSINT+Privacy+Exposure+Audit+Engine;Production+FastAPI+%7C+Streamlit+%7C+Docker" />
</p>

<p align="center">
  <a href="https://abu-sameer-66-phantomeye.hf.space">
    <img src="https://img.shields.io/badge/LIVE%20DEMO-HuggingFace-00E5FF?style=for-the-badge&logo=huggingface&logoColor=0D0D2B"/>
  </a>
  <a href="https://phantomeye-production.up.railway.app/docs">
    <img src="https://img.shields.io/badge/API%20DOCS-Railway-7B2FFF?style=for-the-badge&logo=railway&logoColor=white"/>
  </a>
  <a href="https://github.com/Abu-Sameer-66/PhantomEye">
    <img src="https://img.shields.io/badge/GitHub-PhantomEye-1A0533?style=for-the-badge&logo=github&logoColor=00E5FF"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Detection-7B2FFF?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/ByteTrack-Tracking-4A00E0?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OpenCV-LBPH%20Re--ID-1A0533?style=for-the-badge&logo=opencv&logoColor=00E5FF"/>
  <img src="https://img.shields.io/badge/FastAPI-8%20Endpoints-7B2FFF?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Deployed-4A00E0?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-1A0533?style=for-the-badge"/>
</p>

---

## 👁 What is PhantomEye?

> **PhantomEye** is an **end-to-end AI surveillance intelligence platform** that transforms raw camera feeds into actionable intelligence — in real time.

Unlike basic CCTV systems that only record, PhantomEye **reasons** over video:

- Detects every person with **YOLOv8-nano** at high confidence
- Assigns **persistent unique IDs** across frames with ByteTrack
- Generates **behavioral heatmaps** — showing where people dwell, linger, and move
- Fires **automated loitering alerts** when anomalies are detected
- Runs **OSINT privacy audits** — computing exposure scores from face embeddings
- Exposes everything through a **production-grade REST API** with interactive docs

Built for **law enforcement, retail intelligence, campus security, healthcare monitoring, and privacy research**.

---

## ⚡ Why PhantomEye is Different

Most CV projects stop at detection. PhantomEye goes further:

| Capability | Typical CV Project | PhantomEye |
|---|---|---|
| Person detection | ✅ | ✅ YOLOv8-nano |
| Multi-object tracking | ❌ | ✅ ByteTrack persistent IDs |
| Behavioral analytics | ❌ | ✅ Heatmap + dwell time |
| Loitering alerts | ❌ | ✅ Automated triggers |
| OSINT privacy audit | ❌ | ✅ Exposure score 0–100 |
| Production API | ❌ | ✅ FastAPI 8 endpoints |
| Live deployment | ❌ | ✅ HuggingFace + Railway |
| User data privacy | ❌ | ✅ Zero pre-loaded data |

---

## 🏗️ System Architecture
```text
INPUT LAYER
   │
   ├── Live RTSP Feed / Video File / Image Upload
   │
   ▼
VISION PIPELINE
   │
   ├── YOLOv8-nano ──── Person Detection (confidence + bbox)
   │
   ├── ByteTrack ─────── Persistent ID tracking across frames
   │                     Color-coded bounding boxes + trajectory trails
   │
   └── OSNet / LBPH ──── Re-ID Engine (cross-camera matching)
                         Cosine similarity gallery search
   │
   ▼
INTELLIGENCE LAYER
   │
   ├── Behavioral Analytics
   │     ├── NumPy heatmap accumulation
   │     ├── Dwell time per person (seconds)
   │     └── Loitering alert (threshold-based)
   │
   ├── OSINT Audit Engine
   │     ├── Face embedding extraction
   │     ├── Gallery cosine similarity search
   │     └── Privacy exposure score (0–100) + risk level
   │
   └── Event Logger ──── JSON report generation
   │
   ▼
OUTPUT LAYER
   │
   ├── FastAPI REST API ── 8 production endpoints (OAS 3.1)
   ├── Streamlit Dashboard ── Elite cyberpunk UI
   ├── WebSocket ── Real-time frame streaming
   └── PDF / JSON Reports ── Exportable audit logs
```

---

## 🚀 Live Deployment

| Service | Platform | URL |
|---|---|---|
| Interactive Dashboard | HuggingFace Spaces | [phantomeye.hf.space](https://abu-sameer-66-phantomeye.hf.space) |
| REST API | Railway | [phantomeye-production.up.railway.app](https://phantomeye-production.up.railway.app) |
| API Documentation | Railway | [/docs](https://phantomeye-production.up.railway.app/docs) |

---

## 🔌 API Reference

**Base URL:** `https://phantomeye-production.up.railway.app`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | System info + module status |
| GET | `/health` | Live health check + gallery size |
| POST | `/detect` | Person detection on image upload |
| POST | `/track/video` | Multi-object tracking on video |
| POST | `/osint/audit` | Privacy exposure audit |
| POST | `/osint/add-to-gallery` | Add person to reference gallery |
| GET | `/osint/gallery` | List registered persons |
| GET | `/outputs` | List processed output files |

### Quick test:
```bash
curl -X POST "https://phantomeye-production.up.railway.app/detect" \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "status": "success",
  "total_persons": 4,
  "detections": [
    { "id": 1, "bbox": [120, 80, 310, 420], "confidence": 0.87 },
    { "id": 2, "bbox": [450, 95, 620, 430], "confidence": 0.74 }
  ]
}
```

---

## 🧠 Core Modules

### Module 1 — Person Detection (`core/detection.py`)
YOLOv8-nano configured for class-0 (person) only detection. Returns bounding boxes, confidence scores, and annotated frames. CPU-optimized for deployment on standard hardware.

### Module 2 — Multi-Object Tracking (`core/tracker.py`)
Custom ByteTrack implementation with IOU-based matching. Each person receives a persistent color-coded ID with trajectory visualization. Handles occlusion and re-entry.

### Module 3 — Behavioral Analytics (`core/analytics.py`)
Accumulates person positions into a NumPy heatmap over time. Computes per-person dwell time in seconds. Triggers loitering alerts above configurable thresholds. Exports heatmap overlays and JSON event logs.

### Module 4 — OSINT Privacy Audit (`core/osint.py`)
Extracts face embeddings using OpenCV LBPH. Searches against a user-provided reference gallery via histogram correlation. Returns exposure score (0–100), risk level (LOW / MEDIUM / HIGH), and matched identities.

### Module 5 — REST API (`api/main.py`)
FastAPI backend with 8 production endpoints. Full OAS 3.1 documentation auto-generated. CORS enabled for frontend integration. Processes user-uploaded files in memory — nothing stored server-side.

---

## 🔒 Privacy & Security Design

PhantomEye is built with **privacy-first architecture**:

- **Zero pre-loaded data** — no faces, videos, or images in the repository
- **In-session processing** — uploaded files processed in RAM, never written to disk permanently
- **User-controlled gallery** — only data you explicitly upload is used
- **Ethical OSINT framing** — audit module designed for privacy defense, not surveillance offense
- **Open source** — full transparency into every processing step

---

## 📦 Local Setup
```bash
git clone https://github.com/Abu-Sameer-66/PhantomEye.git
cd PhantomEye

conda create -n phantomeye python=3.10 -y
conda activate phantomeye

pip install -r requirements.txt
```

**Run Streamlit dashboard:**
```bash
streamlit run app.py
```

**Run FastAPI backend:**
```bash
python api/main.py
```

**Run detection on a video:**
```bash
python core/detection.py data/videos/your_video.mp4
```

**Run OSINT audit:**
```bash
python core/osint.py data/gallery/query.jpg
```

---

## 🗂️ Repository Structure
```
PhantomEye/
├── core/
│   ├── detection.py       # YOLOv8 person detector
│   ├── tracker.py         # ByteTrack multi-object tracker
│   ├── analytics.py       # Behavioral heatmap + alerts
│   └── osint.py           # OSINT privacy audit engine
├── api/
│   ├── main.py            # FastAPI backend (8 endpoints)
│   └── routes/            # Modular route handlers
├── app.py                 # Streamlit dashboard
├── config.py              # Global configuration
├── Dockerfile             # Docker deployment
├── requirements.txt       # Dependencies
└── README.md
```

---

## 🌍 Real-World Applications

| Domain | Use Case |
|---|---|
| Law Enforcement | Cross-camera suspect tracking, evidence extraction |
| Retail Analytics | Customer heatmaps, queue monitoring, loss prevention |
| Campus Security | Attendance automation, unauthorized access detection |
| Healthcare | Patient fall detection, ICU wandering alerts |
| Privacy Research | Digital footprint auditing, OSINT defense |

---

## 👤 Author

**Abu Sameer** — AI/ML Engineer · Computer Vision Researcher · Open Source Contributor

[![Portfolio](https://img.shields.io/badge/Portfolio-sameer--nadeem--portfolio.vercel.app-00E5FF?style=for-the-badge)](https://sameer-nadeem-portfolio.vercel.app)
[![GitHub](https://img.shields.io/badge/GitHub-Abu--Sameer--66-1A0533?style=for-the-badge&logo=github&logoColor=00E5FF)](https://github.com/Abu-Sameer-66)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sameer%20Nadeem-7B2FFF?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/sameer-nadeem-66339a357/)
[![Kaggle](https://img.shields.io/badge/Kaggle-sameernadeem66-4A00E0?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/sameernadeem66)

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&ColorList=7B2FFF,1A0533,0D0D2B,4A00E0&height=120&section=footer" width="100%"/>
</div>
