<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0a,50:1a0040,100:0d1f0d&height=300&section=header&text=👁%20PhantomEye&fontSize=75&fontColor=00ff88&animation=fadeIn&fontAlignY=38&desc=AI-Powered%20Surveillance%20Intelligence%20System&descSize=18&descAlignY=58&descAlign=50&descColor=00aa55" width="100%"/>

<br/>

<img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=00FF88&center=true&vCenter=true&width=800&lines=Person+Re-ID+Across+Camera+Networks;Real-Time+Behavioral+Heatmap+Engine;OSINT+Privacy+Exposure+Score+0%E2%80%93100;Production+FastAPI+%7C+Docker+%7C+HuggingFace" />

<br/><br/>

<a href="https://abu-sameer-66-phantomeye.hf.space">
<img src="https://img.shields.io/badge/%F0%9F%9F%A2%20LIVE%20DEMO-HuggingFace%20Spaces-00ff88?style=for-the-badge&labelColor=0d1f0d"/>
</a>
&nbsp;
<a href="https://phantomeye-production.up.railway.app/docs">
<img src="https://img.shields.io/badge/%F0%9F%93%A1%20API%20DOCS-Railway-00aa55?style=for-the-badge&labelColor=003322"/>
</a>
&nbsp;
<a href="https://medium.com/@sameerdataanalyst66/i-built-an-ai-that-watches-tracks-and-audits-phantomeye-is-live-afe2f62bcb7b">
<img src="https://img.shields.io/badge/%F0%9F%93%96%20MEDIUM-Full%20Article-00ff88?style=for-the-badge&labelColor=0d1f0d"/>
</a>
&nbsp;
<a href="https://github.com/Abu-Sameer-66/PhantomEye/blob/main/LICENSE">
<img src="https://img.shields.io/badge/License-MIT-003322?style=for-the-badge&labelColor=0d1f0d"/>
</a>

<br/><br/>

<img src="https://img.shields.io/badge/YOLOv8-nano-00ff88?style=flat-square&logo=pytorch&logoColor=black&labelColor=0d1f0d"/>
<img src="https://img.shields.io/badge/ByteTrack-IOU%20Matching-00aa55?style=flat-square&labelColor=003322"/>
<img src="https://img.shields.io/badge/OpenCV-LBPH%20Face-00ff88?style=flat-square&logo=opencv&logoColor=black&labelColor=0d1f0d"/>
<img src="https://img.shields.io/badge/FastAPI-8%20Endpoints-00aa55?style=flat-square&logo=fastapi&logoColor=white&labelColor=003322"/>
<img src="https://img.shields.io/badge/Streamlit-Dashboard-00ff88?style=flat-square&logo=streamlit&logoColor=black&labelColor=0d1f0d"/>
<img src="https://img.shields.io/badge/Docker-Containerized-00aa55?style=flat-square&logo=docker&logoColor=white&labelColor=003322"/>
<img src="https://img.shields.io/badge/Python-3.10-00ff88?style=flat-square&logo=python&logoColor=black&labelColor=0d1f0d"/>

</div>

---

## What is PhantomEye?

Most computer vision projects stop at detection. They draw a box around a person and call it done. The box appears. The box disappears. Nothing is remembered. Nothing is understood.

**PhantomEye goes further.**

It is an end-to-end AI surveillance intelligence platform that transforms passive camera feeds into a live reasoning engine — detecting, tracking, analyzing behavior, and auditing privacy, all in one unified system.

Built from scratch. Deployed live. Zero pre-loaded data.

---

## Capabilities

<table>
<tr>
<td width="50%">

### 👁 Person Detection
YOLOv8-nano configured for class-0 only. Returns bounding boxes and confidence scores on standard CPU in milliseconds. No GPU required.

### 🎯 Multi-Object Tracking
Custom ByteTrack with IOU matching. Each person receives a persistent color-coded ID with trajectory trail — across frames, through occlusion, across re-entries.

### 🔥 Behavioral Heatmap
NumPy position accumulation builds a live heatmap of human movement. High-activity zones appear red. Dwell time tracked per person in seconds.

</td>
<td width="50%">

### ⚠️ Loitering Alerts
Automated alerts fire when a person exceeds the dwell threshold. No human monitoring required.

### 🔍 OSINT Privacy Audit
Upload a face — get a Privacy Exposure Score from 0 to 100. LBPH embedding search against a reference gallery. Risk classification: LOW / MEDIUM / HIGH.

### ⚡ Production REST API
FastAPI backend with 8 endpoints. OAS 3.1 interactive docs. CORS enabled. In-memory file processing — nothing stored server-side.

</td>
</tr>
</table>

---

## Why PhantomEye is Different

| Capability | Typical CV Project | PhantomEye |
|:---|:---:|:---:|
| Person detection | ✅ | ✅ YOLOv8-nano |
| Persistent ID tracking | ❌ | ✅ ByteTrack |
| Behavioral heatmap | ❌ | ✅ NumPy accumulation |
| Dwell time analytics | ❌ | ✅ Per-person seconds |
| Automated loitering alert | ❌ | ✅ Threshold-based |
| OSINT privacy audit | ❌ | ✅ Score 0–100 |
| Production REST API | ❌ | ✅ 8 endpoints, OAS 3.1 |
| Live 24/7 deployment | ❌ | ✅ HuggingFace + Railway |
| Zero pre-loaded data | ❌ | ✅ Privacy-first |

---

## System Architecture
```
┌─────────────────────────────────────────────────┐
│                   INPUT LAYER                   │
│   Image Upload  /  Video File  /  RTSP Feed     │
└───────────────────────┬─────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────┐
│                 VISION PIPELINE                 │
│                                                 │
│  YOLOv8-nano ──── Person Detection              │
│      │            bbox + confidence             │
│      │                                          │
│  ByteTrack ─────── Persistent ID Assignment     │
│      │             Color trails + occlusion     │
│      │                                          │
│  LBPH Engine ───── Re-ID Gallery Search         │
│                    Cosine similarity match      │
└───────────────────────┬─────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────┐
│               INTELLIGENCE LAYER                │
│                                                 │
│  Behavioral Analytics                           │
│  ├── NumPy heatmap accumulation                 │
│  ├── Per-person dwell time (seconds)            │
│  └── Automated loitering alerts                 │
│                                                 │
│  OSINT Audit Engine                             │
│  ├── Face embedding extraction                  │
│  ├── Gallery similarity search                  │
│  └── Exposure score (0–100) + risk level        │
└───────────────────────┬─────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────┐
│                  OUTPUT LAYER                   │
│                                                 │
│  FastAPI REST API   ── 8 endpoints, OAS 3.1     │
│  Streamlit Dashboard── Cyberpunk UI             │
│  JSON Reports       ── Exportable audit logs    │
└─────────────────────────────────────────────────┘
```

---

## Live Deployment

| Service | Platform | Status |
|:---|:---|:---:|
| [Interactive Dashboard](https://abu-sameer-66-phantomeye.hf.space) | HuggingFace Spaces | 🟢 Live |
| [REST API](https://phantomeye-production.up.railway.app) | Railway | 🟢 Live |
| [API Documentation](https://phantomeye-production.up.railway.app/docs) | Railway | 🟢 Live |

---

## API Reference

**Base URL:** `https://phantomeye-production.up.railway.app`

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | System info + version |
| `GET` | `/health` | Live health check |
| `POST` | `/detect` | Person detection on image |
| `POST` | `/track/video` | Multi-object tracking on video |
| `POST` | `/osint/audit` | Privacy exposure audit |
| `POST` | `/osint/add-to-gallery` | Register person to gallery |
| `GET` | `/osint/gallery` | List gallery persons |
| `GET` | `/outputs` | List output files |

**Quick test:**
```bash
curl -X POST "https://phantomeye-production.up.railway.app/detect" \
  -F "file=@crowd.jpg"
```
```json
{
  "status": "success",
  "total_persons": 8,
  "detections": [
    { "id": 1, "bbox": [120, 80, 310, 420], "confidence": 0.87 },
    { "id": 2, "bbox": [450, 95, 620, 430], "confidence": 0.74 }
  ]
}
```

---

## Local Setup
```bash
git clone https://github.com/Abu-Sameer-66/PhantomEye.git
cd PhantomEye

conda create -n phantomeye python=3.10 -y
conda activate phantomeye
pip install -r requirements.txt
```
```bash
# Streamlit dashboard
streamlit run app.py

# FastAPI backend
python api/main.py

# Detection on video
python core/detection.py data/videos/your_video.mp4

# OSINT audit
python core/osint.py data/gallery/query.jpg
```

---

## Repository Structure
```
PhantomEye/
├── core/
│   ├── detection.py        YOLOv8 person detector
│   ├── tracker.py          ByteTrack multi-object tracker
│   ├── analytics.py        Heatmap + dwell time + alerts
│   └── osint.py            OSINT privacy audit engine
├── api/
│   ├── main.py             FastAPI backend — 8 endpoints
│   └── routes/             Modular route handlers
├── app.py                  Streamlit dashboard
├── config.py               Global configuration
├── Dockerfile              Container deployment
└── requirements.txt        Dependencies
```

---

## Real-World Applications

| Domain | Use Case |
|:---|:---|
| Law Enforcement | Cross-camera suspect tracking, automated evidence extraction |
| Retail Intelligence | Customer heatmaps, queue monitoring, loss prevention |
| Campus Security | Contactless attendance, unauthorized access detection |
| Healthcare | Patient fall detection, ICU wandering alerts |
| Privacy Research | Digital footprint auditing, OSINT defense tools |

---

## Privacy-First Design

- **Zero pre-loaded data** — no faces, videos, or images in the repository
- **In-session processing** — uploaded files processed in RAM only, never stored
- **User-controlled gallery** — only data you explicitly upload is referenced
- **Ethical OSINT framing** — audit module built for privacy defense, not offense
- **Fully open source** — every processing step is transparent and auditable

---

## Roadmap

- [ ] Deep metric learning Re-ID — OSNet + triplet loss on Market-1501
- [ ] Cross-camera person matching across multiple feeds
- [ ] Natural language queries — plain English → structured CV filters via LLM
- [ ] Anonymization mode — behavioral analytics with automatic face blurring
- [ ] Edge deployment — Raspberry Pi + Jetson Nano optimization

---

## Author

<div align="center">

**Abu Sameer** — AI/ML Engineer · Computer Vision Researcher · GSoC 2026 Contributor Aspirant

<br/>

<a href="https://sameer-nadeem-portfolio.vercel.app"><img src="https://img.shields.io/badge/Portfolio-sameer--nadeem--portfolio-00ff88?style=for-the-badge&labelColor=0d1f0d"/></a>
<a href="https://github.com/Abu-Sameer-66"><img src="https://img.shields.io/badge/GitHub-Abu--Sameer--66-00aa55?style=for-the-badge&logo=github&labelColor=003322"/></a>
<a href="https://www.linkedin.com/in/sameer-nadeem-66339a357/"><img src="https://img.shields.io/badge/LinkedIn-Sameer%20Nadeem-00ff88?style=for-the-badge&logo=linkedin&labelColor=0d1f0d"/></a>
<a href="https://www.kaggle.com/sameernadeem66"><img src="https://img.shields.io/badge/Kaggle-sameernadeem66-00aa55?style=for-the-badge&logo=kaggle&labelColor=003322"/></a>
<a href="https://medium.com/@sameerdataanalyst66/i-built-an-ai-that-watches-tracks-and-audits-phantomeye-is-live-afe2f62bcb7b"><img src="https://img.shields.io/badge/Medium-Full%20Article-00ff88?style=for-the-badge&logo=medium&labelColor=0d1f0d"/></a>

</div>

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1f0d,50:003322,100:0a0a0a&height=120&section=footer" width="100%"/>
</div>
