<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0a,50:1a0040,100:0d1f0d&height=300&section=header&text=📷%20PhantomEye&fontSize=75&fontColor=00ff88&animation=fadeIn&fontAlignY=38&desc=AI-Powered%20Surveillance%20Intelligence%20System&descSize=18&descAlignY=58&descAlign=50&descColor=00aa55" width="100%"/>

<br/>

<img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=00FF88&center=true&vCenter=true&width=900&lines=Person+Detection+%7C+ByteTrack+Multi-Object+Tracking;Deep+ReID+OSNet+—+Rank-1+81.7%25+mAP+58.5%25;Emotion+Intelligence+—+Age+%7C+Gender+%7C+Emotion;Weapon+Detection+—+9+Classes+mAP50+53.2%25;Natural+Language+Queries+—+English+%2B+Roman+Urdu;OSINT+Privacy+Audit+—+Exposure+Score+0%E2%80%93100" />

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

<img src="https://img.shields.io/badge/YOLOv8-Detection-00ff88?style=flat-square&logo=pytorch&logoColor=black&labelColor=0d1f0d"/>
<img src="https://img.shields.io/badge/ByteTrack-Multi--Object%20Tracking-00aa55?style=flat-square&labelColor=003322"/>
<img src="https://img.shields.io/badge/OSNet-Deep%20ReID%20Rank--1%2081.7%25-00ff88?style=flat-square&logo=pytorch&logoColor=black&labelColor=0d1f0d"/>
<img src="https://img.shields.io/badge/DeepFace-Emotion%20Intelligence-00aa55?style=flat-square&labelColor=003322"/>
<img src="https://img.shields.io/badge/Groq%20LLaMA3-NL%20Query%20Engine-00ff88?style=flat-square&labelColor=0d1f0d"/>
<img src="https://img.shields.io/badge/YOLOv8%20Custom-Weapon%20Detection-00aa55?style=flat-square&labelColor=003322"/>
<img src="https://img.shields.io/badge/FastAPI-8%20Endpoints-00ff88?style=flat-square&logo=fastapi&logoColor=white&labelColor=0d1f0d"/>
<img src="https://img.shields.io/badge/Docker-Containerized-00aa55?style=flat-square&logo=docker&logoColor=white&labelColor=003322"/>
<img src="https://img.shields.io/badge/Python-3.10-00ff88?style=flat-square&logo=python&logoColor=black&labelColor=0d1f0d"/>

</div>

---

## What is PhantomEye?

Most computer vision projects stop at detection. They draw a box around a person and call it done. The box appears. The box disappears. Nothing is remembered. Nothing is understood.

**PhantomEye goes further.**

It is a full-stack AI surveillance intelligence platform that transforms passive camera feeds into a live reasoning engine — detecting, tracking, analyzing behavior, auditing identity, recognizing emotion, querying in natural language, and detecting weapons, all in one unified system.

Built entirely from scratch. Trained on real datasets. Deployed live. Zero pre-loaded data.

---

## Intelligence Modules

<table>
<tr>
<td width="50%">

### 📷 Person Detection
YOLOv8-nano configured for class-0 only. Returns bounding boxes and confidence scores on standard CPU in milliseconds. No GPU required at inference time.

### 🎯 Multi-Object Tracking
Custom ByteTrack with IOU matching. Each person receives a persistent color-coded ID with trajectory trail — across frames, through occlusion, across re-entries.

### 🔥 Behavioral Heatmap
NumPy position accumulation builds a live heatmap of human movement. High-activity zones appear red. Dwell time tracked per person in seconds. Loitering alerts fire automatically.

### 🧠 Deep Person Re-ID
OSNet x0.25 trained from scratch on Market-1501 (12,936 images, 751 identities). **Rank-1: 81.7% — mAP: 58.5%.** Identifies the same person across camera networks using body appearance alone — no face required.

</td>
<td width="50%">

### 😶 Emotion Intelligence
DeepFace pipeline — detects age, gender, and dominant emotion per face. Powered by TensorFlow with OpenCV face detector backend. Optimized for CPU deployment.

### 💬 Natural Language Query Engine
Groq LLaMA 3 powered query parser. Ask questions in plain English or Roman Urdu — the system extracts structured filters automatically. First open-source surveillance system with multilingual NL query support.

### 🔫 Weapon Detection
YOLOv8 custom trained on 9 weapon classes — Handgun, Knife, Shotgun, Sniper, Automatic Rifle, SMG, Sword, Bazooka, Grenade Launcher. **mAP50: 53.2% — Handgun: 89.5% — Shotgun: 96.3% — SMG: 98.6%.** Real-time threat alert on detection.

### 🔍 OSINT Privacy Audit
Upload a face — get a Privacy Exposure Score from 0 to 100. LBPH embedding search against a reference gallery. Risk classification: LOW / MEDIUM / HIGH.

</td>
</tr>
</table>

---

## Benchmark Results

| Module | Model | Metric | Score |
|:---|:---|:---|:---:|
| Person Detection | YOLOv8-nano | Confidence | >85% avg |
| Multi-Object Tracking | ByteTrack | ID Persistence | Across occlusion |
| Deep Re-ID | OSNet x0.25 | **Rank-1** | **81.7%** |
| Deep Re-ID | OSNet x0.25 | **mAP** | **58.5%** |
| Emotion Recognition | DeepFace | Face Detection | OpenCV backend |
| Weapon Detection | YOLOv8n custom | **mAP50** | **53.2%** |
| Weapon Detection | YOLOv8n custom | Handgun AP | 89.5% |
| Weapon Detection | YOLOv8n custom | Shotgun AP | 96.3% |
| Weapon Detection | YOLOv8n custom | SMG AP | 98.6% |
| NL Query Engine | Groq LLaMA 3 | Languages | English + Roman Urdu |

---

## Why PhantomEye is Different

| Capability | Typical CV Project | PhantomEye |
|:---|:---:|:---:|
| Person detection | ✅ | ✅ YOLOv8-nano |
| Persistent ID tracking | ❌ | ✅ ByteTrack |
| Behavioral heatmap | ❌ | ✅ NumPy accumulation |
| Dwell time analytics | ❌ | ✅ Per-person seconds |
| Loitering alert | ❌ | ✅ Threshold-based |
| Deep Re-ID (no face) | ❌ | ✅ OSNet Rank-1 81.7% |
| Emotion recognition | ❌ | ✅ DeepFace — Age + Gender + Emotion |
| Weapon detection | ❌ | ✅ 9-class YOLOv8 custom |
| NL query interface | ❌ | ✅ English + Roman Urdu |
| OSINT privacy audit | ❌ | ✅ Score 0–100 |
| Production REST API | ❌ | ✅ 8 endpoints, OAS 3.1 |
| Live 24/7 deployment | ❌ | ✅ HuggingFace + Railway |
| Zero pre-loaded data | ❌ | ✅ Privacy-first |

---

## System Architecture
┌──────────────────────────────────────────────────────┐
│                     INPUT LAYER                      │
│    Image Upload  /  Video File  /  RTSP Feed         │
└────────────────────────┬─────────────────────────────┘
│
┌────────────────────────▼─────────────────────────────┐
│                  VISION PIPELINE                     │
│                                                      │
│  YOLOv8-nano ─────── Person Detection                │
│       │               bbox + confidence              │
│       │                                              │
│  ByteTrack ──────── Persistent ID Assignment         │
│       │              Color trails + occlusion        │
│       │                                              │
│  OSNet x0.25 ────── Deep Person Re-ID                │
│                      Rank-1 81.7% on Market-1501     │
└────────────────────────┬─────────────────────────────┘
│
┌────────────────────────▼─────────────────────────────┐
│               INTELLIGENCE LAYER                     │
│                                                      │
│  Behavioral Analytics                                │
│  ├── NumPy heatmap accumulation                      │
│  ├── Per-person dwell time (seconds)                 │
│  └── Automated loitering alerts                      │
│                                                      │
│  Emotion Intelligence                                │
│  ├── DeepFace — Age + Gender + Emotion               │
│  └── OpenCV detector backend (CPU optimized)         │
│                                                      │
│  Weapon Detection                                    │
│  ├── YOLOv8 custom — 9 weapon classes                │
│  └── Real-time threat alert on detection             │
│                                                      │
│  NL Query Engine                                     │
│  ├── Groq LLaMA 3 — query parser                     │
│  └── English + Roman Urdu → structured filters       │
│                                                      │
│  OSINT Audit Engine                                  │
│  ├── LBPH face embedding extraction                  │
│  ├── Gallery similarity search                       │
│  └── Exposure score (0–100) + risk level             │
└────────────────────────┬─────────────────────────────┘
│
┌────────────────────────▼─────────────────────────────┐
│                   OUTPUT LAYER                       │
│                                                      │
│  FastAPI REST API    ── 8 endpoints, OAS 3.1         │
│  Streamlit Dashboard ── Cyberpunk UI                 │
│  JSON Reports        ── Exportable audit logs        │
└──────────────────────────────────────────────────────┘

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

# Detection on image or video
python core/detection.py

# OSINT audit
python core/osint.py

# Weapon detection
python core/weapon.py

# Emotion analysis
python core/emotion.py
```

---

## Repository Structure
PhantomEye/
├── core/
│   ├── detection.py        YOLOv8 person detector
│   ├── tracker.py          ByteTrack multi-object tracker
│   ├── analytics.py        Heatmap + dwell time + loitering alerts
│   ├── osint.py            OSINT privacy audit engine
│   ├── emotion.py          DeepFace emotion + age + gender
│   ├── reid.py             OSNet deep Re-ID module
│   ├── weapon.py           YOLOv8 weapon detection
│   └── nlquery.py          Groq NL query parser
├── models/
│   ├── osnet_phantomeye_reid.pth   Trained Re-ID weights
│   └── weapon_detector.pt          Trained weapon detector
├── api/
│   ├── main.py             FastAPI backend — 8 endpoints
│   └── routes/             Modular route handlers
├── app.py                  Streamlit dashboard — 7 modules
├── config.py               Global configuration
├── Dockerfile              Container deployment
└── requirements.txt        Dependencies

---

## Real-World Applications

| Domain | Use Case |
|:---|:---|
| Law Enforcement | Cross-camera suspect tracking, weapon threat detection, automated evidence extraction |
| Retail Intelligence | Customer heatmaps, queue monitoring, suspicious behavior detection |
| Campus Security | Unauthorized access detection, behavioral anomaly alerts |
| Healthcare | Patient wandering alerts, fall detection, ICU monitoring |
| Border Security | Weapon screening, person Re-ID across checkpoints |
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

- [x] YOLOv8 person detection — CPU optimized
- [x] ByteTrack multi-object tracking
- [x] Behavioral heatmap + loitering alerts
- [x] OSINT privacy audit engine
- [x] FastAPI production backend — 8 endpoints
- [x] Cyberpunk Streamlit dashboard
- [x] HuggingFace + Railway live deployment
- [x] DeepFace emotion intelligence module
- [x] Groq NL query engine — English + Roman Urdu
- [x] OSNet Deep Re-ID — Rank-1 81.7% on Market-1501
- [x] YOLOv8 weapon detection — 9 classes mAP50 53.2%
- [ ] PDF intelligence report generator
- [ ] JWT authentication + API key management
- [ ] RTSP live stream support
- [ ] Anonymization mode — face blur + full analytics
- [ ] Edge deployment — Raspberry Pi + Jetson Nano

---

## Author

<div align="center">

**Abu Sameer** — AI/ML Engineer · Computer Vision Researcher · GSoC 2026 Contributor

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
