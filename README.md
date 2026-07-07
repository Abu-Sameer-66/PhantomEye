---
title: PhantomEye
emoji: 👁
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:020408,50:0a1628,100:020408&height=280&section=header&text=👁%20PhantomEye&fontSize=72&fontColor=00b4ff&animation=fadeIn&fontAlignY=38&desc=AI-Powered%20Surveillance%20Intelligence%20System&descSize=17&descAlignY=58&descAlign=50&descColor=7ab3d4" width="100%"/>

<br/>

<img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=600&size=18&pause=1200&color=00B4FF&center=true&vCenter=true&width=950&lines=11+Intelligence+Modules+%7C+3+Novel+Research+Algorithms;Threat+Momentum+Score+%E2%80%94+Temporal+Compound+Threat+Accumulation;Behavioral+DNA+%E2%80%94+Camera-Agnostic+Re-ID+Without+Face+Recognition;Social+Graph+Intelligence+%E2%80%94+Group+Detection+From+Movement;YOLOv8+%7C+ByteTrack+%7C+DeepFace+%7C+Groq+LLaMA+3+%7C+OSNet;Runs+Entirely+on+CPU+%E2%80%94+No+GPU+Required" />

<br/><br/>

<a href="https://abu-sameer-66-phantomeye.hf.space">
<img src="https://img.shields.io/badge/🟢%20LIVE%20DEMO-HuggingFace%20Spaces-00b4ff?style=for-the-badge&labelColor=020408"/>
</a>
&nbsp;
<a href="https://abu-sameer-66-phantomeye.hf.space/docs">
<img src="https://img.shields.io/badge/📡%20API%20DOCS-FastAPI%20OAS%203.1-00fff0?style=for-the-badge&labelColor=020408"/>
</a>
&nbsp;
<a href="https://medium.com/@sameerdataanalyst66/i-built-an-ai-that-watches-tracks-and-audits-phantomeye-is-live-afe2f62bcb7b">
<img src="https://img.shields.io/badge/📖%20MEDIUM-Full%20Article-7ab3d4?style=for-the-badge&labelColor=020408"/>
</a>
&nbsp;
<a href="https://github.com/Abu-Sameer-66/PhantomEye/blob/main/LICENSE">
<img src="https://img.shields.io/badge/License-MIT-3a6080?style=for-the-badge&labelColor=020408"/>
</a>

<br/><br/>

<img src="https://img.shields.io/badge/Python-3.10-00b4ff?style=flat-square&logo=python&logoColor=white&labelColor=020408"/>
<img src="https://img.shields.io/badge/YOLOv8-Detection%20%2B%20Weapon-00b4ff?style=flat-square&logo=pytorch&logoColor=white&labelColor=020408"/>
<img src="https://img.shields.io/badge/ByteTrack-Multi--Object%20Tracking-00b4ff?style=flat-square&labelColor=020408"/>
<img src="https://img.shields.io/badge/OSNet-ReID%20Rank--1%2081.7%25-00b4ff?style=flat-square&logo=pytorch&logoColor=white&labelColor=020408"/>
<img src="https://img.shields.io/badge/DeepFace-Emotion%20Intelligence-00b4ff?style=flat-square&labelColor=020408"/>
<img src="https://img.shields.io/badge/Groq%20LLaMA3-NL%20Query%20Engine-00b4ff?style=flat-square&labelColor=020408"/>
<img src="https://img.shields.io/badge/FastAPI-REST%20API-00b4ff?style=flat-square&logo=fastapi&logoColor=white&labelColor=020408"/>
<img src="https://img.shields.io/badge/Docker-Containerized-00b4ff?style=flat-square&logo=docker&logoColor=white&labelColor=020408"/>
<img src="https://img.shields.io/badge/HuggingFace-Live%2024%2F7-00b4ff?style=flat-square&labelColor=020408"/>

</div>

---

## What is PhantomEye?

Most computer vision projects stop at detection. A box appears around a person. The box disappears. Nothing is remembered. Nothing is understood. Nothing is decided.

**PhantomEye goes further — much further.**

It is a full-stack AI surveillance intelligence platform that transforms passive camera feeds into a live reasoning engine. Eleven modules working in concert: detecting every person, tracking their identity across frames, mapping behavioral patterns, auditing privacy exposure, recognizing emotion, querying in natural language, detecting weapons, and — uniquely — running three novel research algorithms that no existing open-source system implements.

Built entirely from scratch. Trained on real datasets. Deployed live. Zero pre-loaded data. Runs on standard CPU.

---

## Novel Research Contributions

> These three algorithms represent original research. No existing open-source surveillance system implements any of them.

### Threat Momentum Score (TMS v1.0)

Binary threat detection is wrong by design. A person loitering alone is a weak signal. A person loitering while displaying stress emotion is stronger. The same person then entering a restricted zone while showing rapid movement and gaze anomaly — that is a different situation entirely.

TMS models threat as an accumulating value. Each behavioral signal compounds over time using a weighted amplifier model: when the score is already elevated, new signals contribute proportionally more. The score decays with a 45-second half-life when signals stop — modeling how real threat situations escalate gradually, not instantaneously.

```
TMS(t) = TMS(t-1) × decay_factor + Σ(signal_i × weight_i × amplifier)

where: amplifier = 1 + (TMS / 200)
       decay_factor = 0.5^(Δt / 45s)

Signals: loitering(0.28) · stress_emotion(0.22) · rapid_movement(0.18)
         proximity_violation(0.15) · gaze_anomaly(0.10) · group_formation(0.07)

Levels: CLEAR → LOW → MEDIUM → HIGH → CRITICAL
```

### Behavioral DNA Fingerprint (BDF v1.0)

Face recognition fails when subjects wear masks, hats, or are too distant. BDF identifies the same person across cameras using behavioral signature alone — a 5-component feature vector extracted purely from movement patterns.

When a person re-enters the scene under a different tracking ID, BDF matches them to their previous identity using cosine similarity. Tested: Person 3 matched Person 1 at 99.99% similarity after re-entry simulation.

```
BDF_vector = [
    gait_signature,        # stride rhythm histogram (10-bin)
    velocity_profile,      # speed distribution over time (10-bin)
    spatial_preference,    # normalized 8×8 grid heatmap of visited zones
    social_distance_avg,   # mean distance maintained from nearby persons
    dwell_zone_signature   # normalized 8×8 grid of stopping locations
]

Similarity: cosine(BDF_a, BDF_b)
Match threshold: 0.82
Min observations required: 15 frames
```

### Social Graph Intelligence (SGI v1.0)

Three bank robbers enter a building separately, minutes apart, from different entrances. Standard surveillance sees three unrelated individuals. SGI detects their association before any overt action occurs — purely from movement correlation.

```
Link_strength(A, B) =
    proximity_score × 0.40 +
    pearson_velocity_correlation × 0.35 +
    dwell_zone_overlap × 0.25

Group detection: BFS connected-component analysis on confirmed links
Alert condition: coordinated link_type within group of ≥ 2 persons
```

---

## Intelligence Modules

<table>
<tr>
<td width="50%">

### 🎯 Person Detection
YOLOv8-nano configured for class-0 only. Confidence threshold 0.4. Returns bounding boxes and confidence scores on standard CPU. No GPU required.

### 🔥 Behavioral Analytics
Custom ByteTrack with IOU matching assigns persistent IDs across frames through occlusion. NumPy heatmap accumulates movement density. Per-person dwell time tracked in seconds. Automated loitering alerts.

### 🕵️ OSINT Privacy Audit
Upload a face — receive a Privacy Exposure Score from 0 to 100. LBPH embedding matched against reference gallery using cosine similarity. Risk: LOW / MEDIUM / HIGH.

### 🧠 Emotion Intelligence
DeepFace pipeline detects age, gender, and dominant emotion per face from 7 classes. False-positive filter rejects faces smaller than 15% of frame area. Multi-subject support.

### 💬 NL Query Engine
Groq LLaMA 3 (llama-3.1-8b-instant) parses surveillance queries in English or Roman Urdu. Extracts structured filters: emotion, gender, age, dwell time, loitering. First open-source system with Roman Urdu surveillance query support.

### ⚠️ Weapon Detection
YOLOv8 custom trained on 714 real-world images across 9 classes. Immediate threat alert on detection with class name and confidence.

</td>
<td width="50%">

### 📊 Threat Momentum Score *(Novel)*
Temporal compound threat accumulation. 6 behavioral signals. Compound amplifier effect. 45-second decay half-life. 5 threat levels. See research contribution above.

### 🧬 Behavioral DNA *(Novel)*
Camera-agnostic person re-identification using 5-component behavioral feature vector. Works through masks, hats, low resolution, distance. Cosine similarity matching at 82% threshold.

### 🕸️ Social Graph Intelligence *(Novel)*
Real-time group detection from movement correlation alone. No prior information required. Proximity + velocity synchronization + shared dwell zones. BFS connected-component group extraction.

### 📄 Intel Report
One-click classified PDF generation using fpdf2. Dark background, green terminal-style text, weapon threat sections in red. CLASSIFIED header. Zero server-side storage.

### 🔍 Deep Person Re-ID
OSNet x0.25 trained from scratch on Market-1501 (12,936 images, 751 identities). **Rank-1: 81.7% — mAP: 58.5%.** Identifies the same person across camera networks using body appearance alone.

### ⚡ System Intel
Live dashboard showing all active modules, model benchmarks, API endpoint reference, and deployment metadata.

</td>
</tr>
</table>

---

## Benchmark Results

| Module | Model | Metric | Result |
|:---|:---|:---|:---:|
| Person Detection | YOLOv8-nano | Avg Confidence | >85% |
| Multi-Object Tracking | ByteTrack | ID Persistence | Across occlusion |
| Deep Re-ID | OSNet x0.25 | **Rank-1 Accuracy** | **81.7%** |
| Deep Re-ID | OSNet x0.25 | **mAP** | **58.5%** |
| Emotion Recognition | DeepFace | Face Detection | OpenCV backend |
| Weapon Detection | YOLOv8 Custom | **mAP50** | **53.2%** |
| Weapon Detection | YOLOv8 Custom | Handgun AP | 89.5% |
| Weapon Detection | YOLOv8 Custom | Shotgun AP | 96.3% |
| Weapon Detection | YOLOv8 Custom | SMG AP | 98.6% |
| NL Query Engine | Groq LLaMA 3 | Languages | English + Roman Urdu |
| BDF Re-ID | BDF v1.0 | Match Similarity | 99.99% (sim) |
| TMS | TMS v1.0 | Escalation | CLEAR → CRITICAL |

---

## Capability Comparison

| Capability | Typical CV Project | Commercial System | PhantomEye |
|:---|:---:|:---:|:---:|
| Person detection | ✅ | ✅ | ✅ YOLOv8-nano |
| Persistent ID tracking | ❌ | ✅ | ✅ ByteTrack |
| Behavioral heatmap | ❌ | partial | ✅ NumPy accumulation |
| Dwell time analytics | ❌ | partial | ✅ Per-person seconds |
| Loitering alert | ❌ | ✅ | ✅ Threshold-based |
| Deep Re-ID (no face) | ❌ | ✅ $$$  | ✅ OSNet 81.7% |
| Emotion recognition | ❌ | partial | ✅ DeepFace 7 classes |
| Weapon detection | ❌ | ✅ $$$ | ✅ 9-class custom |
| NL query interface | ❌ | ❌ | ✅ English + Roman Urdu |
| OSINT privacy audit | ❌ | ❌ | ✅ Score 0–100 |
| **Temporal threat scoring** | ❌ | ❌ | ✅ **TMS v1.0 (novel)** |
| **Behavioral re-ID** | ❌ | ❌ | ✅ **BDF v1.0 (novel)** |
| **Social group detection** | ❌ | ❌ | ✅ **SGI v1.0 (novel)** |
| Production REST API | ❌ | ✅ $$$ | ✅ FastAPI OAS 3.1 |
| Live 24/7 deployment | ❌ | ✅ $$$ | ✅ HuggingFace free |
| Open source | ❌ | ❌ | ✅ Full transparency |
| GPU required | N/A | ✅ | ❌ CPU only |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    INPUT LAYER                      │
│         Image Upload  /  Video File                 │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│                  VISION PIPELINE                    │
│                                                     │
│  YOLOv8-nano ──────── Person Detection              │
│       │                bbox + confidence            │
│  ByteTrack ─────────── Persistent ID Tracking       │
│       │                color trails + occlusion     │
│  OSNet x0.25 ────────── Deep Re-ID                  │
│                         Rank-1 81.7%                │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│              INTELLIGENCE LAYER                     │
│                                                     │
│  Behavioral Analytics   heatmap + dwell + loitering │
│  Emotion Intelligence   DeepFace age/gender/emotion │
│  Weapon Detection       YOLOv8 9-class custom       │
│  NL Query Engine        Groq LLaMA 3                │
│  OSINT Audit            LBPH embedding + gallery    │
│                                                     │
│  ── NOVEL RESEARCH ──────────────────────────────── │
│  TMS v1.0               temporal compound scoring   │
│  BDF v1.0               behavioral re-ID no face    │
│  SGI v1.0               implicit group detection    │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│                  OUTPUT LAYER                       │
│  FastAPI REST API        OAS 3.1 + CORS             │
│  Streamlit Dashboard     cyberpunk UI               │
│  PDF Intel Report        fpdf2 classified           │
└─────────────────────────────────────────────────────┘
```

---

## Live Deployment

| Service | Platform | URL | Status |
|:---|:---|:---|:---:|
| Interactive Dashboard | HuggingFace Spaces | [abu-sameer-66-phantomeye.hf.space](https://abu-sameer-66-phantomeye.hf.space) | 🟢 Live |
| REST API | HuggingFace (Docker) | [abu-sameer-66-phantomeye.hf.space/docs](https://abu-sameer-66-phantomeye.hf.space/docs) | 🟢 Live |

---

## API Reference

**Base URL:** `https://abu-sameer-66-phantomeye.hf.space`

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

```bash
curl -X POST "https://abu-sameer-66-phantomeye.hf.space/detect" \
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
# Streamlit dashboard — all 11 modules
streamlit run app.py

# FastAPI backend
python api/main.py

# Test individual core modules
python core/threat_momentum.py
python core/behavioral_dna.py
python core/social_graph.py
python core/detection.py
python core/weapon.py
```

---

## Repository Structure

```
PhantomEye/
├── core/
│   ├── detection.py          YOLOv8 person detector
│   ├── tracker.py            ByteTrack multi-object tracker
│   ├── analytics.py          Heatmap + dwell + loitering
│   ├── osint.py              OSINT privacy audit
│   ├── emotion.py            DeepFace emotion intelligence
│   ├── reid.py               OSNet deep Re-ID
│   ├── weapon.py             YOLOv8 weapon detection
│   ├── nlquery.py            Groq NL query parser
│   ├── reporter.py           fpdf2 PDF generator
│   ├── threat_momentum.py    TMS v1.0 — novel algorithm
│   ├── behavioral_dna.py     BDF v1.0 — novel algorithm
│   └── social_graph.py       SGI v1.0 — novel algorithm
├── models/
│   ├── osnet_phantomeye_reid.pth
│   └── weapon_detector.pt
├── api/
│   ├── main.py               FastAPI backend
│   └── routes/               Modular route handlers
├── app.py                    Streamlit dashboard — 11 modules
├── start.sh                  Dual-process startup (Streamlit + FastAPI)
├── Dockerfile                Container deployment
└── requirements.txt
```

---

## Real-World Applications

| Domain | Use Case |
|:---|:---|
| Law Enforcement | Cross-camera suspect tracking, weapon detection, behavioral profiling |
| Retail Intelligence | Customer flow heatmaps, queue monitoring, theft pattern detection |
| Campus Security | Unauthorized access, crowd anomaly, coordinated group detection |
| Healthcare | Patient wandering alerts, fall detection zone monitoring |
| Border Security | Weapon screening, behavioral re-ID across checkpoints |
| Privacy Research | Digital footprint auditing, OSINT defense and awareness |

---

## Privacy-First Design

- **Zero pre-loaded data** — no faces, videos, or images in the repository
- **In-session processing** — uploaded files processed in RAM only, never stored
- **User-controlled gallery** — only data you explicitly upload is referenced
- **Ethical framing** — OSINT module built for privacy defense, not offense
- **Full transparency** — every processing step is open source and auditable

---

## Roadmap

- [x] YOLOv8 person detection — CPU optimized
- [x] ByteTrack multi-object tracking
- [x] Behavioral heatmap + dwell time + loitering alerts
- [x] OSINT privacy audit — LBPH + exposure score
- [x] FastAPI production backend — OAS 3.1
- [x] HuggingFace Docker deployment — 24/7 live
- [x] DeepFace emotion intelligence
- [x] Groq NL query engine — English + Roman Urdu
- [x] OSNet Deep Re-ID — Rank-1 81.7% on Market-1501
- [x] YOLOv8 weapon detection — 9 classes mAP50 53.2%
- [x] PDF classified intelligence report — fpdf2
- [x] **Threat Momentum Score (TMS v1.0)** — novel algorithm
- [x] **Behavioral DNA Fingerprint (BDF v1.0)** — novel algorithm
- [x] **Social Graph Intelligence (SGI v1.0)** — novel algorithm
- [x] Dual-process deployment — Streamlit + FastAPI on HuggingFace
- [ ] Predictive Exit Vector — trajectory-based camera handoff
- [ ] RTSP live stream support — real IP camera input
- [ ] Anonymization mode — face blur with analytics preserved
- [ ] Research paper — IEEE Access submission
- [ ] Next.js frontend — Vercel deployment
- [ ] Edge AI — Raspberry Pi + Jetson Nano

---

## Author

<div align="center">

**Abu Sameer** — AI/ML Engineer · Computer Vision Researcher · Open Source Contributor

*BS Data Science, Islamia University of Bahawalpur (2023–2027)*
*Lead Deep Learning Researcher, IUB AI Research Lab*

<br/>

<a href="https://sameer-nadeem-portfolio.vercel.app">
<img src="https://img.shields.io/badge/Portfolio-sameer--nadeem--portfolio-00b4ff?style=for-the-badge&labelColor=020408"/>
</a>
<a href="https://github.com/Abu-Sameer-66">
<img src="https://img.shields.io/badge/GitHub-Abu--Sameer--66-00b4ff?style=for-the-badge&logo=github&labelColor=020408"/>
</a>
<a href="https://www.linkedin.com/in/sameer-nadeem-66339a357/">
<img src="https://img.shields.io/badge/LinkedIn-Sameer%20Nadeem-00b4ff?style=for-the-badge&logo=linkedin&labelColor=020408"/>
</a>
<a href="https://www.kaggle.com/sameernadeem66">
<img src="https://img.shields.io/badge/Kaggle-sameernadeem66-00b4ff?style=for-the-badge&logo=kaggle&labelColor=020408"/>
</a>
<a href="https://medium.com/@sameerdataanalyst66/i-built-an-ai-that-watches-tracks-and-audits-phantomeye-is-live-afe2f62bcb7b">
<img src="https://img.shields.io/badge/Medium-Full%20Article-00b4ff?style=for-the-badge&logo=medium&labelColor=020408"/>
</a>

</div>

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:020408,50:0a1628,100:020408&height=100&section=footer" width="100%"/>
</div>
