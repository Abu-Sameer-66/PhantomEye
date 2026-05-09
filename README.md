---
title: PhantomEye
emoji: 👁
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0a,50:1a0040,100:0d1f0d&height=300&section=header&text=ðŸ“·%20PhantomEye&fontSize=75&fontColor=00ff88&animation=fadeIn&fontAlignY=38&desc=AI-Powered%20Surveillance%20Intelligence%20System&descSize=18&descAlignY=58&descAlign=50&descColor=00aa55" width="100%"/>

<br/>

<img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=00FF88&center=true&vCenter=true&width=900&lines=Person+Detection+%7C+ByteTrack+Multi-Object+Tracking;Deep+ReID+OSNet+â€”+Rank-1+81.7%25+mAP+58.5%25;Emotion+Intelligence+â€”+Age+%7C+Gender+%7C+Emotion;Weapon+Detection+â€”+9+Classes+mAP50+53.2%25;Natural+Language+Queries+â€”+English+%2B+Roman+Urdu;OSINT+Privacy+Audit+â€”+Exposure+Score+0%E2%80%93100" />

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

It is a full-stack AI surveillance intelligence platform that transforms passive camera feeds into a live reasoning engine â€” detecting, tracking, analyzing behavior, auditing identity, recognizing emotion, querying in natural language, and detecting weapons, all in one unified system.

Built entirely from scratch. Trained on real datasets. Deployed live. Zero pre-loaded data.

---

## Intelligence Modules

<table>
<tr>
<td width="50%">

### ðŸ“· Person Detection
YOLOv8-nano configured for class-0 only. Returns bounding boxes and confidence scores on standard CPU in milliseconds. No GPU required at inference time.

### ðŸŽ¯ Multi-Object Tracking
Custom ByteTrack with IOU matching. Each person receives a persistent color-coded ID with trajectory trail â€” across frames, through occlusion, across re-entries.

### ðŸ”¥ Behavioral Heatmap
NumPy position accumulation builds a live heatmap of human movement. High-activity zones appear red. Dwell time tracked per person in seconds. Loitering alerts fire automatically.

### ðŸ§  Deep Person Re-ID
OSNet x0.25 trained from scratch on Market-1501 (12,936 images, 751 identities). **Rank-1: 81.7% â€” mAP: 58.5%.** Identifies the same person across camera networks using body appearance alone â€” no face required.

</td>
<td width="50%">

### ðŸ˜¶ Emotion Intelligence
DeepFace pipeline â€” detects age, gender, and dominant emotion per face. Powered by TensorFlow with OpenCV face detector backend. Optimized for CPU deployment.

### ðŸ’¬ Natural Language Query Engine
Groq LLaMA 3 powered query parser. Ask questions in plain English or Roman Urdu â€” the system extracts structured filters automatically. First open-source surveillance system with multilingual NL query support.

### ðŸ”« Weapon Detection
YOLOv8 custom trained on 9 weapon classes â€” Handgun, Knife, Shotgun, Sniper, Automatic Rifle, SMG, Sword, Bazooka, Grenade Launcher. **mAP50: 53.2% â€” Handgun: 89.5% â€” Shotgun: 96.3% â€” SMG: 98.6%.** Real-time threat alert on detection.

### ðŸ” OSINT Privacy Audit
Upload a face â€” get a Privacy Exposure Score from 0 to 100. LBPH embedding search against a reference gallery. Risk classification: LOW / MEDIUM / HIGH.

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
| Person detection | âœ… | âœ… YOLOv8-nano |
| Persistent ID tracking | âŒ | âœ… ByteTrack |
| Behavioral heatmap | âŒ | âœ… NumPy accumulation |
| Dwell time analytics | âŒ | âœ… Per-person seconds |
| Loitering alert | âŒ | âœ… Threshold-based |
| Deep Re-ID (no face) | âŒ | âœ… OSNet Rank-1 81.7% |
| Emotion recognition | âŒ | âœ… DeepFace â€” Age + Gender + Emotion |
| Weapon detection | âŒ | âœ… 9-class YOLOv8 custom |
| NL query interface | âŒ | âœ… English + Roman Urdu |
| OSINT privacy audit | âŒ | âœ… Score 0â€“100 |
| Production REST API | âŒ | âœ… 8 endpoints, OAS 3.1 |
| Live 24/7 deployment | âŒ | âœ… HuggingFace + Railway |
| Zero pre-loaded data | âŒ | âœ… Privacy-first |

---

## System Architecture
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                     INPUT LAYER                      â”‚
â”‚    Image Upload  /  Video File  /  RTSP Feed         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                  VISION PIPELINE                     â”‚
â”‚                                                      â”‚
â”‚  YOLOv8-nano â”€â”€â”€â”€â”€â”€â”€ Person Detection                â”‚
â”‚       â”‚               bbox + confidence              â”‚
â”‚       â”‚                                              â”‚
â”‚  ByteTrack â”€â”€â”€â”€â”€â”€â”€â”€ Persistent ID Assignment         â”‚
â”‚       â”‚              Color trails + occlusion        â”‚
â”‚       â”‚                                              â”‚
â”‚  OSNet x0.25 â”€â”€â”€â”€â”€â”€ Deep Person Re-ID                â”‚
â”‚                      Rank-1 81.7% on Market-1501     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚               INTELLIGENCE LAYER                     â”‚
â”‚                                                      â”‚
â”‚  Behavioral Analytics                                â”‚
â”‚  â”œâ”€â”€ NumPy heatmap accumulation                      â”‚
â”‚  â”œâ”€â”€ Per-person dwell time (seconds)                 â”‚
â”‚  â””â”€â”€ Automated loitering alerts                      â”‚
â”‚                                                      â”‚
â”‚  Emotion Intelligence                                â”‚
â”‚  â”œâ”€â”€ DeepFace â€” Age + Gender + Emotion               â”‚
â”‚  â””â”€â”€ OpenCV detector backend (CPU optimized)         â”‚
â”‚                                                      â”‚
â”‚  Weapon Detection                                    â”‚
â”‚  â”œâ”€â”€ YOLOv8 custom â€” 9 weapon classes                â”‚
â”‚  â””â”€â”€ Real-time threat alert on detection             â”‚
â”‚                                                      â”‚
â”‚  NL Query Engine                                     â”‚
â”‚  â”œâ”€â”€ Groq LLaMA 3 â€” query parser                     â”‚
â”‚  â””â”€â”€ English + Roman Urdu â†’ structured filters       â”‚
â”‚                                                      â”‚
â”‚  OSINT Audit Engine                                  â”‚
â”‚  â”œâ”€â”€ LBPH face embedding extraction                  â”‚
â”‚  â”œâ”€â”€ Gallery similarity search                       â”‚
â”‚  â””â”€â”€ Exposure score (0â€“100) + risk level             â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   OUTPUT LAYER                       â”‚
â”‚                                                      â”‚
â”‚  FastAPI REST API    â”€â”€ 8 endpoints, OAS 3.1         â”‚
â”‚  Streamlit Dashboard â”€â”€ Cyberpunk UI                 â”‚
â”‚  JSON Reports        â”€â”€ Exportable audit logs        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

---

## Live Deployment

| Service | Platform | Status |
|:---|:---|:---:|
| [Interactive Dashboard](https://abu-sameer-66-phantomeye.hf.space) | HuggingFace Spaces | ðŸŸ¢ Live |
| [REST API](https://phantomeye-production.up.railway.app) | Railway | ðŸŸ¢ Live |
| [API Documentation](https://phantomeye-production.up.railway.app/docs) | Railway | ðŸŸ¢ Live |

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
â”œâ”€â”€ core/
â”‚   â”œâ”€â”€ detection.py        YOLOv8 person detector
â”‚   â”œâ”€â”€ tracker.py          ByteTrack multi-object tracker
â”‚   â”œâ”€â”€ analytics.py        Heatmap + dwell time + loitering alerts
â”‚   â”œâ”€â”€ osint.py            OSINT privacy audit engine
â”‚   â”œâ”€â”€ emotion.py          DeepFace emotion + age + gender
â”‚   â”œâ”€â”€ reid.py             OSNet deep Re-ID module
â”‚   â”œâ”€â”€ weapon.py           YOLOv8 weapon detection
â”‚   â””â”€â”€ nlquery.py          Groq NL query parser
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ osnet_phantomeye_reid.pth   Trained Re-ID weights
â”‚   â””â”€â”€ weapon_detector.pt          Trained weapon detector
â”œâ”€â”€ api/
â”‚   â”œâ”€â”€ main.py             FastAPI backend â€” 8 endpoints
â”‚   â””â”€â”€ routes/             Modular route handlers
â”œâ”€â”€ app.py                  Streamlit dashboard â€” 7 modules
â”œâ”€â”€ config.py               Global configuration
â”œâ”€â”€ Dockerfile              Container deployment
â””â”€â”€ requirements.txt        Dependencies

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

- **Zero pre-loaded data** â€” no faces, videos, or images in the repository
- **In-session processing** â€” uploaded files processed in RAM only, never stored
- **User-controlled gallery** â€” only data you explicitly upload is referenced
- **Ethical OSINT framing** â€” audit module built for privacy defense, not offense
- **Fully open source** â€” every processing step is transparent and auditable

---

## Roadmap

- [x] YOLOv8 person detection â€” CPU optimized
- [x] ByteTrack multi-object tracking
- [x] Behavioral heatmap + loitering alerts
- [x] OSINT privacy audit engine
- [x] FastAPI production backend â€” 8 endpoints
- [x] Cyberpunk Streamlit dashboard
- [x] HuggingFace + Railway live deployment
- [x] DeepFace emotion intelligence module
- [x] Groq NL query engine â€” English + Roman Urdu
- [x] OSNet Deep Re-ID â€” Rank-1 81.7% on Market-1501
- [x] YOLOv8 weapon detection â€” 9 classes mAP50 53.2%
- [ ] PDF intelligence report generator
- [ ] JWT authentication + API key management
- [ ] RTSP live stream support
- [ ] Anonymization mode â€” face blur + full analytics
- [ ] Edge deployment â€” Raspberry Pi + Jetson Nano

---

## Author

<div align="center">

**Abu Sameer** â€” AI/ML Engineer Â· Computer Vision Researcher Â· GSoC 2026 Contributor

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


