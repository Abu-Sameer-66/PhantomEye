import cv2
import sys
import time
import numpy as np
import streamlit as st
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from core.detection import PersonDetector
from core.tracker import ByteTracker
from core.analytics import BehavioralAnalyzer
from core.osint import OSINTAudit

st.set_page_config(
    page_title="PhantomEye — AI Surveillance Intelligence",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500;600&family=Exo+2:wght@100;200;300;400;700;900&display=swap');

:root {
    --bg-primary: #020408;
    --bg-secondary: #050d15;
    --bg-card: rgba(6, 18, 32, 0.85);
    --bg-glass: rgba(0, 180, 255, 0.04);
    --accent-blue: #00b4ff;
    --accent-cyan: #00fff0;
    --accent-amber: #ffb300;
    --accent-red: #ff3355;
    --accent-green: #00ff88;
    --border-glow: rgba(0, 180, 255, 0.3);
    --border-subtle: rgba(0, 180, 255, 0.12);
    --text-primary: #e8f4ff;
    --text-secondary: #7ab3d4;
    --text-dim: #3a6080;
    --grid-color: rgba(0, 180, 255, 0.04);
    --shadow-blue: 0 0 40px rgba(0, 180, 255, 0.15);
    --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.6);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background:
        radial-gradient(ellipse at 20% 50%, rgba(0, 60, 120, 0.15) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(0, 30, 80, 0.2) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(0, 40, 100, 0.1) 0%, transparent 60%),
        linear-gradient(180deg, #020408 0%, #030a14 100%) !important;
    min-height: 100vh;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(var(--grid-color) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-color) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ── HERO / LANDING ─────────────────────────────── */
.hero-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 92vh; padding: 3rem 1rem;
    position: relative;
}

.hero-wrap::before {
    content: '';
    position: absolute;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(0, 180, 255, 0.08) 0%, transparent 70%);
    border-radius: 50%;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    animation: pulse-glow 4s ease-in-out infinite;
}

@keyframes pulse-glow {
    0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.6; }
    50% { transform: translate(-50%, -50%) scale(1.1); opacity: 1; }
}

.hero-eye {
    font-size: 5rem; margin-bottom: 1.5rem;
    animation: float 5s ease-in-out infinite;
    filter: drop-shadow(0 0 30px rgba(0, 180, 255, 0.8));
}

@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(-2deg); }
    50% { transform: translateY(-20px) rotate(2deg); }
}

.hero-title {
    font-family: 'Exo 2', sans-serif;
    font-size: clamp(3.5rem, 8vw, 7rem);
    font-weight: 900;
    letter-spacing: 0.15em;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-blue) 40%, var(--accent-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    margin-bottom: 0.5rem;
    animation: title-reveal 1s ease-out forwards;
}

@keyframes title-reveal {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: clamp(0.9rem, 2vw, 1.1rem);
    font-weight: 300;
    letter-spacing: 0.4em;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
    text-transform: uppercase;
}

.hero-status {
    font-size: 0.7rem;
    color: var(--accent-green);
    letter-spacing: 0.3em;
    margin-bottom: 3rem;
    opacity: 0.8;
}

.hero-status::before {
    content: '● ';
    animation: blink 1.5s infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.2; }
}

/* ── MODULE GRID ─────────────────────────────────── */
.module-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    width: 100%;
    max-width: 1100px;
    margin: 0 auto 3rem;
}

.mod-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 2rem 1.8rem;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
    backdrop-filter: blur(20px);
}

.mod-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}

.mod-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at top left, rgba(0, 180, 255, 0.08) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.4s;
}

.mod-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-6px);
    box-shadow: var(--shadow-blue), var(--shadow-card);
}

.mod-card:hover::before { opacity: 1; }
.mod-card:hover::after { opacity: 1; }

.mod-icon {
    font-size: 1.8rem;
    margin-bottom: 1rem;
    color: var(--accent-blue);
    display: block;
}

.mod-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    color: var(--accent-blue);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

.mod-desc {
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.7;
    letter-spacing: 0.02em;
}

/* ── SCAN LINE ───────────────────────────────────── */
.scan-line {
    width: 100%;
    max-width: 900px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), var(--accent-blue), transparent);
    margin: 2rem auto;
    position: relative;
    overflow: hidden;
}

.scan-line::after {
    content: '';
    position: absolute;
    width: 80px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 240, 0.8), transparent);
    animation: scan 3s linear infinite;
}

@keyframes scan {
    from { left: -80px; }
    to { left: 100%; }
}

/* ── APP HEADER ──────────────────────────────────── */
.app-header {
    font-family: 'Exo 2', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.3em;
    background: linear-gradient(135deg, #fff, var(--accent-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}

.app-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    color: var(--text-dim);
    letter-spacing: 0.4em;
    text-align: center;
    margin-bottom: 2rem;
    text-transform: uppercase;
}

/* ── NAV BUTTONS ─────────────────────────────────── */
.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    font-size: 0.82rem !important;
    background: var(--bg-card) !important;
    color: var(--accent-blue) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.2rem !important;
    transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1) !important;
    position: relative !important;
    overflow: hidden !important;
    text-transform: uppercase !important;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0, 180, 255, 0.15), transparent);
    transition: left 0.4s;
}

.stButton > button:hover {
    background: rgba(0, 180, 255, 0.1) !important;
    border-color: var(--accent-blue) !important;
    color: var(--accent-cyan) !important;
    box-shadow: 0 0 20px rgba(0, 180, 255, 0.2), inset 0 0 20px rgba(0, 180, 255, 0.05) !important;
    transform: translateY(-2px) !important;
}

.stButton > button:hover::before { left: 100%; }

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0, 100, 200, 0.4), rgba(0, 180, 255, 0.2)) !important;
    border-color: var(--accent-blue) !important;
    color: #fff !important;
    box-shadow: 0 0 20px rgba(0, 180, 255, 0.2) !important;
}

/* ── SECTION HEADERS ─────────────────────────────── */
.section-hdr {
    font-family: 'Exo 2', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    color: var(--accent-blue);
    text-transform: uppercase;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 0.5rem;
    position: relative;
}

.section-hdr::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0;
    width: 80px; height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
}

.section-sub {
    font-size: 0.75rem;
    color: var(--text-secondary);
    letter-spacing: 0.15em;
    margin-bottom: 2rem;
    text-transform: uppercase;
}

/* ── TERMINAL STATUS BAR ─────────────────────────── */
.terminal {
    background: rgba(0, 10, 20, 0.9);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--accent-blue);
    border-radius: 6px;
    padding: 0.8rem 1.2rem;
    font-size: 0.72rem;
    color: var(--accent-green);
    letter-spacing: 0.15em;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}

.terminal::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 255, 136, 0.015) 2px,
        rgba(0, 255, 136, 0.015) 4px
    );
    pointer-events: none;
}

/* ── STREAMLIT OVERRIDES ─────────────────────────── */
.stFileUploader {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-glow) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

.stTextInput > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.stTextInput > div > div:focus-within {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 15px rgba(0, 180, 255, 0.15) !important;
}

.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

.stNumberInput > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
}

.stSlider > div > div > div {
    background: var(--accent-blue) !important;
}

div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

div[data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.2em !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
}

div[data-testid="metric-container"] div[data-testid="metric-value"] {
    color: var(--accent-blue) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 700 !important;
}

div[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

.stSuccess {
    background: rgba(0, 255, 136, 0.08) !important;
    border: 1px solid rgba(0, 255, 136, 0.3) !important;
    border-radius: 8px !important;
    color: var(--accent-green) !important;
}

.stError, .stWarning {
    background: rgba(255, 51, 85, 0.08) !important;
    border: 1px solid rgba(255, 51, 85, 0.3) !important;
    border-radius: 8px !important;
}

.stInfo {
    background: rgba(0, 180, 255, 0.08) !important;
    border: 1px solid rgba(0, 180, 255, 0.2) !important;
    border-radius: 8px !important;
    color: var(--accent-blue) !important;
}

hr {
    border-color: var(--border-subtle) !important;
    margin: 1.5rem 0 !important;
}

/* ── SCROLLBAR ───────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: var(--accent-blue);
    border-radius: 2px;
    opacity: 0.5;
}

/* ── SPINNER ─────────────────────────────────────── */
.stSpinner > div {
    border-color: var(--accent-blue) transparent transparent transparent !important;
}

/* ── SIDEBAR HIDE ────────────────────────────────── */
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ── INITIALIZE BUTTON ───────────────────────────── */
.init-btn {
    display: inline-block;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.3em;
    color: var(--accent-cyan);
    background: transparent;
    border: 1px solid var(--accent-blue);
    border-radius: 6px;
    padding: 0.8rem 2.5rem;
    cursor: pointer;
    text-transform: uppercase;
    position: relative;
    overflow: hidden;
    transition: all 0.3s;
}

.init-btn:hover {
    background: rgba(0, 180, 255, 0.1);
    box-shadow: 0 0 30px rgba(0, 180, 255, 0.3);
}

/* ── PRO BADGE ───────────────────────────────────── */
.pro-badge {
    display: inline-block;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    color: var(--accent-amber);
    background: rgba(255, 179, 0, 0.1);
    border: 1px solid rgba(255, 179, 0, 0.4);
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    text-transform: uppercase;
    margin-left: 0.5rem;
    vertical-align: middle;
}

/* ── ANIMATIONS ──────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.stMarkdown, .stButton, .stFileUploader {
    animation: fadeInUp 0.4s ease-out forwards;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    return PersonDetector()

@st.cache_resource
def load_osint():
    return OSINTAudit()


def landing():
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-eye">👁</div>
      <div class="hero-title">PHANTOMEYE</div>
      <div class="hero-sub">AI-POWERED SURVEILLANCE INTELLIGENCE SYSTEM</div>
      <div class="hero-status">[ SYSTEM ONLINE ] · CLASSIFIED · BUILD v1.0.0</div>
      <div class="scan-line"></div>
      <div class="module-grid">
        <div class="mod-card">
          <div class="mod-icon">⬡</div>
          <div class="mod-name">Person Detection</div>
          <div class="mod-desc">YOLOv8-nano detects every person in any image with confidence scores and bounding boxes.</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">⬡</div>
          <div class="mod-name">Behavioral Analytics</div>
          <div class="mod-desc">Real-time heatmap, dwell time tracking, and automated loitering alerts from video feeds.</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">⬡</div>
          <div class="mod-name">OSINT Audit</div>
          <div class="mod-desc">Upload a face — get a privacy exposure score from 0 to 100 with matched identities.</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">⬡</div>
          <div class="mod-name">Emotion Intelligence</div>
          <div class="mod-desc">DeepFace powered age, gender and emotion recognition on any face image.</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">⬡</div>
          <div class="mod-name">NL Query Engine</div>
          <div class="mod-desc">Ask questions in plain English or Roman Urdu — AI extracts filters automatically.</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">⬡</div>
          <div class="mod-name">Weapon Detection</div>
          <div class="mod-desc">YOLOv8 custom trained on 9 weapon classes — Handgun, Shotgun, SMG, Rifle and more.</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">⬡</div>
          <div class="mod-name">System Intel</div>
          <div class="mod-desc">Live system status, module health, API endpoints, and deployment information.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns([1, 2, 1])
    with cols[1]:
        if st.button("INITIALIZE SYSTEM  →", key="enter_btn"):
            st.session_state.page = "home"
            st.rerun()


def home():
    st.markdown('<div class="app-header">👁 PHANTOMEYE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-sub">SELECT INTELLIGENCE MODULE</div>',
        unsafe_allow_html=True
    )

    modules = [
        ("DETECTION",  "Person Detection"),
        ("ANALYTICS",  "Behavioral Analytics"),
        ("OSINT",      "OSINT Audit"),
        ("EMOTION",    "Emotion Intelligence"),
        ("NL QUERY",  "NL Query Engine"),
        ("WEAPON",    "Weapon Detection"),
        ("REPORT",    "Intel Report"),
        ("INTEL",      "System Intel"),
    ]
    cols = st.columns(len(modules))
    for i, (key, label) in enumerate(modules):
        with cols[i]:
            if st.button(label, key=f"mod_{key}"):
                st.session_state.page = key
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="terminal">All modules online · YOLOv8 loaded · '
        'ByteTrack active · OSINT gallery ready · DeepFace online</div>',
        unsafe_allow_html=True
    )


def back_button():
    if st.button("← BACK TO MODULES"):
        st.session_state.page = "home"
        st.rerun()


def detection_page():
    render_trust_bar()
    back_button()
    st.markdown('<div class="section-hdr">Person Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="terminal">YOLOv8-nano · CPU inference · '
        'upload any image to detect persons</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], key="det_up")

    if uploaded:
        data  = np.frombuffer(uploaded.read(), np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Cannot decode image.")
            return

        with st.spinner("SCANNING..."):
            detector   = load_detector()
            t0         = time.time()
            detections = detector.detect(image)
            elapsed    = round(time.time() - t0, 3)
            annotated  = detector.draw(image, detections)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PERSONS",   len(detections))
        c2.metric("TIME",      f"{elapsed}s")
        c3.metric("MODEL",     "YOLOv8n")
        c4.metric("DEVICE",    "CPU")

        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption="DETECTION OUTPUT", use_container_width=True
        )

        if detections:
            st.markdown('<div class="section-hdr">Detection Log</div>', unsafe_allow_html=True)
            for i, d in enumerate(detections):
                with st.expander(f"PERSON_{i+1:03d}  CONF:{d['confidence']}"):
                    st.json({"id": i+1, "bbox": list(d["bbox"]), "confidence": d["confidence"]})


def analytics_page():
    render_trust_bar()
    back_button()
    st.markdown('<div class="section-hdr">Behavioral Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="terminal">Upload video · heatmap · dwell time · loitering alerts</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("", type=["mp4", "avi", "mov"], key="ana_up")

    if uploaded:
        tmp = Path("outputs") / f"tmp_{int(time.time())}.mp4"
        tmp.parent.mkdir(exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(uploaded.read())

        cap   = cv2.VideoCapture(str(tmp))
        fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        st.markdown(
            f'<div class="terminal">{w}x{h} @ {fps}fps · {total} frames loaded</div>',
            unsafe_allow_html=True
        )

        if st.button("RUN ANALYSIS"):
            detector = load_detector()
            tracker  = ByteTracker()
            analyzer = BehavioralAnalyzer(w, h, fps)
            cap      = cv2.VideoCapture(str(tmp))
            limit    = min(total, fps * 15)
            prog     = st.progress(0)
            stat     = st.empty()

            for i in range(limit):
                ret, frame = cap.read()
                if not ret: break
                dets   = detector.detect(frame)
                active = tracker.update(dets)
                analyzer.update(active)
                prog.progress(int((i / limit) * 100))
                if i % 25 == 0:
                    stat.markdown(
                        f'<div class="terminal">FRAME {i}/{limit} · ACTIVE: {len(active)}</div>',
                        unsafe_allow_html=True
                    )

            cap.release()
            tmp.unlink(missing_ok=True)
            prog.progress(100)
            stat.empty()

            s = analyzer.summary()
            st.success("ANALYSIS COMPLETE")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PERSONS",    s.get("total_persons", 0))
            c2.metric("AVG DWELL",  f"{s.get('avg_dwell_sec', 0)}s")
            c3.metric("MAX DWELL",  f"{s.get('max_dwell_sec', 0)}s")
            c4.metric("ALERTS",     s.get("total_alerts", 0))

            if s.get("total_alerts", 0) > 0:
                st.warning(f"LOITERING ALERT — IDs: {s.get('loiterers', [])}")

            heat = analyzer.get_heatmap_overlay(np.zeros((h, w, 3), dtype=np.uint8))
            st.image(
                cv2.cvtColor(heat, cv2.COLOR_BGR2RGB),
                caption="BEHAVIORAL HEATMAP — RED = HIGH ACTIVITY",
                use_container_width=True
            )


def osint_page():
    render_trust_bar()
    if not pro_gate("OSINT AUDIT"): return
    back_button()
    st.markdown('<div class="section-hdr">OSINT Privacy Audit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="terminal">Upload face photo · privacy exposure score · '
        'gallery match · risk report</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        query_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="osint_up")
    with c2:
        osint = load_osint()
        st.metric("GALLERY", f"{len(osint.gallery)} persons")
        st.metric("ENGINE", "LBPH Face")

    if query_file and st.button("EXECUTE AUDIT"):
        data  = np.frombuffer(query_file.read(), np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Cannot decode image.")
            return

        with st.spinner("RUNNING AUDIT..."):
            result = osint.audit(image, query_id=Path(query_file.name).stem)

        risk    = result["risk_level"]
        score   = result["exposure_score"]
        matches = result["matches"]

        c1, c2, c3 = st.columns(3)
        c1.metric("RISK LEVEL",     risk)
        c2.metric("EXPOSURE SCORE", f"{score}/100")
        c3.metric("MATCHES",        len(matches))

        st.markdown(
            f'<div class="terminal">{result["message"]}</div>',
            unsafe_allow_html=True
        )

        if matches:
            st.markdown('<div class="section-hdr">Match Log</div>', unsafe_allow_html=True)
            for m in matches:
                st.markdown(
                    f'<div class="terminal">MATCH: {m["matched_id"]} · '
                    f'CONF: {m["confidence"]}% · SRC: {m["source"]}</div>',
                    unsafe_allow_html=True
                )

        vis = osint.visualize(image, result)
        st.image(
            cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
            caption="OSINT VISUALIZATION",
            use_container_width=True
        )


def intel_page():
    render_trust_bar()
    back_button()
    st.markdown('<div class="section-hdr">System Intelligence</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SYSTEM",   "PhantomEye")
    c2.metric("VERSION",  "1.0.0")
    c3.metric("STATUS",   "ONLINE")
    c4.metric("MODULES",  "5 ACTIVE")

    st.markdown("<br>", unsafe_allow_html=True)

    modules = [
        ("DETECTION",  "YOLOv8-nano",    "Person detection on any image or video"),
        ("EMOTION",    "DeepFace + TF",  "Age · Gender · Emotion recognition per face"),
        ("NL QUERY",   "Groq LLaMA 3",   "Natural language queries — English + Roman Urdu"),
        ("WEAPON",     "YOLOv8 Custom",  "9-class weapon detection — mAP50 53.2%"),
        ("REPORT",     "fpdf2",          "Branded PDF intelligence report -- one click export"),
        ("TRACKING",   "ByteTrack",       "Persistent ID tracking across frames"),
        ("ANALYTICS",  "NumPy + OpenCV",  "Heatmap · dwell time · loitering alerts"),
        ("OSINT",      "LBPH Face",       "Privacy exposure scoring + gallery match"),
        ("API",        "FastAPI",          "8 endpoints · OAS 3.1 · port 8000"),
    ]

    for name, tech, desc in modules:
        with st.expander(f"{name}  ·  {tech}  ·  ACTIVE"):
            st.markdown(f'<div class="terminal">{desc}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.json({
        "author"  : "Abu-Sameer-66",
        "github"  : "https://github.com/Abu-Sameer-66/PhantomEye",
        "stack"   : ["Python 3.10", "YOLOv8", "OpenCV", "FastAPI", "Streamlit"],
        "api"     : "http://localhost:8000/docs",
        "status"  : "online",
    })



@st.cache_resource
def load_emotion_model():
    from core.emotion import process_frame_emotion
    return process_frame_emotion

def emotion_page():
    render_trust_bar()
    process_frame_emotion = load_emotion_model()
    if st.button("← BACK TO MODULES"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown('<div class="section-hdr">EMOTION INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Age · Gender · Emotion recognition powered by DeepFace</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded:
        import numpy as np
        from PIL import Image
        img = Image.open(uploaded).convert("RGB")
        frame = np.array(img)
        frame_bgr = frame[:, :, ::-1].copy()

        with st.spinner("Analyzing faces..."):
            annotated, results = process_frame_emotion(frame_bgr)

        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, caption="ORIGINAL", use_container_width=True)
        with col2:
            annotated_rgb = annotated[:, :, ::-1]
            st.image(annotated_rgb, caption="EMOTION ANALYSIS", use_container_width=True)

        if results:
            st.markdown("---")
            st.markdown("### DETECTED SUBJECTS")
            for i, r in enumerate(results):
                emotion = r.get("dominant_emotion", "N/A").upper()
                age = int(r.get("age", 0))
                gender = r.get("dominant_gender", r.get("gender", "N/A"))
                if isinstance(gender, dict):
                    gender = max(gender, key=gender.get)

                c1, c2, c3 = st.columns(3)
                c1.metric("EMOTION", emotion)
                c2.metric("AGE", f"{age} yrs")
                c3.metric("GENDER", gender.upper())
        else:
            st.warning("No faces detected in this image.")
    else:
        st.info("Upload a face image to begin emotion analysis.")



def nlquery_page():
    render_trust_bar()
    from core.nlquery import parse_nl_query, apply_filters
    if st.button("← BACK TO MODULES"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown('<div class="section-hdr">NL QUERY ENGINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ask questions in English or Roman Urdu — AI extracts filters</div>', unsafe_allow_html=True)

    query = st.text_input("Enter your query", placeholder="e.g. show me all angry men | log jo loiter kar rahy thy")

    if query:
        with st.spinner("Analyzing query..."):
            result = parse_nl_query(query)

        if result['success']:
            filters = result['filters']
            st.success(f"Understood: {filters['summary']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("EMOTION", filters['emotion'] or "ANY")
            col2.metric("GENDER", filters['gender'] or "ANY")
            col3.metric("MAX AGE", filters['max_age'] or "ANY")

            col4, col5 = st.columns(2)
            col4.metric("LOITERING", "YES" if filters['loitering'] else "ANY")
            col5.metric("MIN DWELL", f"{filters['min_dwell_seconds']}s" if filters['min_dwell_seconds'] else "ANY")

            st.markdown("---")
            st.markdown("### SIMULATE AGAINST SAMPLE DATA")

            sample_records = [
                {"id": 1, "emotion": "angry", "gender": "Man", "age": 28, "dwell_seconds": 45, "loitering": False},
                {"id": 2, "emotion": "neutral", "gender": "Woman", "age": 22, "dwell_seconds": 180, "loitering": True},
                {"id": 3, "emotion": "happy", "gender": "Man", "age": 35, "dwell_seconds": 20, "loitering": False},
                {"id": 4, "emotion": "angry", "gender": "Man", "age": 41, "dwell_seconds": 200, "loitering": True},
                {"id": 5, "emotion": "sad", "gender": "Woman", "age": 19, "dwell_seconds": 90, "loitering": False},
                {"id": 6, "emotion": "fear", "gender": "Man", "age": 26, "dwell_seconds": 310, "loitering": True},
            ]

            matched = apply_filters(sample_records, filters)

            if matched:
                st.success(f"{len(matched)} subject(s) matched out of {len(sample_records)}")
                import pandas as pd
                df = pd.DataFrame(matched)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No subjects matched this query in sample data.")
        else:
            st.error(f"Query parse failed: {result['error']}")
    else:
        st.info("Type a query above — English or Roman Urdu both work.")



@st.cache_resource
def load_weapon_model_cached():
    from core.weapon import load_weapon_model
    return load_weapon_model()

def weapon_page():
    render_trust_bar()
    if not pro_gate("WEAPON DETECTION"): return
    if st.button("← BACK TO MODULES"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown('<div class="section-hdr">WEAPON DETECTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Real-time weapon detection — Handgun · Knife · Shotgun · SMG · Rifle · Sword</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded:
        import numpy as np
        from PIL import Image
        from core.weapon import detect_weapons

        img = Image.open(uploaded).convert("RGB")
        frame = np.array(img)
        frame_bgr = frame[:, :, ::-1].copy()

        model = load_weapon_model_cached()

        with st.spinner("Scanning for weapons..."):
            annotated, detections = detect_weapons(frame_bgr, model)

        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, caption="ORIGINAL", use_container_width=True)
        with col2:
            annotated_rgb = annotated[:, :, ::-1]
            st.image(annotated_rgb, caption="THREAT ANALYSIS", use_container_width=True)

        st.markdown("---")
        if detections:
            st.error(f"⚠ THREAT DETECTED — {len(detections)} weapon(s) found!")
            st.markdown("### DETECTED THREATS")
            for d in detections:
                c1, c2 = st.columns(2)
                c1.metric("WEAPON CLASS", d['class_name'])
                c2.metric("CONFIDENCE", f"{d['confidence']:.0%}")
        else:
            st.success("✓ NO WEAPONS DETECTED — Scene clear")
    else:
        st.info("Upload an image to scan for weapons.")



def report_page():
    render_trust_bar()
    if not pro_gate("INTELLIGENCE REPORT"): return
    from core.reporter import generate_report
    if st.button("<- BACK TO MODULES"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown('<div class="section-hdr">INTELLIGENCE REPORT</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Generate a branded PDF intelligence report from session data</div>', unsafe_allow_html=True)

    st.markdown("### SESSION DATA")
    col1, col2 = st.columns(2)
    with col1:
        session_id = st.text_input("Session ID", value="PE-SESSION-001")
        total_persons = st.number_input("Total Persons", min_value=0, value=5)
        duration = st.number_input("Duration (seconds)", min_value=0, value=300)
    with col2:
        loitering_alerts = st.number_input("Loitering Alerts", min_value=0, value=1)
        nl_query = st.text_input("NL Query (optional)", value="")
        nl_result = st.text_input("NL Result (optional)", value="")

    st.markdown("### DETECTED SUBJECTS")
    num_subjects = st.slider("Number of subjects", 1, 10, 3)
    detections = []
    for i in range(num_subjects):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        detections.append({
            "id": i + 1,
            "emotion": c1.selectbox(f"Emotion {i+1}", ["neutral","angry","happy","sad","fear","surprise"], key=f"em_{i}"),
            "gender": c2.selectbox(f"Gender {i+1}", ["Man","Woman"], key=f"gen_{i}"),
            "age": c3.number_input(f"Age {i+1}", 10, 80, 25, key=f"age_{i}"),
            "dwell_seconds": c4.number_input(f"Dwell {i+1}", 0, 600, 60, key=f"dw_{i}"),
            "loitering": c5.checkbox(f"Loiter {i+1}", key=f"lo_{i}"),
        })

    st.markdown("### WEAPON DETECTIONS")
    has_weapon = st.checkbox("Weapon detected in session?")
    weapon_detections = []
    if has_weapon:
        wc1, wc2 = st.columns(2)
        weapon_class = wc1.selectbox("Weapon Class", ["Handgun","Knife","Shotgun","SMG","Automatic Rifle","Sniper","Sword"])
        weapon_conf = wc2.slider("Confidence", 0.3, 1.0, 0.85)
        weapon_detections.append({"class_name": weapon_class, "confidence": weapon_conf})

    st.markdown("---")
    if st.button("GENERATE PDF REPORT", type="primary"):
        data = {
            "session_id": session_id,
            "total_persons": total_persons,
            "duration_seconds": duration,
            "loitering_alerts": loitering_alerts,
            "weapon_detections": weapon_detections,
            "detections": detections,
            "heatmap_img": None,
            "frame_sample": None,
            "nl_query": nl_query,
            "nl_result": nl_result,
        }
        with st.spinner("Generating intelligence report..."):
            path = generate_report(data)

        with open(path, "rb") as f:
            pdf_bytes = f.read()

        st.success("Report generated successfully!")
        st.download_button(
            label="DOWNLOAD PDF REPORT",
            data=pdf_bytes,
            file_name=f"phantomeye_report_{session_id}.pdf",
            mime="application/pdf"
        )




def pro_gate(module_name: str) -> bool:
    """Returns True if user can access. Shows upgrade prompt if locked."""
    tier = st.session_state.get("tier", "free")
    if tier == "pro":
        return True

    st.markdown(f"""
    <div style="text-align:center; padding:4rem 2rem; background:rgba(0,10,20,0.9);
        border:1px solid rgba(255,179,0,0.3); border-radius:16px; margin:2rem 0;">
        <div style="font-size:3rem; margin-bottom:1rem;">🔒</div>
        <div style="font-family:monospace; font-size:1.3rem; font-weight:700;
            letter-spacing:0.2em; color:#ffb300; text-transform:uppercase;
            margin-bottom:0.8rem;">{module_name}</div>
        <div style="font-family:monospace; font-size:0.8rem; color:#7ab3d4;
            margin-bottom:0.5rem; letter-spacing:0.1em;">
            This module requires PhantomEye Pro</div>
        <div style="font-size:0.72rem; color:#3a6080; margin-bottom:2rem; letter-spacing:0.08em;">
            Unlock advanced intelligence capabilities with a Pro subscription</div>
        <div style="background:rgba(0,20,40,0.8); border:1px solid rgba(0,180,255,0.15);
            border-radius:12px; padding:1.5rem 2rem; margin-bottom:2rem; display:inline-block; text-align:left;">
            <div style="color:#ffb300; font-size:1.5rem; font-weight:700; text-align:center; margin-bottom:1rem;">
                $29 / month</div>
            <div style="color:#00b4ff; font-size:0.75rem; margin-bottom:0.4rem;">✓  10,000 API calls/day</div>
            <div style="color:#00b4ff; font-size:0.75rem; margin-bottom:0.4rem;">✓  All 8 intelligence modules</div>
            <div style="color:#00b4ff; font-size:0.75rem; margin-bottom:0.4rem;">✓  PDF report export</div>
            <div style="color:#00b4ff; font-size:0.75rem;">✓  Priority processing</div>
        </div>
        <div style="font-size:0.65rem; color:#3a6080; letter-spacing:0.15em; margin-top:1rem;">
            SESSION: {st.session_state.get("session_id", "N/A")} · Your data is never stored</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.link_button(
            "UPGRADE TO PRO →",
            "mailto:sameersain361@gmail.com?subject=PhantomEye Pro Upgrade&body=Session: " + st.session_state.get("session_id", "N/A"),
            use_container_width=True
        )
        st.caption("You will be contacted within 24 hours with your Pro API key.")
    return False


def render_trust_bar():
    tier = st.session_state.get("tier", "free")
    sid = st.session_state.get("session_id", "PE-XXXXXXXX")
    calls = st.session_state.get("api_calls", 0)
    limit = 100 if tier == "free" else 10000
    used_pct = min(int((calls / limit) * 100), 100)
    tier_color = "#ffb300" if tier == "pro" else "#00b4ff"
    tier_label = "PRO" if tier == "pro" else "FREE"

    st.markdown(f"""
    <div style="
        display: flex; justify-content: space-between; align-items: center;
        background: rgba(0,10,20,0.8); border: 1px solid rgba(0,180,255,0.15);
        border-radius: 8px; padding: 0.6rem 1.2rem; margin-bottom: 1.5rem;
        font-family: IBM Plex Mono, monospace; font-size: 0.7rem;
    ">
        <div style="color: #7ab3d4;">
            <span style="color: #00b4ff;">●</span> SESSION: <span style="color: #e8f4ff;">{sid}</span>
        </div>
        <div style="color: #7ab3d4;">
            <span style="
                background: rgba({255 if tier=='free' else 255},{179 if tier=='pro' else 180},{0 if tier=='pro' else 255},0.15);
                border: 1px solid {tier_color};
                color: {tier_color};
                padding: 0.1rem 0.6rem; border-radius: 4px;
                font-weight: 700; letter-spacing: 0.2em;
            ">{tier_label}</span>
            &nbsp;&nbsp; CALLS: <span style="color: #e8f4ff;">{calls}/{limit}</span>
        </div>
        <div style="color: #00ff88; font-size: 0.65rem;">
            🔒 SESSION-ONLY · ZERO DATA STORED
        </div>
    </div>
    <div style="
        height: 2px; background: rgba(0,180,255,0.1);
        border-radius: 2px; margin-bottom: 1.5rem; overflow: hidden;
    ">
        <div style="
            width: {used_pct}%;
            height: 100%;
            background: linear-gradient(90deg, #00b4ff, #00fff0);
            border-radius: 2px;
            transition: width 0.5s;
        "></div>
    </div>
    """, unsafe_allow_html=True)

    if tier == "free" and calls >= 80:
        st.warning(f"⚡ {limit - calls} free calls remaining today. Upgrade to Pro for 10,000 calls/day.")


def main():
    if "page" not in st.session_state:
        st.session_state.page = "landing"
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = "PE-" + str(uuid.uuid4())[:8].upper()
    if "tier" not in st.session_state:
        st.session_state.tier = "free"
    if "api_calls" not in st.session_state:
        st.session_state.api_calls = 0

    page = st.session_state.page

    if page == "landing":
        landing()
    elif page == "home":
        home()
    elif page == "DETECTION":
        detection_page()
    elif page == "ANALYTICS":
        analytics_page()
    elif page == "OSINT":
        osint_page()
    elif page == "REPORT":
        report_page()
    elif page == "WEAPON":
        weapon_page()
    elif page == "NL QUERY":
        nlquery_page()
    elif page == "EMOTION":
        emotion_page()
    elif page == "INTEL":
        intel_page()


if __name__ == "__main__":
    main()