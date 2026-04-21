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
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Share Tech Mono', monospace; }
.stApp { background: #000000; }

.hero-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 88vh; padding: 2rem 0;
}
.hero-eye {
    font-size: 5rem; margin-bottom: 1rem;
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%,100% { transform: translateY(0px); }
    50%      { transform: translateY(-14px); }
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(2.4rem, 6vw, 4rem);
    font-weight: 900; color: #00ff88;
    letter-spacing: 10px; text-align: center;
    text-shadow: 0 0 40px #00ff88, 0 0 80px #00ff4433;
    animation: glow 3s ease-in-out infinite;
    margin-bottom: 0.5rem;
}
@keyframes glow {
    0%,100% { text-shadow: 0 0 30px #00ff88, 0 0 60px #00ff4422; }
    50%      { text-shadow: 0 0 60px #00ff88, 0 0 120px #00ff4455, 0 0 200px #00ff4411; }
}
.hero-sub {
    font-size: 0.8rem; color: #00aa55;
    letter-spacing: 4px; text-align: center;
    margin-bottom: 0.4rem;
}
.hero-status {
    font-size: 0.65rem; color: #003322;
    letter-spacing: 3px; text-align: center;
    margin-bottom: 3rem;
    animation: blink 2s infinite;
}
@keyframes blink {
    0%,100% { opacity: 1; } 50% { opacity: 0.3; }
}
.module-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px; width: 100%; max-width: 800px;
    margin-bottom: 2rem;
}
.mod-card {
    background: #050f05;
    border: 1px solid #00ff8822;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    cursor: pointer;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.mod-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 1px;
    background: linear-gradient(90deg, transparent, #00ff88, transparent);
    transform: translateX(-100%);
    transition: transform 0.4s ease;
}
.mod-card:hover::before { transform: translateX(0); }
.mod-card:hover {
    border-color: #00ff8866;
    box-shadow: 0 0 30px #00ff8811;
    transform: translateY(-3px);
}
.mod-icon { font-size: 1.6rem; margin-bottom: 0.6rem; }
.mod-name {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem; font-weight: 700;
    color: #00ff88; letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 0.4rem;
}
.mod-desc { font-size: 0.7rem; color: #005522; line-height: 1.6; }
.scan-line {
    width: 100%; max-width: 800px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ff8844, transparent);
    margin: 0.5rem 0 1.5rem;
    animation: scan 3s linear infinite;
}
@keyframes scan { 0% { opacity: 0.3; } 50% { opacity: 1; } 100% { opacity: 0.3; } }
.hero-btn {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem; letter-spacing: 3px;
    color: #00ff88; background: transparent;
    border: 1px solid #00ff8844;
    border-radius: 4px;
    padding: 0.6rem 2rem;
    cursor: pointer;
    transition: all 0.2s;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.hero-btn:hover {
    background: #00ff8811;
    border-color: #00ff88;
    box-shadow: 0 0 20px #00ff8833;
}
.app-header {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem; font-weight: 900;
    color: #00ff88; letter-spacing: 6px;
    text-align: center; padding: 1rem 0 0.2rem;
    text-shadow: 0 0 20px #00ff8866;
}
.app-sub {
    font-size: 0.65rem; color: #005522;
    letter-spacing: 2px; text-align: center;
    margin-bottom: 1.5rem;
}
.section-hdr {
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem; color: #00ff88;
    letter-spacing: 4px; text-transform: uppercase;
    border-bottom: 1px solid #00ff8822;
    padding-bottom: 0.5rem; margin: 1rem 0;
}
.terminal {
    background: #030f03;
    border: 1px solid #00ff8822;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.75rem; color: #00aa55;
    margin: 0.4rem 0; line-height: 1.6;
}
.terminal::before { content: '> '; color: #00ff88; }
.back-btn-wrap { margin-bottom: 1.2rem; }
[data-testid="metric-container"] {
    background: #050f05 !important;
    border: 1px solid #00ff8822 !important;
    border-radius: 8px !important;
    padding: 0.8rem !important;
}
[data-testid="metric-container"] label {
    color: #005522 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important; letter-spacing: 2px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00ff88 !important;
    font-family: 'Orbitron', monospace !important;
}
.stButton button {
    background: transparent !important;
    border: 1px solid #00ff8844 !important;
    color: #00ff88 !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    border-radius: 4px !important; transition: all 0.2s !important;
    width: 100% !important;
}
.stButton button:hover {
    background: #00ff8811 !important;
    border-color: #00ff88 !important;
    box-shadow: 0 0 16px #00ff8833 !important;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #003322, #00ff88) !important;
}
.stExpander {
    background: #050f05 !important;
    border: 1px solid #00ff8822 !important;
    border-radius: 8px !important;
}
div[data-testid="stSidebar"] { display: none !important; }
hr { border-color: #00ff8811 !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #00ff8833; border-radius: 2px; }
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
          <div class="mod-name">Emotion Intel</div>
          <div class="mod-desc">Upload a face — get a privacy exposure score from 0 to 100 with matched identities.</div>
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


def main():
    if "page" not in st.session_state:
        st.session_state.page = "landing"

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
    elif page == "EMOTION":
        emotion_page()
    elif page == "INTEL":
        intel_page()


if __name__ == "__main__":
    main()