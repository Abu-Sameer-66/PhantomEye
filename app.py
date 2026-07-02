# import cv2
# import sys
# import time
# import uuid
# import numpy as np
# import streamlit as st
# from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent))

# from core.detection import PersonDetector
# from core.tracker import ByteTracker
# from core.analytics import BehavioralAnalyzer
# from core.osint import OSINTAudit

# st.set_page_config(
#     page_title="PhantomEye — AI Surveillance Intelligence",
#     page_icon="👁",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500;600&family=Exo+2:wght@100;200;300;400;700;900&display=swap');

# :root {
#     --bg-primary: #020408;
#     --bg-card: rgba(6, 18, 32, 0.88);
#     --accent-blue: #00b4ff;
#     --accent-cyan: #00fff0;
#     --accent-red: #ff3355;
#     --accent-green: #00ff88;
#     --accent-gold: #f0b429;
#     --border-glow: rgba(0, 180, 255, 0.4);
#     --border-subtle: rgba(0, 180, 255, 0.1);
#     --text-primary: #e8f4ff;
#     --text-secondary: #7ab3d4;
#     --text-dim: #3a6080;
#     --grid-color: rgba(0, 180, 255, 0.03);
#     --shadow-blue: 0 0 60px rgba(0, 180, 255, 0.2);
#     --shadow-card: 0 12px 40px rgba(0, 0, 0, 0.8);
# }

# *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

# html, body, [class*="css"] {
#     font-family: 'IBM Plex Mono', monospace;
#     background: var(--bg-primary) !important;
#     color: var(--text-primary) !important;
# }

# /* TOP ACCENT LINE */
# .stApp::after {
#     content: ''; position: fixed; top: 0; left: 0; right: 0; height: 2px;
#     background: linear-gradient(90deg,
#         transparent 0%, var(--accent-blue) 15%,
#         var(--accent-cyan) 50%, var(--accent-blue) 85%, transparent 100%);
#     z-index: 9999; animation: topbar 4s ease-in-out infinite alternate;
# }
# @keyframes topbar { from { opacity: 0.5; } to { opacity: 1; filter: brightness(1.5); } }

# /* BACKGROUND */
# .stApp {
#     background:
#         radial-gradient(ellipse at 10% 30%, rgba(0,80,160,0.15) 0%, transparent 50%),
#         radial-gradient(ellipse at 90% 10%, rgba(0,40,100,0.2) 0%, transparent 45%),
#         radial-gradient(ellipse at 50% 90%, rgba(0,50,120,0.1) 0%, transparent 55%),
#         linear-gradient(180deg, #020408 0%, #030b16 100%) !important;
#     min-height: 100vh;
# }

# /* GRID */
# .stApp::before {
#     content: ''; position: fixed; inset: 0;
#     background-image:
#         linear-gradient(var(--grid-color) 1px, transparent 1px),
#         linear-gradient(90deg, var(--grid-color) 1px, transparent 1px);
#     background-size: 56px 56px;
#     pointer-events: none; z-index: 0;
# }

# /* SESSION BAR */
# .session-bar {
#     display: flex; justify-content: space-between; align-items: center;
#     background: rgba(0,8,18,0.75); border: 1px solid rgba(0,180,255,0.08);
#     border-radius: 6px; padding: 0.5rem 1.4rem; margin-bottom: 2rem;
#     font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
#     backdrop-filter: blur(20px);
#     box-shadow: 0 1px 20px rgba(0,0,0,0.4);
# }
# .session-bar .sid { color: var(--text-dim); letter-spacing: 0.05em; }
# .session-bar .sid span { color: var(--accent-blue); font-weight: 500; }
# .session-bar .status { color: var(--accent-green); letter-spacing: 0.25em; font-size: 0.62rem; }
# .session-bar .status::before { content: '● '; animation: blink 1.5s infinite; }
# .session-bar .badge {
#     font-family: 'Rajdhani', sans-serif; font-size: 0.58rem; font-weight: 700;
#     letter-spacing: 0.3em; text-transform: uppercase; color: var(--accent-cyan);
#     background: rgba(0,255,240,0.06); border: 1px solid rgba(0,255,240,0.25);
#     border-radius: 3px; padding: 0.15rem 0.7rem;
# }

# /* HERO */
# .hero-wrap {
#     display: flex; flex-direction: column; align-items: center; justify-content: center;
#     min-height: 92vh; padding: 3rem 1rem; position: relative; text-align: center;
# }
# .hero-wrap::before {
#     content: ''; position: absolute; width: 800px; height: 800px;
#     background: radial-gradient(circle, rgba(0,180,255,0.06) 0%, transparent 68%);
#     border-radius: 50%; top: 50%; left: 50%; transform: translate(-50%,-50%);
#     animation: pulse-bg 6s ease-in-out infinite;
# }
# @keyframes pulse-bg {
#     0%,100% { transform: translate(-50%,-50%) scale(1); opacity: 0.4; }
#     50% { transform: translate(-50%,-50%) scale(1.15); opacity: 0.9; }
# }
# .hero-eye {
#     font-size: 5.5rem; margin-bottom: 1.5rem;
#     animation: float 6s ease-in-out infinite;
#     filter: drop-shadow(0 0 50px rgba(0,180,255,1));
# }
# @keyframes float {
#     0%,100% { transform: translateY(0) rotate(-2deg); }
#     50% { transform: translateY(-24px) rotate(2deg); }
# }
# .hero-title {
#     font-family: 'Exo 2', sans-serif;
#     font-size: clamp(3.5rem, 8.5vw, 8rem); font-weight: 900; letter-spacing: 0.1em;
#     background: linear-gradient(140deg, #ffffff 0%, #60c8ff 35%, var(--accent-cyan) 100%);
#     -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
#     margin-bottom: 0.6rem; line-height: 0.9;
#     animation: reveal 0.8s ease-out both;
# }
# @keyframes reveal { from { opacity: 0; transform: translateY(32px); } to { opacity: 1; transform: translateY(0); } }
# .hero-sub {
#     font-family: 'Rajdhani', sans-serif; font-size: clamp(0.78rem, 1.8vw, 1rem);
#     font-weight: 300; letter-spacing: 0.5em; color: var(--text-dim);
#     margin-bottom: 0.6rem; text-transform: uppercase;
# }
# .hero-status {
#     font-family: 'IBM Plex Mono', monospace;
#     font-size: 0.65rem; color: var(--accent-green);
#     letter-spacing: 0.28em; margin-bottom: 2.5rem; opacity: 0.9;
# }
# .hero-status::before { content: '● '; animation: blink 1.5s infinite; }
# @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.1; } }

# /* STATS */
# .stats-row { display: flex; gap: 1.5rem; margin-bottom: 2.5rem; justify-content: center; flex-wrap: wrap; }
# .stat-item {
#     text-align: center; background: rgba(6,18,32,0.7);
#     border: 1px solid rgba(0,180,255,0.12); border-radius: 10px;
#     padding: 1rem 2rem; backdrop-filter: blur(20px); min-width: 105px;
#     transition: border-color 0.3s, box-shadow 0.3s;
# }
# .stat-item:hover { border-color: rgba(0,180,255,0.3); box-shadow: 0 0 20px rgba(0,180,255,0.1); }
# .stat-value { font-family: 'Exo 2', sans-serif; font-size: 1.7rem; font-weight: 900; color: var(--accent-blue); display: block; }
# .stat-label { font-size: 0.58rem; letter-spacing: 0.28em; color: var(--text-dim); text-transform: uppercase; margin-top: 0.3rem; display: block; }

# /* MODULE GRID */
# .module-grid {
#     display: grid; grid-template-columns: repeat(auto-fit, minmax(255px, 1fr));
#     gap: 1.25rem; width: 100%; max-width: 1240px; margin: 0 auto 3rem;
# }
# .mod-card {
#     background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 14px;
#     padding: 1.8rem 1.6rem; position: relative; overflow: hidden;
#     transition: all 0.38s cubic-bezier(0.23,1,0.32,1); backdrop-filter: blur(24px);
# }
# .mod-card::before {
#     content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1.5px;
#     background: linear-gradient(90deg, transparent 0%, var(--accent-blue) 30%, var(--accent-cyan) 70%, transparent 100%);
#     opacity: 0; transition: opacity 0.35s;
# }
# .mod-card::after {
#     content: ''; position: absolute; inset: 0;
#     background: radial-gradient(ellipse at 0% 0%, rgba(0,180,255,0.08) 0%, transparent 60%);
#     opacity: 0; transition: opacity 0.38s;
# }
# .mod-card:hover { border-color: rgba(0,180,255,0.32); transform: translateY(-6px) scale(1.005); box-shadow: var(--shadow-blue), var(--shadow-card); }
# .mod-card:hover::before { opacity: 1; }
# .mod-card:hover::after  { opacity: 1; }
# .mod-card.research-card { border-color: rgba(255,51,85,0.15); }
# .mod-card.research-card::before { background: linear-gradient(90deg, transparent, var(--accent-red), #ff8800, transparent); }
# .mod-card.research-card::after  { background: radial-gradient(ellipse at 0% 0%, rgba(255,51,85,0.07) 0%, transparent 60%); }
# .mod-card.research-card:hover { border-color: rgba(255,51,85,0.45); box-shadow: 0 0 50px rgba(255,51,85,0.1), var(--shadow-card); }

# .mod-icon { font-size: 1.9rem; margin-bottom: 0.9rem; display: block; line-height: 1; }
# .mod-name {
#     font-family: 'Rajdhani', sans-serif; font-size: 0.9rem; font-weight: 700;
#     letter-spacing: 0.22em; color: var(--accent-blue); text-transform: uppercase; margin-bottom: 0.45rem;
# }
# .mod-name.red { color: var(--accent-red); }
# .mod-tag {
#     display: inline-block; font-size: 0.55rem; letter-spacing: 0.15em;
#     color: var(--accent-cyan); background: rgba(0,255,240,0.06);
#     border: 1px solid rgba(0,255,240,0.18); border-radius: 3px;
#     padding: 0.12rem 0.55rem; margin-bottom: 0.65rem; text-transform: uppercase;
# }
# .mod-tag.red { color: var(--accent-red); background: rgba(255,51,85,0.06); border-color: rgba(255,51,85,0.22); }
# .mod-desc { font-size: 0.74rem; color: var(--text-secondary); line-height: 1.72; }
# .mod-meta {
#     font-size: 0.6rem; color: var(--text-dim); margin-top: 0.85rem;
#     border-top: 1px solid rgba(0,180,255,0.07); padding-top: 0.65rem;
#     letter-spacing: 0.03em; line-height: 1.5;
# }

# /* SCAN LINE */
# .scan-line {
#     width: 100%; max-width: 860px; height: 1px;
#     background: linear-gradient(90deg, transparent, rgba(0,180,255,0.3), rgba(0,255,240,0.5), rgba(0,180,255,0.3), transparent);
#     margin: 2rem auto; position: relative; overflow: hidden;
# }
# .scan-line::after {
#     content: ''; position: absolute; width: 90px; height: 100%;
#     background: linear-gradient(90deg, transparent, rgba(0,255,240,1), transparent);
#     animation: scan 3.5s linear infinite;
# }
# @keyframes scan { from { left: -90px; } to { left: 100%; } }

# /* APP HEADER */
# .app-header {
#     font-family: 'Exo 2', sans-serif; font-size: 1.6rem; font-weight: 800;
#     letter-spacing: 0.35em;
#     background: linear-gradient(135deg, #ffffff 0%, #70d4ff 60%, var(--accent-cyan) 100%);
#     -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
#     text-align: center; padding: 1.5rem 0 0.35rem;
# }
# .app-sub {
#     font-family: 'Rajdhani', sans-serif; font-size: 0.68rem; color: var(--text-dim);
#     letter-spacing: 0.5em; text-align: center; margin-bottom: 1.5rem; text-transform: uppercase;
# }

# /* NAV DIVIDER */
# .nav-divider {
#     display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;
# }
# .nav-divider-line { flex: 1; height: 1px; background: var(--border-subtle); }
# .nav-divider-label {
#     font-family: 'IBM Plex Mono', monospace; font-size: 0.55rem; color: var(--text-dim);
#     letter-spacing: 0.25em; text-transform: uppercase; white-space: nowrap;
# }

# /* BUTTONS */
# .stButton > button {
#     font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
#     letter-spacing: 0.1em !important; font-size: 0.78rem !important;
#     background: rgba(6,18,32,0.9) !important; color: var(--accent-blue) !important;
#     border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 7px !important;
#     padding: 0.65rem 0.8rem !important; transition: all 0.25s ease !important;
#     text-transform: uppercase !important; width: 100% !important;
#     white-space: nowrap !important;
# }
# .stButton > button:hover {
#     background: rgba(0,180,255,0.1) !important; border-color: rgba(0,180,255,0.4) !important;
#     color: var(--accent-cyan) !important;
#     box-shadow: 0 0 20px rgba(0,180,255,0.2), inset 0 0 15px rgba(0,180,255,0.05) !important;
#     transform: translateY(-2px) !important;
# }
# .stButton > button[kind="primary"] {
#     background: linear-gradient(135deg, rgba(0,90,180,0.5), rgba(0,180,255,0.25)) !important;
#     border-color: var(--accent-blue) !important; color: #fff !important;
#     box-shadow: 0 0 30px rgba(0,180,255,0.3) !important;
# }
# .stButton > button[kind="primary"]:hover {
#     box-shadow: 0 0 40px rgba(0,180,255,0.5) !important;
# }

# /* SECTION HEADERS */
# .section-hdr {
#     font-family: 'Exo 2', sans-serif; font-size: 1.2rem; font-weight: 700;
#     letter-spacing: 0.28em; color: var(--accent-blue); text-transform: uppercase;
#     padding: 0.5rem 0; border-bottom: 1px solid rgba(0,180,255,0.1);
#     margin-bottom: 0.5rem; position: relative;
# }
# .section-hdr::after {
#     content: ''; position: absolute; bottom: -1px; left: 0; width: 70px; height: 1.5px;
#     background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
# }
# .section-hdr.red { color: var(--accent-red); }
# .section-hdr.red::after { background: linear-gradient(90deg, var(--accent-red), #ff8800); }
# .section-sub { font-size: 0.7rem; color: var(--text-secondary); letter-spacing: 0.18em; margin-bottom: 1.8rem; text-transform: uppercase; }

# /* TERMINAL */
# .terminal {
#     background: rgba(0,6,16,0.95); border: 1px solid rgba(0,180,255,0.1);
#     border-left: 2px solid var(--accent-blue); border-radius: 0 5px 5px 0;
#     padding: 0.75rem 1.2rem; font-size: 0.7rem; color: var(--accent-green);
#     letter-spacing: 0.12em; margin-top: 1.5rem; position: relative; overflow: hidden;
#     box-shadow: inset 0 0 30px rgba(0,0,0,0.5);
# }
# .terminal::before {
#     content: ''; position: absolute; inset: 0;
#     background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,136,0.008) 2px, rgba(0,255,136,0.008) 4px);
#     pointer-events: none;
# }

# /* INFO BOX */
# .info-box {
#     background: rgba(0,8,20,0.8); border: 1px solid rgba(0,180,255,0.09);
#     border-radius: 8px; padding: 1.1rem 1.4rem; margin-bottom: 1.5rem;
#     font-size: 0.74rem; color: var(--text-secondary); line-height: 1.85;
#     box-shadow: inset 0 1px 0 rgba(0,180,255,0.05);
# }
# .info-box strong { color: var(--accent-blue); font-weight: 500; }

# /* STREAMLIT WIDGET OVERRIDES */
# .stFileUploader { background: var(--bg-card) !important; border: 1px dashed rgba(0,180,255,0.25) !important; border-radius: 10px !important; padding: 1rem !important; }
# .stTextInput > div > div { background: rgba(4,12,24,0.9) !important; border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 7px !important; color: var(--text-primary) !important; font-family: 'IBM Plex Mono', monospace !important; }
# .stTextInput > div > div:focus-within { border-color: rgba(0,180,255,0.4) !important; box-shadow: 0 0 12px rgba(0,180,255,0.12) !important; }
# .stSelectbox > div > div { background: rgba(4,12,24,0.9) !important; border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 7px !important; color: var(--text-primary) !important; }
# .stNumberInput > div > div { background: rgba(4,12,24,0.9) !important; border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 7px !important; }
# .stSlider > div > div > div { background: var(--accent-blue) !important; }

# div[data-testid="metric-container"] {
#     background: rgba(4,14,28,0.85) !important; border: 1px solid rgba(0,180,255,0.1) !important;
#     border-radius: 10px !important; padding: 1rem !important; transition: all 0.25s;
# }
# div[data-testid="metric-container"]:hover { border-color: rgba(0,180,255,0.28) !important; box-shadow: 0 0 16px rgba(0,180,255,0.08) !important; }
# div[data-testid="metric-container"] label { color: var(--text-dim) !important; font-size: 0.62rem !important; letter-spacing: 0.22em !important; font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important; }
# div[data-testid="metric-container"] div[data-testid="metric-value"] { color: var(--accent-blue) !important; font-family: 'Exo 2', sans-serif !important; font-weight: 800 !important; }

# div[data-testid="stDataFrame"] { background: rgba(4,14,28,0.85) !important; border: 1px solid rgba(0,180,255,0.1) !important; border-radius: 10px !important; overflow: hidden !important; }

# .stSuccess { background: rgba(0,255,136,0.06) !important; border: 1px solid rgba(0,255,136,0.25) !important; border-radius: 7px !important; color: var(--accent-green) !important; }
# .stError, .stWarning { background: rgba(255,51,85,0.06) !important; border: 1px solid rgba(255,51,85,0.25) !important; border-radius: 7px !important; }
# .stInfo { background: rgba(0,180,255,0.06) !important; border: 1px solid rgba(0,180,255,0.18) !important; border-radius: 7px !important; color: var(--accent-blue) !important; }

# hr { border-color: rgba(0,180,255,0.08) !important; margin: 1.5rem 0 !important; }
# ::-webkit-scrollbar { width: 3px; }
# ::-webkit-scrollbar-track { background: var(--bg-primary); }
# ::-webkit-scrollbar-thumb { background: rgba(0,180,255,0.4); border-radius: 2px; }
# .stSpinner > div { border-color: var(--accent-blue) transparent transparent transparent !important; }
# section[data-testid="stSidebar"] { display: none !important; }
# #MainMenu { visibility: hidden; }
# footer { visibility: hidden; }
# header { visibility: hidden; }

# @keyframes fadeInUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
# .stMarkdown, .stButton, .stFileUploader { animation: fadeInUp 0.38s ease-out both; }
# </style>
# """, unsafe_allow_html=True)


# @st.cache_resource
# def load_detector():
#     return PersonDetector()

# @st.cache_resource
# def load_osint():
#     return OSINTAudit()

# @st.cache_resource
# def load_emotion_model():
#     from core.emotion import process_frame_emotion
#     return process_frame_emotion

# @st.cache_resource
# def load_weapon_model_cached():
#     from core.weapon import load_weapon_model
#     return load_weapon_model()


# def render_session_bar():
#     sid = st.session_state.get("session_id", "PE-XXXXXXXX")
#     st.markdown(f"""
#     <div class="session-bar">
#         <div class="sid"><span>●</span>&nbsp;&nbsp;SESSION: <span>{sid}</span></div>
#         <div class="status">ALL SYSTEMS ONLINE</div>
#         <div class="badge">OPEN ACCESS</div>
#     </div>
#     """, unsafe_allow_html=True)


# def back_button():
#     if st.button("← BACK TO MODULES"):
#         st.session_state.page = "home"
#         st.rerun()


# def landing():
#     st.markdown("""
#     <div class="hero-wrap">
#       <div class="hero-eye">👁</div>
#       <div class="hero-title">PHANTOMEYE</div>
#       <div class="hero-sub">AI-Powered Surveillance Intelligence System</div>
#       <div class="hero-status">[ SYSTEM ONLINE ] · OPEN ACCESS · BUILD v3.3</div>

#       <div class="stats-row">
#         <div class="stat-item"><span class="stat-value">12</span><span class="stat-label">Modules</span></div>
#         <div class="stat-item"><span class="stat-value">4</span><span class="stat-label">Novel Algorithms</span></div>
#         <div class="stat-item"><span class="stat-value">9</span><span class="stat-label">Weapon Classes</span></div>
#         <div class="stat-item"><span class="stat-value">CPU</span><span class="stat-label">No GPU Required</span></div>
#       </div>

#       <div class="scan-line"></div>

#       <div class="module-grid">
#         <div class="mod-card">
#           <div class="mod-icon">🎯</div>
#           <div class="mod-name">Person Detection</div>
#           <div class="mod-tag">YOLOv8-nano</div>
#           <div class="mod-desc">Real-time person detection on any uploaded image. Returns bounding boxes and per-person confidence scores. Runs entirely on CPU — no GPU required.</div>
#           <div class="mod-meta">Model: yolov8n.pt · Class 0 only · Confidence: 0.4 · CPU optimized</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">🔥</div>
#           <div class="mod-name">Behavioral Analytics</div>
#           <div class="mod-tag">ByteTrack · OpenCV</div>
#           <div class="mod-desc">Persistent person IDs across frames, live behavioral heatmap showing movement density, per-person dwell times, and automated loitering alerts from any video.</div>
#           <div class="mod-meta">Tracker: ByteTrack IOU · Heatmap: NumPy · Alert threshold: 60s · Max: 15s</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">🕵️</div>
#           <div class="mod-name">OSINT Audit</div>
#           <div class="mod-tag">LBPH Face Recognition</div>
#           <div class="mod-desc">Upload a face and receive a Privacy Exposure Score from 0 to 100. LBPH embeddings matched against a reference gallery. Risk classified as LOW, MEDIUM, or HIGH.</div>
#           <div class="mod-meta">Engine: OpenCV LBPH · Similarity: cosine · Score: 0–100 · No data stored</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">🧠</div>
#           <div class="mod-name">Emotion Intelligence</div>
#           <div class="mod-tag">DeepFace · TensorFlow</div>
#           <div class="mod-desc">Multi-face emotion analysis. Returns dominant emotion, estimated age, and gender per face. False-positive filter rejects faces smaller than 15% of frame area.</div>
#           <div class="mod-meta">Backend: DeepFace · Detector: OpenCV · Min face: 15% · 7 emotion classes</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">💬</div>
#           <div class="mod-name">NL Query Engine</div>
#           <div class="mod-tag">Groq LLaMA 3</div>
#           <div class="mod-desc">Type a surveillance query in plain English or Roman Urdu. LLaMA 3 extracts structured filters — emotion, gender, age, dwell time, loitering — then matches against records.</div>
#           <div class="mod-meta">Model: llama-3.1-8b-instant · English + Roman Urdu · Output: JSON filters</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">⚠️</div>
#           <div class="mod-name">Weapon Detection</div>
#           <div class="mod-tag">YOLOv8 Custom · 9 Classes</div>
#           <div class="mod-desc">Custom YOLOv8 trained on 714 real weapon images. Handgun 89.5%, Shotgun 96.3%, SMG 98.6% average precision. Immediate threat alert fires on any detection.</div>
#           <div class="mod-meta">Classes: Handgun · Knife · Shotgun · Sniper · AR · SMG · Sword · Bazooka · GL</div>
#         </div>
#         <div class="mod-card research-card">
#           <div class="mod-icon">📊</div>
#           <div class="mod-name red">Threat Momentum Score</div>
#           <div class="mod-tag red">Novel Algorithm · TMS v1.0</div>
#           <div class="mod-desc">Original research. Accumulates threat signals over time using a compound interest model — loitering, stress emotion, rapid movement, restricted zone, gaze anomaly, group formation.</div>
#           <div class="mod-meta">6 signals · Decay: 45s half-life · Amplifier: score/200 · 5 threat levels</div>
#         </div>
#         <div class="mod-card research-card">
#           <div class="mod-icon">🧬</div>
#           <div class="mod-name red">Behavioral DNA</div>
#           <div class="mod-tag red">Novel Algorithm · BDF v1.0</div>
#           <div class="mod-desc">Camera-agnostic person re-identification using behavioral signature alone. Identifies the same person across cameras without face recognition — works through masks, hats, distance.</div>
#           <div class="mod-meta">5 components: gait · velocity · spatial · social distance · dwell zones · Threshold: 82%</div>
#         </div>
#         <div class="mod-card research-card">
#           <div class="mod-icon">🕸️</div>
#           <div class="mod-name red">Social Graph</div>
#           <div class="mod-tag red">Novel Algorithm · SGI v1.0</div>
#           <div class="mod-desc">Detects who is associated with whom from movement correlation alone — no prior information needed. Three people entering separately but coordinating get flagged before any overt action.</div>
#           <div class="mod-meta">Proximity · velocity sync · dwell overlap · BFS connected-component group detection</div>
#         </div>
#         <div class="mod-card research-card">
#           <div class="mod-icon">🚀</div>
#           <div class="mod-name red">Predictive Exit Vector</div>
#           <div class="mod-tag red">Novel Algorithm · PEV v1.0</div>
#           <div class="mod-desc">Predicts which frame boundary a person will cross and how many seconds remain — 3 to 5 seconds before actual exit. Velocity smoothing plus linear trajectory extrapolation. Designed for camera handoff in multi-camera surveillance grids.</div>
#           <div class="mod-meta">Trajectory extrapolation · velocity smoothing · boundary proximity · confidence scoring · no open-source equivalent</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">📄</div>
#           <div class="mod-name">Intel Report</div>
#           <div class="mod-tag">fpdf2 · PDF Export</div>
#           <div class="mod-desc">Generate a classified PDF intelligence report from any session. Session overview, weapon threat log in red, per-subject behavioral records, and NL query history.</div>
#           <div class="mod-meta">fpdf2 · Dark bg + green text · CLASSIFIED header · Threat sections in red</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">⚡</div>
#           <div class="mod-name">System Intel</div>
#           <div class="mod-tag">Live Status</div>
#           <div class="mod-desc">Live system dashboard with all active modules, tech stack, benchmark results, API endpoint reference, and full deployment metadata for complete transparency.</div>
#           <div class="mod-meta">v3.3.0 · HuggingFace Spaces · FastAPI OAS 3.1 · GitHub open source</div>
#         </div>
#       </div>
#     </div>
#     """, unsafe_allow_html=True)

#     cols = st.columns([1, 2, 1])
#     with cols[1]:
#         if st.button("INITIALIZE SYSTEM  →", key="enter_btn"):
#             st.session_state.page = "home"
#             st.rerun()


# def home():
#     render_session_bar()
#     st.markdown('<div class="app-header">👁 PHANTOMEYE</div>', unsafe_allow_html=True)
#     st.markdown('<div class="app-sub">SELECT INTELLIGENCE MODULE · ALL SYSTEMS ACTIVE</div>', unsafe_allow_html=True)

#     # Row 1 — Core modules
#     st.markdown("""
#     <div class="nav-divider">
#         <div class="nav-divider-line"></div>
#         <div class="nav-divider-label">Core Intelligence</div>
#         <div class="nav-divider-line"></div>
#     </div>
#     """, unsafe_allow_html=True)

#     row1 = [
#         ("DETECTION", "Detection"),
#         ("ANALYTICS", "Analytics"),
#         ("OSINT",     "OSINT"),
#         ("EMOTION",   "Emotion"),
#         ("NL QUERY",  "NL Query"),
#         ("WEAPON",    "Weapon"),
#     ]
#     cols1 = st.columns(6)
#     for i, (key, label) in enumerate(row1):
#         with cols1[i]:
#             if st.button(label, key=f"mod_{key}"):
#                 st.session_state.page = key
#                 st.rerun()

#     # Row 2 — Research + utility
#     st.markdown("""
#     <div class="nav-divider" style="margin-top:0.75rem;">
#         <div class="nav-divider-line"></div>
#         <div class="nav-divider-label">Novel Research · Utility</div>
#         <div class="nav-divider-line"></div>
#     </div>
#     """, unsafe_allow_html=True)

#     row2 = [
#         ("THREAT", "Threat Score"),
#         ("BDF",    "Behavioral DNA"),
#         ("SGI",    "Social Graph"),
#         ("PEV",    "Predictive Exit"),
#         ("REPORT", "Report"),
#         ("INTEL",  "System"),
#     ]
#     cols2 = st.columns(6)
#     for i, (key, label) in enumerate(row2):
#         with cols2[i]:
#             if st.button(label, key=f"mod2_{key}"):
#                 st.session_state.page = key
#                 st.rerun()

#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown(
#         '<div class="terminal">[ PHANTOMEYE v3.3 ] · YOLOv8 loaded · ByteTrack active · '
#         'DeepFace online · Groq LLaMA connected · Weapon model ready · '
#         'TMS v1.0 active · BDF v1.0 active · SGI v1.0 active · PEV v1.0 active · All 12 modules ONLINE</div>',
#         unsafe_allow_html=True
#     )


# def detection_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">Person Detection</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">YOLOv8-nano · CPU inference · class 0 persons only · confidence threshold 0.4</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> Upload any image and PhantomEye runs YOLOv8-nano inference entirely on CPU. Configured for class 0 (person) detection only at a confidence threshold of 0.4. Each detected person receives a bounding box and confidence score. Expand the detection log below the output image to inspect raw bbox coordinates and confidence per subject. No GPU required at any point.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">yolov8n.pt · device: cpu · class 0 only · confidence threshold: 0.4</div>', unsafe_allow_html=True)

#     uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], key="det_up")
#     if uploaded:
#         data  = np.frombuffer(uploaded.read(), np.uint8)
#         image = cv2.imdecode(data, cv2.IMREAD_COLOR)
#         if image is None:
#             st.error("Cannot decode image.")
#             return
#         with st.spinner("Running inference..."):
#             detector   = load_detector()
#             t0         = time.time()
#             detections = detector.detect(image)
#             elapsed    = round(time.time() - t0, 3)
#             annotated  = detector.draw(image, detections)
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("PERSONS DETECTED", len(detections))
#         c2.metric("INFERENCE TIME",   f"{elapsed}s")
#         c3.metric("MODEL",            "YOLOv8n")
#         c4.metric("DEVICE",           "CPU")
#         st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection output", use_container_width=True)
#         if detections:
#             st.markdown('<div class="section-hdr">Detection Log</div>', unsafe_allow_html=True)
#             st.markdown('<div class="section-sub">Expand each entry to inspect bounding box coordinates and confidence score</div>', unsafe_allow_html=True)
#             for i, d in enumerate(detections):
#                 with st.expander(f"PERSON_{i+1:03d}  ·  CONF: {d['confidence']}"):
#                     st.json({"id": i+1, "bbox": list(d["bbox"]), "confidence": d["confidence"]})


# def analytics_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">Behavioral Analytics</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">ByteTrack · behavioral heatmap · dwell time · loitering alerts</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> Upload a video and PhantomEye processes up to 15 seconds of footage. ByteTrack assigns a persistent ID to each person and maintains it across frames, including through brief occlusion. A NumPy heatmap accumulates every pixel position each person visits — high-activity zones appear red. Dwell time is tracked per ID in seconds. If any person remains in one area beyond the loitering threshold, an alert fires listing their tracked ID.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">ByteTrack IOU matching · heatmap: NumPy accumulation · loitering threshold: 60s · max analysis window: 15s</div>', unsafe_allow_html=True)

#     uploaded = st.file_uploader("", type=["mp4", "avi", "mov"], key="ana_up")
#     if uploaded:
#         tmp = Path("outputs") / f"tmp_{int(time.time())}.mp4"
#         tmp.parent.mkdir(exist_ok=True)
#         with open(tmp, "wb") as f:
#             f.write(uploaded.read())
#         cap   = cv2.VideoCapture(str(tmp))
#         fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
#         w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         cap.release()
#         st.markdown(f'<div class="terminal">{w}x{h} @ {fps}fps · {total} total frames · analysis cap: {min(total, fps*15)} frames</div>', unsafe_allow_html=True)
#         if st.button("RUN BEHAVIORAL ANALYSIS"):
#             detector = load_detector()
#             tracker  = ByteTracker()
#             analyzer = BehavioralAnalyzer(w, h, fps)
#             cap      = cv2.VideoCapture(str(tmp))
#             limit    = min(total, fps * 15)
#             prog     = st.progress(0)
#             stat     = st.empty()
#             for i in range(limit):
#                 ret, frame = cap.read()
#                 if not ret: break
#                 dets   = detector.detect(frame)
#                 active = tracker.update(dets)
#                 analyzer.update(active)
#                 prog.progress(int((i / limit) * 100))
#                 if i % 25 == 0:
#                     stat.markdown(f'<div class="terminal">Processing frame {i}/{limit} · active persons: {len(active)}</div>', unsafe_allow_html=True)
#             cap.release()
#             tmp.unlink(missing_ok=True)
#             prog.progress(100)
#             stat.empty()
#             s = analyzer.summary()
#             st.success("Analysis complete")
#             c1, c2, c3, c4 = st.columns(4)
#             c1.metric("TOTAL PERSONS", s.get("total_persons", 0))
#             c2.metric("AVG DWELL",     f"{s.get('avg_dwell_sec', 0)}s")
#             c3.metric("MAX DWELL",     f"{s.get('max_dwell_sec', 0)}s")
#             c4.metric("LOITER ALERTS", s.get("total_alerts", 0))
#             if s.get("total_alerts", 0) > 0:
#                 st.warning(f"Loitering detected — Subject IDs: {s.get('loiterers', [])}")
#             heat = analyzer.get_heatmap_overlay(np.zeros((h, w, 3), dtype=np.uint8))
#             st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), caption="Behavioral heatmap — red zones indicate highest activity density", use_container_width=True)


# def osint_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">OSINT Privacy Audit</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">LBPH face embedding · gallery match · exposure score 0–100 · risk classification</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> Upload a face photo and PhantomEye extracts an LBPH (Local Binary Pattern Histogram) embedding from the detected face region. This is compared against every person in the reference gallery using cosine similarity. The Privacy Exposure Score (0–100) reflects recognition confidence — higher score means stronger match. Risk level: LOW (score &lt; 40), MEDIUM (40–70), HIGH (&gt; 70). All processing in-session only — nothing stored server-side at any point.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">Engine: OpenCV LBPH · Similarity: cosine distance · Score: 0–100 · Risk: LOW / MEDIUM / HIGH · No data retention</div>', unsafe_allow_html=True)

#     c1, c2 = st.columns([1, 1])
#     with c1:
#         query_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="osint_up")
#     with c2:
#         osint = load_osint()
#         st.metric("GALLERY SIZE", f"{len(osint.gallery)} persons")
#         st.metric("ENGINE",       "LBPH Face Recognition")
#     if query_file and st.button("EXECUTE AUDIT"):
#         data  = np.frombuffer(query_file.read(), np.uint8)
#         image = cv2.imdecode(data, cv2.IMREAD_COLOR)
#         if image is None:
#             st.error("Cannot decode image.")
#             return
#         with st.spinner("Running audit..."):
#             result = osint.audit(image, query_id=Path(query_file.name).stem)
#         c1, c2, c3 = st.columns(3)
#         c1.metric("RISK LEVEL",     result["risk_level"])
#         c2.metric("EXPOSURE SCORE", f"{result['exposure_score']}/100")
#         c3.metric("MATCHES FOUND",  len(result["matches"]))
#         st.markdown(f'<div class="terminal">{result["message"]}</div>', unsafe_allow_html=True)
#         if result["matches"]:
#             st.markdown('<div class="section-hdr">Match Log</div>', unsafe_allow_html=True)
#             for m in result["matches"]:
#                 st.markdown(f'<div class="terminal">MATCH: {m["matched_id"]} · CONF: {m["confidence"]}% · SOURCE: {m["source"]}</div>', unsafe_allow_html=True)
#         vis = osint.visualize(image, result)
#         st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="OSINT visualization output", use_container_width=True)


# def emotion_page():
#     render_session_bar()
#     process_frame_emotion = load_emotion_model()
#     back_button()
#     st.markdown('<div class="section-hdr">Emotion Intelligence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">DeepFace · TensorFlow · dominant emotion · age · gender per face</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> PhantomEye runs DeepFace analysis on every detected face in the uploaded image. Returns dominant emotion from 7 classes (angry, fear, sad, happy, surprise, neutral, disgust), an estimated age, and gender classification. A false-positive filter discards any face region smaller than 15% of the frame area — prevents noise from distant or partially visible faces. Multiple faces in a single image are processed independently.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">DeepFace + TensorFlow · OpenCV face detector · min face size: 15% of frame · 7 emotion classes · multi-subject</div>', unsafe_allow_html=True)

#     uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
#     if uploaded:
#         from PIL import Image
#         img       = Image.open(uploaded).convert("RGB")
#         frame     = np.array(img)
#         frame_bgr = frame[:, :, ::-1].copy()
#         with st.spinner("Analyzing faces..."):
#             annotated, results = process_frame_emotion(frame_bgr)
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(frame, caption="Original", use_container_width=True)
#         with col2:
#             st.image(annotated[:, :, ::-1], caption="Emotion analysis output", use_container_width=True)
#         if results:
#             st.markdown("<hr>")
#             st.markdown('<div class="section-hdr">Detected Subjects</div>', unsafe_allow_html=True)
#             for i, r in enumerate(results):
#                 emotion = r.get("dominant_emotion", "N/A").upper()
#                 age     = int(r.get("age", 0))
#                 gender  = r.get("dominant_gender", r.get("gender", "N/A"))
#                 if isinstance(gender, dict):
#                     gender = max(gender, key=gender.get)
#                 c1, c2, c3 = st.columns(3)
#                 c1.metric(f"SUBJECT {i+1} EMOTION", emotion)
#                 c2.metric("AGE ESTIMATE",            f"{age} yrs")
#                 c3.metric("GENDER",                  gender.upper())
#         else:
#             st.warning("No faces detected in this image.")
#     else:
#         st.info("Upload a face image to begin analysis.")


# def nlquery_page():
#     render_session_bar()
#     from core.nlquery import parse_nl_query, apply_filters
#     back_button()
#     st.markdown('<div class="section-hdr">NL Query Engine</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Groq LLaMA 3 · English + Roman Urdu · structured filter extraction</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> Type any surveillance query in natural language — English or Roman Urdu both work. Groq's LLaMA 3 (llama-3.1-8b-instant) parses the intent and extracts structured filters: emotion type, gender, age range, minimum dwell time, and loitering status. Filters are applied against person records and matching subjects are returned in a filterable table. This is the first open-source surveillance system with multilingual NL query support including Roman Urdu.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">llama-3.1-8b-instant via Groq · JSON structured filter extraction · Roman Urdu supported · apply_filters() on person records</div>', unsafe_allow_html=True)

#     query = st.text_input("Enter your query", placeholder="show me angry men who were loitering  |  log jo loiter kar rahy thy")
#     if query:
#         with st.spinner("Parsing query..."):
#             result = parse_nl_query(query)
#         if result['success']:
#             filters = result['filters']
#             st.success(f"Understood: {filters['summary']}")
#             col1, col2, col3 = st.columns(3)
#             col1.metric("EMOTION",   filters['emotion']  or "ANY")
#             col2.metric("GENDER",    filters['gender']   or "ANY")
#             col3.metric("MAX AGE",   filters['max_age']  or "ANY")
#             col4, col5 = st.columns(2)
#             col4.metric("LOITERING", "YES" if filters['loitering'] else "ANY")
#             col5.metric("MIN DWELL", f"{filters['min_dwell_seconds']}s" if filters['min_dwell_seconds'] else "ANY")
#             st.markdown("<hr>")
#             st.markdown('<div class="section-hdr">Filter Results — Sample Dataset</div>', unsafe_allow_html=True)
#             sample_records = [
#                 {"id": 1, "emotion": "angry",   "gender": "Man",   "age": 28, "dwell_seconds": 45,  "loitering": False},
#                 {"id": 2, "emotion": "neutral",  "gender": "Woman", "age": 22, "dwell_seconds": 180, "loitering": True},
#                 {"id": 3, "emotion": "happy",    "gender": "Man",   "age": 35, "dwell_seconds": 20,  "loitering": False},
#                 {"id": 4, "emotion": "angry",    "gender": "Man",   "age": 41, "dwell_seconds": 200, "loitering": True},
#                 {"id": 5, "emotion": "sad",      "gender": "Woman", "age": 19, "dwell_seconds": 90,  "loitering": False},
#                 {"id": 6, "emotion": "fear",     "gender": "Man",   "age": 26, "dwell_seconds": 310, "loitering": True},
#             ]
#             matched = apply_filters(sample_records, filters)
#             if matched:
#                 st.success(f"{len(matched)} subject(s) matched from {len(sample_records)} records")
#                 import pandas as pd
#                 st.dataframe(pd.DataFrame(matched), use_container_width=True)
#             else:
#                 st.warning("No subjects matched this query.")
#         else:
#             st.error(f"Parse failed: {result['error']}")
#     else:
#         st.info("Type a query above — English or Roman Urdu both work.")


# def weapon_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">Weapon Detection</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">YOLOv8 custom trained · 9 weapon classes · real-time threat alert</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> A custom YOLOv8 model trained from scratch on 714 real-world weapon images across 9 classes — trained on Kaggle T4 GPU. Achieves Handgun 89.5%, Shotgun 96.3%, SMG 98.6% average precision at mAP50 53.2%. Upload any image — detected weapons are highlighted with red bounding boxes and an immediate threat alert fires with the weapon class and confidence score. A clean result confirms the scene is clear.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">weapon_detector.pt · mAP50: 53.2% · Handgun: 89.5% · Shotgun: 96.3% · SMG: 98.6% · 714 training images · 9 classes</div>', unsafe_allow_html=True)

#     uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
#     if uploaded:
#         from PIL import Image
#         from core.weapon import detect_weapons
#         img       = Image.open(uploaded).convert("RGB")
#         frame     = np.array(img)
#         frame_bgr = frame[:, :, ::-1].copy()
#         model     = load_weapon_model_cached()
#         with st.spinner("Scanning for threats..."):
#             annotated, detections = detect_weapons(frame_bgr, model)
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(frame, caption="Original", use_container_width=True)
#         with col2:
#             st.image(annotated[:, :, ::-1], caption="Threat analysis output", use_container_width=True)
#         st.markdown("<hr>")
#         if detections:
#             st.error(f"THREAT DETECTED — {len(detections)} weapon(s) identified")
#             st.markdown('<div class="section-hdr red">Detected Threats</div>', unsafe_allow_html=True)
#             for d in detections:
#                 c1, c2 = st.columns(2)
#                 c1.metric("WEAPON CLASS", d['class_name'])
#                 c2.metric("CONFIDENCE",   f"{d['confidence']:.0%}")
#         else:
#             st.success("No weapons detected — scene clear")
#     else:
#         st.info("Upload an image to begin weapon scan.")


# def threat_page():
#     render_session_bar()
#     back_button()
#     from core.threat_momentum import ThreatMomentumEngine

#     st.markdown('<div class="section-hdr red">Threat Momentum Score</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Novel temporal threat accumulation · compound behavioral signal model · TMS v1.0</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>Research contribution:</strong> Unlike binary threat detection systems that output a single yes/no result, TMS accumulates behavioral signals over time using a compound interest model. Each new signal contributes to the score weighted by importance. When the score is already elevated, new signals contribute proportionally more — the amplifier effect. The score decays with a 45-second half-life when no signals arrive, modeling how real threat situations escalate gradually, not instantaneously.<br><br><strong>6 signals and weights:</strong> loitering (0.28) · stress emotion (0.22) · rapid movement (0.18) · proximity violation (0.15) · gaze anomaly (0.10) · group formation (0.07)</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">TMS v1.0 · decay half-life: 45s · amplifier: 1 + score/200 · 5 levels: CLEAR / LOW / MEDIUM / HIGH / CRITICAL</div>', unsafe_allow_html=True)

#     if "tms_engine" not in st.session_state:
#         st.session_state.tms_engine = ThreatMomentumEngine()
#     engine = st.session_state.tms_engine

#     st.markdown("### Subject Input")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         person_id     = st.number_input("Person ID", min_value=1, value=1)
#         dwell_seconds = st.number_input("Dwell Time (seconds)", min_value=0.0, value=0.0, step=5.0)
#         is_loitering  = st.checkbox("Loitering detected")
#     with c2:
#         emotion       = st.selectbox("Detected Emotion", ["none", "neutral", "angry", "fear", "disgust", "sad", "happy", "surprise"])
#         in_restricted = st.checkbox("In restricted zone")
#         group_anomaly = st.checkbox("Group anomaly detected")
#     with c3:
#         px = st.number_input("Position X (px)", min_value=0, value=320)
#         py = st.number_input("Position Y (px)", min_value=0, value=240)

#     col_a, col_b = st.columns(2)
#     with col_a:
#         if st.button("UPDATE THREAT SCORE", type="primary"):
#             result = engine.update_person(
#                 person_id=person_id, position=(px, py),
#                 emotion=None if emotion == "none" else emotion,
#                 dwell_seconds=dwell_seconds, is_loitering=is_loitering,
#                 in_restricted_zone=in_restricted, group_anomaly=group_anomaly,
#             )
#             st.session_state.last_tms = result
#     with col_b:
#         if st.button("RESET THIS PERSON"):
#             engine.reset_person(person_id)
#             if "last_tms" in st.session_state:
#                 del st.session_state.last_tms
#             st.success(f"Person {person_id} profile cleared.")

#     if "last_tms" in st.session_state:
#         r = st.session_state.last_tms
#         level_colors = {"CLEAR": "#10b981", "LOW": "#3b82f6", "MEDIUM": "#f59e0b", "HIGH": "#ef4444", "CRITICAL": "#ff0033"}
#         color = level_colors.get(r.threat_level, "#ffffff")
#         st.markdown(f"""
#         <div style="text-align:center; padding:2.5rem; margin:1.5rem 0;
#             background:rgba(0,4,12,0.97); border:2px solid {color};
#             border-radius:12px; box-shadow: 0 0 60px {color}18;">
#             <div style="font-family:'IBM Plex Mono',monospace; font-size:0.58rem; color:#2a4060; letter-spacing:0.4em; margin-bottom:0.8rem; text-transform:uppercase;">
#                 Threat Momentum Score · Person {r.person_id}
#             </div>
#             <div style="font-size:5.5rem; font-weight:900; color:{color}; font-family:'Exo 2',sans-serif; line-height:0.9; letter-spacing:-0.02em;">{r.tms_score:.1f}</div>
#             <div style="font-size:1rem; font-weight:700; color:{color}; letter-spacing:0.5em; margin-top:0.7rem; font-family:'Rajdhani',sans-serif;">{r.threat_level}</div>
#             <div style="font-size:0.62rem; color:#2a4060; margin-top:0.8rem; font-family:'IBM Plex Mono',monospace; letter-spacing:0.08em;">
#                 Momentum: {r.momentum:+.2f}/frame &nbsp;&nbsp;|&nbsp;&nbsp; Time in system: {r.time_in_system}s
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
#         if r.alert:
#             st.error(r.alert_message)
#         c1, c2, c3 = st.columns(3)
#         c1.metric("ACTIVE SIGNALS", len(r.active_signals))
#         c2.metric("MOMENTUM",       f"{r.momentum:+.3f}")
#         c3.metric("TIME IN SYSTEM", f"{r.time_in_system}s")
#         if r.signal_breakdown:
#             st.markdown('<div class="section-hdr">Signal Breakdown</div>', unsafe_allow_html=True)
#             import pandas as pd
#             df = pd.DataFrame([{"Signal": k.replace("_", " ").upper(), "Score Contribution": round(v, 3)} for k, v in r.signal_breakdown.items()])
#             st.dataframe(df, use_container_width=True)

#     st.markdown("<hr>")
#     st.markdown('<div class="section-hdr">Session Summary</div>', unsafe_allow_html=True)
#     summary = engine.summary()
#     s1, s2, s3, s4 = st.columns(4)
#     s1.metric("PERSONS TRACKED", summary["total_persons_tracked"])
#     s2.metric("TOTAL ALERTS",    summary["total_alerts"])
#     s3.metric("HIGHEST TMS",     summary["highest_tms"])
#     s4.metric("AVG TMS",         summary["avg_tms"])
#     if summary["level_distribution"]:
#         st.markdown('<div class="terminal">Distribution: ' + ' · '.join(f"{k}: {v}" for k, v in summary["level_distribution"].items()) + '</div>', unsafe_allow_html=True)
#     if st.button("RESET ALL PROFILES"):
#         engine.reset_all()
#         if "last_tms" in st.session_state:
#             del st.session_state.last_tms
#         st.success("All threat profiles cleared.")


# def bdf_page():
#     render_session_bar()
#     back_button()
#     from core.behavioral_dna import BehavioralDNAEngine

#     st.markdown('<div class="section-hdr red">Behavioral DNA Fingerprint</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Camera-agnostic re-identification · no face required · pure movement signature · BDF v1.0</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>Research contribution:</strong> Identifies the same person across cameras using behavioral signature alone — gait rhythm, velocity profile, spatial preference zones, social distance pattern, and dwell locations. Works with masks, hats, and at distances where face recognition completely fails. When a person re-enters the scene with a new tracking ID, BDF matches them to their previous identity using cosine similarity on a 5-component behavioral feature vector. Match threshold: 82%.<br><br><strong>5 behavioral components:</strong> gait signature (stride rhythm histogram) · velocity profile (speed distribution) · spatial preference (normalized grid heatmap) · social distance average · dwell zone signature (stopping locations)</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">BDF v1.0 · 5 behavioral signals · cosine similarity · match threshold: 82% · min observations: 15 frames</div>', unsafe_allow_html=True)

#     if "bdf_engine" not in st.session_state:
#         st.session_state.bdf_engine = BehavioralDNAEngine(640, 480)
#     engine = st.session_state.bdf_engine

#     st.markdown("### Add Observations")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         obs_id   = st.number_input("Person ID", min_value=1, value=1)
#         pos_x    = st.number_input("Position X", min_value=0, max_value=640, value=320)
#     with c2:
#         pos_y    = st.number_input("Position Y", min_value=0, max_value=480, value=240)
#         soc_dist = st.number_input("Nearest person distance (px)", min_value=0.0, value=100.0)
#     with c3:
#         n_obs = st.number_input("Observations to simulate", min_value=1, max_value=100, value=30)

#     if st.button("SIMULATE OBSERVATIONS"):
#         for i in range(int(n_obs)):
#             x = int(pos_x + i * 2 + np.random.randn() * 3)
#             y = int(pos_y + np.sin(i * 0.3) * 15 + np.random.randn() * 2)
#             engine.observe(obs_id, (max(0, x), max(0, y)), soc_dist)
#         st.success(f"Added {n_obs} observations for Person {obs_id}")

#     col_a, col_b = st.columns(2)
#     with col_a:
#         if st.button("REGISTER TO GALLERY", type="primary"):
#             bdf = engine.extract_and_register(obs_id)
#             if bdf:
#                 st.success(f"Person {obs_id} registered — confidence: {bdf.confidence:.2f} | observations: {bdf.observation_count}")
#             else:
#                 st.warning(f"Insufficient data. Need at least 15 observations for Person {obs_id}.")
#     with col_b:
#         if st.button("MATCH AGAINST GALLERY"):
#             result = engine.match_against_gallery(obs_id)
#             st.session_state.last_bdf = result

#     if "last_bdf" in st.session_state:
#         r = st.session_state.last_bdf
#         color = "#00b4ff" if r.is_match else "#10b981"
#         st.markdown(f"""
#         <div style="padding:2rem; margin:1rem 0; background:rgba(0,4,12,0.97);
#             border:2px solid {color}; border-radius:10px; box-shadow: 0 0 40px {color}18;">
#             <div style="font-family:'IBM Plex Mono',monospace; font-size:0.58rem; color:#2a4060; letter-spacing:0.35em; margin-bottom:0.6rem; text-transform:uppercase;">
#                 Behavioral DNA Match · Person {r.query_id}
#             </div>
#             <div style="font-size:2.2rem; font-weight:900; color:{color}; font-family:'Exo 2',sans-serif; letter-spacing:0.05em;">{"MATCH FOUND" if r.is_match else "NO MATCH"}</div>
#             <div style="font-size:0.72rem; color:#5a8090; margin-top:0.8rem; font-family:'IBM Plex Mono',monospace; line-height:1.65;">{r.explanation}</div>
#         </div>
#         """, unsafe_allow_html=True)
#         c1, c2, c3 = st.columns(3)
#         c1.metric("SIMILARITY",  f"{r.similarity:.1%}")
#         c2.metric("MATCHED ID",  str(r.matched_id) if r.matched_id else "None")
#         c3.metric("CONFIDENCE",  f"{r.confidence:.2f}")

#     st.markdown("<hr>")
#     st.markdown('<div class="section-hdr">Gallery & Session</div>', unsafe_allow_html=True)
#     summary = engine.summary()
#     s1, s2, s3, s4 = st.columns(4)
#     s1.metric("TRACKED",   summary["persons_tracked"])
#     s2.metric("BDF READY", summary["bdf_ready"])
#     s3.metric("GALLERY",   summary["gallery_size"])
#     s4.metric("MATCHES",   summary["matches_detected"])
#     if st.button("RESET ALL"):
#         engine.reset_all()
#         if "last_bdf" in st.session_state:
#             del st.session_state.last_bdf
#         st.success("BDF engine reset.")


# def sgi_page():
#     render_session_bar()
#     back_button()
#     from core.social_graph import SocialGraphEngine

#     st.markdown('<div class="section-hdr red">Social Graph Intelligence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Real-time group detection · no prior information · pure behavioral correlation · SGI v1.0</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>Research contribution:</strong> Detects "who is with whom" from surveillance footage without any prior information — no face recognition, no name lists, no pre-registration. Three bank robbers entering a building separately — SGI detects their association before any overt action occurs, purely from movement correlation. Uses three behavioral signals: spatial proximity, velocity synchronization (do they accelerate and decelerate together?), and shared dwell zones. Connected-component BFS then extracts groups from the link graph.<br><br><strong>Link strength formula:</strong> proximity score (0.40) + Pearson velocity correlation (0.35) + dwell zone overlap (0.25)</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">SGI v1.0 · proximity threshold: 150px · Pearson velocity correlation · group detection: BFS connected-component analysis</div>', unsafe_allow_html=True)

#     if "sgi_engine" not in st.session_state:
#         st.session_state.sgi_engine = SocialGraphEngine(proximity_px=150)
#     engine = st.session_state.sgi_engine

#     st.markdown("### Simulate Person Movement")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         obs_id = st.number_input("Person ID", min_value=1, value=1)
#         pos_x  = st.number_input("Start Position X", min_value=0, max_value=1920, value=320)
#     with c2:
#         pos_y  = st.number_input("Start Position Y", min_value=0, max_value=1080, value=240)
#         n_obs  = st.number_input("Frames to simulate", min_value=1, max_value=200, value=50)
#     with c3:
#         move_x = st.number_input("Movement X per frame", min_value=-10, max_value=10, value=2)
#         move_y = st.number_input("Movement Y per frame", min_value=-10, max_value=10, value=0)

#     if st.button("SIMULATE MOVEMENT"):
#         for i in range(int(n_obs)):
#             x = int(pos_x + i * move_x + np.random.randn() * 2)
#             y = int(pos_y + i * move_y + np.random.randn() * 2)
#             engine.observe(obs_id, (max(0, x), max(0, y)))
#         engine._update_links()
#         st.success(f"Simulated {n_obs} frames for Person {obs_id}")

#     if st.button("DETECT GROUPS", type="primary"):
#         st.session_state.sgi_result = {
#             "groups":  engine.detect_groups(),
#             "links":   engine.get_all_links(),
#             "summary": engine.summary(),
#         }

#     if "sgi_result" in st.session_state:
#         res     = st.session_state.sgi_result
#         groups  = res["groups"]
#         links   = res["links"]
#         summary = res["summary"]

#         s1, s2, s3, s4 = st.columns(4)
#         s1.metric("PERSONS TRACKED",  summary["persons_tracked"])
#         s2.metric("ACTIVE LINKS",     summary["active_links"])
#         s3.metric("GROUPS DETECTED",  summary["groups_detected"])
#         s4.metric("TOTAL ALERTS",     summary["total_alerts"])

#         if groups:
#             st.markdown('<div class="section-hdr">Detected Groups</div>', unsafe_allow_html=True)
#             for g in groups:
#                 color = "#ef4444" if g.alert else "#00b4ff"
#                 st.markdown(f"""
#                 <div style="padding:1rem 1.5rem; margin:0.5rem 0; background:rgba(0,4,12,0.92); border:1px solid {color}; border-radius:8px;">
#                     <div style="font-family:'Rajdhani',sans-serif; font-size:0.82rem; font-weight:700; color:{color}; letter-spacing:0.18em; margin-bottom:0.4rem;">
#                         GROUP {g.group_id} · {g.formation.upper()} · Cohesion: {g.cohesion:.3f}
#                     </div>
#                     <div style="font-family:'IBM Plex Mono',monospace; font-size:0.66rem; color:#5a8090; line-height:1.5;">
#                         Members: {g.members}{"  ·  ALERT: " + g.alert_reason if g.alert else ""}
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
#         else:
#             st.info("No groups detected yet. Simulate matching movement patterns for multiple persons, then detect groups.")

#         if links:
#             st.markdown('<div class="section-hdr">Social Link Graph</div>', unsafe_allow_html=True)
#             st.markdown('<div class="section-sub">All pairwise behavioral associations detected between tracked persons</div>', unsafe_allow_html=True)
#             import pandas as pd
#             df = pd.DataFrame([{
#                 "Persons":         f"{l.person_a} -- {l.person_b}",
#                 "Strength":        l.strength,
#                 "Type":            l.link_type,
#                 "Frames Observed": l.frame_count,
#                 "Proximity (px)":  l.evidence.get("proximity_px", 0),
#                 "Velocity Corr":   l.evidence.get("velocity_corr", 0),
#             } for l in links])
#             st.dataframe(df, use_container_width=True)

#     st.markdown("<hr>")
#     if st.button("RESET ENGINE"):
#         engine.reset_all()
#         if "sgi_result" in st.session_state:
#             del st.session_state.sgi_result
#         st.success("Social graph engine reset.")


# def pev_page():
#     render_session_bar()
#     back_button()
#     from core.predictive_exit import PredictiveExitEngine

#     st.markdown('<div class="section-hdr red">Predictive Exit Vector</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Frame boundary exit prediction · 3–5 seconds ahead · camera handoff intelligence · PEV v1.0</div>', unsafe_allow_html=True)
#     st.markdown("""
#     <div class='info-box'>
#         <strong>Research contribution:</strong> PEV v1.0 tracks each person's position history and
#         computes a smoothed velocity vector using a sliding window over recent frames.
#         Linear trajectory extrapolation then determines which frame boundary — LEFT, RIGHT, TOP, or BOTTOM —
#         the person will cross and how many seconds remain before exit.
#         Prediction fires <strong>3 to 5 seconds before actual exit</strong>, enabling downstream camera handoff
#         in multi-camera surveillance grids. Confidence is composed from three factors: velocity stability
#         (is the direction consistent?), boundary proximity (how close are they?), and history depth
#         (how many frames observed?). No equivalent open-source implementation exists for
#         real-time multi-person exit prediction in surveillance systems.
#         <br><br>
#         <strong>Algorithm:</strong> position history → sliding-window velocity smoothing →
#         linear trajectory extrapolation → boundary intersection detection →
#         confidence scoring → ExitPrediction output
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown('<div class="terminal">PEV v1.0 · velocity smoothing window: 5 frames · prediction horizon: 4s · confidence: stability × proximity × depth · IUB AI Research Lab</div>', unsafe_allow_html=True)

#     st.markdown("### Simulate Exit Prediction")
#     st.markdown('<div class="section-sub">Manually feed person positions to test the prediction engine in real time</div>', unsafe_allow_html=True)

#     if "pev_engine" not in st.session_state:
#         st.session_state.pev_engine = PredictiveExitEngine(frame_width=640, frame_height=480, fps=25)
#     engine = st.session_state.pev_engine

#     c1, c2, c3 = st.columns(3)
#     with c1:
#         sim_person_id = st.number_input("Person ID", min_value=1, value=1, key="pev_pid")
#         bbox_x1       = st.number_input("BBox X1", min_value=0, max_value=620, value=400, key="pev_x1")
#     with c2:
#         bbox_y1       = st.number_input("BBox Y1", min_value=0, max_value=460, value=200, key="pev_y1")
#         bbox_w        = st.number_input("BBox Width", min_value=20, max_value=200, value=50, key="pev_w")
#     with c3:
#         bbox_h        = st.number_input("BBox Height", min_value=20, max_value=300, value=100, key="pev_h")
#         n_auto_frames = st.number_input("Auto-simulate frames", min_value=1, max_value=50, value=20, key="pev_nf")

#     col_a, col_b, col_c = st.columns(3)

#     with col_a:
#         if st.button("FEED SINGLE FRAME"):
#             dets = [{"person_id": sim_person_id, "bbox": [bbox_x1, bbox_y1, bbox_x1 + bbox_w, bbox_y1 + bbox_h]}]
#             preds = engine.update(dets)
#             st.session_state.pev_result = preds

#     with col_b:
#         if st.button("AUTO-SIMULATE →RIGHT", type="primary"):
#             preds_last = []
#             for i in range(int(n_auto_frames)):
#                 x1 = bbox_x1 + i * 12
#                 dets = [{"person_id": sim_person_id, "bbox": [x1, bbox_y1, x1 + bbox_w, bbox_y1 + bbox_h]}]
#                 preds_last = engine.update(dets)
#             st.session_state.pev_result = preds_last
#             st.success(f"Simulated {n_auto_frames} frames — person moving RIGHT")

#     with col_c:
#         if st.button("RESET ENGINE"):
#             engine.reset()
#             if "pev_result" in st.session_state:
#                 del st.session_state.pev_result
#             st.success("PEV engine reset.")

#     if "pev_result" in st.session_state:
#         preds = st.session_state.pev_result
#         st.markdown("<hr>")
#         st.markdown('<div class="section-hdr">Live Predictions</div>', unsafe_allow_html=True)

#         if not preds or all(p.exit_side == "NONE" for p in preds):
#             st.info("No exit predicted yet — feed more frames or simulate movement toward a boundary.")
#         else:
#             for pred in preds:
#                 if pred.exit_side == "NONE":
#                     continue

#                 side_colors = {
#                     "LEFT":   "#00b4ff",
#                     "RIGHT":  "#00ff88",
#                     "TOP":    "#f0b429",
#                     "BOTTOM": "#ff3355",
#                 }
#                 color  = side_colors.get(pred.exit_side, "#ffffff")
#                 alert_html = (
#                     '<div style="color:#ff3355;font-weight:700;font-size:0.75rem;'
#                     'letter-spacing:0.2em;margin-top:0.5rem;">⚠ EXIT IMMINENT</div>'
#                     if pred.alert else ""
#                 )

#                 st.markdown(f"""
#                 <div style="padding:1.8rem 2rem; margin:0.75rem 0;
#                     background:rgba(0,4,12,0.97); border:2px solid {color};
#                     border-radius:10px; box-shadow: 0 0 40px {color}18;
#                     font-family:'IBM Plex Mono',monospace;">
#                     <div style="font-size:0.55rem; color:#2a4060; letter-spacing:0.4em;
#                         margin-bottom:0.7rem; text-transform:uppercase;">
#                         Predictive Exit Vector · Person {pred.person_id}
#                     </div>
#                     <div style="display:flex; align-items:center; gap:2rem; flex-wrap:wrap;">
#                         <div>
#                             <div style="font-size:2.5rem; font-weight:900; color:{color};
#                                 font-family:'Exo 2',sans-serif; line-height:1;">
#                                 {pred.exit_side}
#                             </div>
#                             <div style="font-size:0.6rem; color:#3a6080; margin-top:0.2rem;
#                                 letter-spacing:0.15em;">EXIT SIDE</div>
#                         </div>
#                         <div>
#                             <div style="font-size:2.5rem; font-weight:900; color:#fff;
#                                 font-family:'Exo 2',sans-serif; line-height:1;">
#                                 {pred.seconds_to_exit:.2f}s
#                             </div>
#                             <div style="font-size:0.6rem; color:#3a6080; margin-top:0.2rem;
#                                 letter-spacing:0.15em;">TIME TO EXIT</div>
#                         </div>
#                         <div>
#                             <div style="font-size:2.5rem; font-weight:900; color:#00fff0;
#                                 font-family:'Exo 2',sans-serif; line-height:1;">
#                                 {pred.confidence:.3f}
#                             </div>
#                             <div style="font-size:0.6rem; color:#3a6080; margin-top:0.2rem;
#                                 letter-spacing:0.15em;">CONFIDENCE</div>
#                         </div>
#                         <div>
#                             <div style="font-size:1.1rem; font-weight:700; color:#7ab3d4;
#                                 font-family:'Rajdhani',sans-serif; line-height:1.3;">
#                                 ({pred.predicted_exit_point[0]}, {pred.predicted_exit_point[1]})
#                             </div>
#                             <div style="font-size:0.6rem; color:#3a6080; margin-top:0.2rem;
#                                 letter-spacing:0.15em;">EXIT POINT (px)</div>
#                         </div>
#                     </div>
#                     <div style="margin-top:0.85rem; font-size:0.62rem; color:#2a4060;">
#                         Velocity: vx={pred.current_velocity[0]:+.2f} · vy={pred.current_velocity[1]:+.2f} px/frame
#                     </div>
#                     {alert_html}
#                 </div>
#                 """, unsafe_allow_html=True)

#         st.markdown("<hr>")
#         st.markdown('<div class="section-hdr">Engine Status</div>', unsafe_allow_html=True)
#         s1, s2 = st.columns(2)
#         s1.metric("ACTIVE TRACKS", len(engine.tracks))
#         s2.metric("FRAMES PROCESSED", engine.frame_counter)

#         if engine.tracks:
#             st.markdown('<div class="section-sub">Tracked persons — history depth per ID</div>', unsafe_allow_html=True)
#             import pandas as pd
#             rows = [{"Person ID": pid, "History Frames": len(t.positions)} for pid, t in engine.tracks.items()]
#             st.dataframe(pd.DataFrame(rows), use_container_width=True)


# def report_page():
#     render_session_bar()
#     from core.reporter import generate_report
#     back_button()
#     st.markdown('<div class="section-hdr">Intelligence Report</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Classified PDF · session data · threat log · subject records · one-click download</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> Fill in session data — total persons detected, loitering alerts, per-subject behavioral records with emotion and dwell time, and any weapon detections from the session. Click Generate and PhantomEye produces a classified PDF using fpdf2. Dark background with green terminal-style text. Weapon threat sections highlighted in red. CLASSIFIED header on the first page. The file is immediately available for download — nothing is stored server-side at any point.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">fpdf2 · dark theme · CLASSIFIED header · weapon threat sections in red · immediate download · zero server-side storage</div>', unsafe_allow_html=True)

#     st.markdown("### Session Data")
#     col1, col2 = st.columns(2)
#     with col1:
#         session_id       = st.text_input("Session ID",           value=st.session_state.get("session_id", "PE-SESSION-001"))
#         total_persons    = st.number_input("Total Persons",      min_value=0, value=5)
#         duration         = st.number_input("Duration (seconds)", min_value=0, value=300)
#     with col2:
#         loitering_alerts = st.number_input("Loitering Alerts",   min_value=0, value=1)
#         nl_query         = st.text_input("NL Query (optional)",  value="")
#         nl_result        = st.text_input("NL Result (optional)", value="")

#     st.markdown("### Detected Subjects")
#     num_subjects = st.slider("Number of subjects", 1, 10, 3)
#     detections = []
#     for i in range(num_subjects):
#         c1, c2, c3, c4, c5, _ = st.columns(6)
#         detections.append({
#             "id":            i + 1,
#             "emotion":       c1.selectbox(f"Emotion {i+1}", ["neutral","angry","happy","sad","fear","surprise"], key=f"em_{i}"),
#             "gender":        c2.selectbox(f"Gender {i+1}",  ["Man","Woman"], key=f"gen_{i}"),
#             "age":           c3.number_input(f"Age {i+1}",  10, 80, 25, key=f"age_{i}"),
#             "dwell_seconds": c4.number_input(f"Dwell {i+1}", 0, 600, 60, key=f"dw_{i}"),
#             "loitering":     c5.checkbox(f"Loiter {i+1}",   key=f"lo_{i}"),
#         })

#     st.markdown("### Weapon Detections")
#     has_weapon = st.checkbox("Weapon detected in session?")
#     weapon_detections = []
#     if has_weapon:
#         wc1, wc2 = st.columns(2)
#         weapon_class = wc1.selectbox("Weapon Class", ["Handgun","Knife","Shotgun","SMG","Automatic Rifle","Sniper","Sword"])
#         weapon_conf  = wc2.slider("Confidence", 0.3, 1.0, 0.85)
#         weapon_detections.append({"class_name": weapon_class, "confidence": weapon_conf})

#     st.markdown("<hr>")
#     if st.button("GENERATE PDF REPORT", type="primary"):
#         data = {
#             "session_id":        session_id,
#             "total_persons":     total_persons,
#             "duration_seconds":  duration,
#             "loitering_alerts":  loitering_alerts,
#             "weapon_detections": weapon_detections,
#             "detections":        detections,
#             "heatmap_img":       None,
#             "frame_sample":      None,
#             "nl_query":          nl_query,
#             "nl_result":         nl_result,
#         }
#         with st.spinner("Generating classified report..."):
#             path = generate_report(data)
#         with open(path, "rb") as f:
#             pdf_bytes = f.read()
#         st.success("Report generated.")
#         st.download_button(
#             label="Download PDF Report",
#             data=pdf_bytes,
#             file_name=f"phantomeye_report_{session_id}.pdf",
#             mime="application/pdf"
#         )


# def intel_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">System Intelligence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Module registry · model benchmarks · deployment info · novel contributions</div>', unsafe_allow_html=True)

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("SYSTEM",  "PhantomEye")
#     c2.metric("VERSION", "v3.3.0")
#     c3.metric("STATUS",  "ONLINE")
#     c4.metric("MODULES", "12 ACTIVE")

#     st.markdown("<br>", unsafe_allow_html=True)
#     modules_info = [
#         ("DETECTION",         "YOLOv8-nano",   "yolov8n.pt · class 0 · confidence 0.4+ · CPU only"),
#         ("ANALYTICS",         "ByteTrack",     "IOU matching · NumPy heatmap · dwell time tracking · loitering threshold: 60s"),
#         ("OSINT",             "LBPH Face",     "LBPH embedding · cosine gallery search · score 0–100 · LOW/MEDIUM/HIGH risk"),
#         ("EMOTION",           "DeepFace + TF", "7 emotion classes · age + gender · OpenCV detector · 15% min face size filter"),
#         ("NL QUERY",          "Groq LLaMA 3",  "llama-3.1-8b-instant · English + Roman Urdu · JSON structured filter extraction"),
#         ("WEAPON",            "YOLOv8 Custom", "9 classes · mAP50 53.2% · Handgun 89.5% · Shotgun 96.3% · SMG 98.6%"),
#         ("THREAT MOMENTUM",   "TMS v1.0",      "Novel · 6 behavioral signals · compound amplifier · 45s decay · 5 threat levels"),
#         ("BEHAVIORAL DNA",    "BDF v1.0",      "Novel · 5 behavioral components · cosine similarity · 82% match threshold"),
#         ("SOCIAL GRAPH",      "SGI v1.0",      "Novel · proximity + velocity sync + dwell overlap · BFS group detection"),
#         ("PREDICTIVE EXIT",   "PEV v1.0",      "Novel · velocity smoothing · linear trajectory extrapolation · boundary prediction 3–5s ahead · camera handoff"),
#         ("REPORT",            "fpdf2",         "Classified PDF · dark theme · CLASSIFIED header · threat sections in red"),
#         ("API",               "FastAPI",        "OAS 3.1 · CORS enabled · uvicorn · modular route handlers"),
#     ]
#     for name, tech, desc in modules_info:
#         with st.expander(f"{name}  ·  {tech}  ·  ACTIVE"):
#             st.markdown(f'<div class="terminal">{desc}</div>', unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.json({
#         "author":              "Abu-Sameer-66",
#         "github":              "https://github.com/Abu-Sameer-66/PhantomEye",
#         "huggingface":         "https://abu-sameer-66-phantomeye.hf.space",
#         "stack":               ["Python 3.10", "YOLOv8", "DeepFace", "ByteTrack", "FastAPI", "Streamlit", "Groq", "fpdf2"],
#         "novel_contributions": [
#             "Threat Momentum Score (TMS v1.0) — temporal compound threat accumulation",
#             "Behavioral DNA Fingerprint (BDF v1.0) — camera-agnostic behavioral re-ID",
#             "Social Graph Intelligence (SGI v1.0) — implicit group detection from movement",
#             "Predictive Exit Vector (PEV v1.0) — 3-5s ahead frame boundary exit prediction",
#         ],
#         "paper_status":        "in progress",
#         "status":              "online",
#         "access":              "open",
#     })


# def main():
#     if "page"       not in st.session_state:
#         st.session_state.page = "landing"
#     if "session_id" not in st.session_state:
#         st.session_state.session_id = "PE-" + str(uuid.uuid4())[:8].upper()

#     page = st.session_state.page

#     if   page == "landing":   landing()
#     elif page == "home":      home()
#     elif page == "DETECTION": detection_page()
#     elif page == "ANALYTICS": analytics_page()
#     elif page == "OSINT":     osint_page()
#     elif page == "EMOTION":   emotion_page()
#     elif page == "NL QUERY":  nlquery_page()
#     elif page == "WEAPON":    weapon_page()
#     elif page == "THREAT":    threat_page()
#     elif page == "BDF":       bdf_page()
#     elif page == "SGI":       sgi_page()
#     elif page == "PEV":       pev_page()
#     elif page == "REPORT":    report_page()
#     elif page == "INTEL":     intel_page()


# if __name__ == "__main__":
#     main()


# import cv2
# import sys
# import time
# import uuid
# import numpy as np
# import streamlit as st
# from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent))

# from core.detection import PersonDetector
# from core.tracker import ByteTracker
# from core.analytics import BehavioralAnalyzer
# from core.osint import OSINTAudit

# st.set_page_config(
#     page_title="PhantomEye — AI Surveillance Intelligence",
#     page_icon="👁",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500;600&family=Exo+2:wght@100;200;300;400;700;900&display=swap');

# :root {
#     --bg-primary: #020408;
#     --bg-card: rgba(6, 18, 32, 0.88);
#     --accent-blue: #00b4ff;
#     --accent-cyan: #00fff0;
#     --accent-red: #ff3355;
#     --accent-green: #00ff88;
#     --accent-gold: #f0b429;
#     --border-glow: rgba(0, 180, 255, 0.4);
#     --border-subtle: rgba(0, 180, 255, 0.1);
#     --text-primary: #e8f4ff;
#     --text-secondary: #7ab3d4;
#     --text-dim: #3a6080;
#     --grid-color: rgba(0, 180, 255, 0.03);
#     --shadow-blue: 0 0 60px rgba(0, 180, 255, 0.2);
#     --shadow-card: 0 12px 40px rgba(0, 0, 0, 0.8);
# }

# *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

# html, body, [class*="css"] {
#     font-family: 'IBM Plex Mono', monospace;
#     background: var(--bg-primary) !important;
#     color: var(--text-primary) !important;
# }

# /* TOP ACCENT LINE */
# .stApp::after {
#     content: ''; position: fixed; top: 0; left: 0; right: 0; height: 2px;
#     background: linear-gradient(90deg,
#         transparent 0%, var(--accent-blue) 15%,
#         var(--accent-cyan) 50%, var(--accent-blue) 85%, transparent 100%);
#     z-index: 9999; animation: topbar 4s ease-in-out infinite alternate;
# }
# @keyframes topbar { from { opacity: 0.5; } to { opacity: 1; filter: brightness(1.5); } }

# /* BACKGROUND */
# .stApp {
#     background:
#         radial-gradient(ellipse at 10% 30%, rgba(0,80,160,0.15) 0%, transparent 50%),
#         radial-gradient(ellipse at 90% 10%, rgba(0,40,100,0.2) 0%, transparent 45%),
#         radial-gradient(ellipse at 50% 90%, rgba(0,50,120,0.1) 0%, transparent 55%),
#         linear-gradient(180deg, #020408 0%, #030b16 100%) !important;
#     min-height: 100vh;
# }

# /* GRID */
# .stApp::before {
#     content: ''; position: fixed; inset: 0;
#     background-image:
#         linear-gradient(var(--grid-color) 1px, transparent 1px),
#         linear-gradient(90deg, var(--grid-color) 1px, transparent 1px);
#     background-size: 56px 56px;
#     pointer-events: none; z-index: 0;
# }

# /* SESSION BAR */
# .session-bar {
#     display: flex; justify-content: space-between; align-items: center;
#     background: rgba(0,8,18,0.75); border: 1px solid rgba(0,180,255,0.08);
#     border-radius: 6px; padding: 0.5rem 1.4rem; margin-bottom: 2rem;
#     font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
#     backdrop-filter: blur(20px);
#     box-shadow: 0 1px 20px rgba(0,0,0,0.4);
# }
# .session-bar .sid { color: var(--text-dim); letter-spacing: 0.05em; }
# .session-bar .sid span { color: var(--accent-blue); font-weight: 500; }
# .session-bar .status { color: var(--accent-green); letter-spacing: 0.25em; font-size: 0.62rem; }
# .session-bar .status::before { content: '● '; animation: blink 1.5s infinite; }
# .session-bar .badge {
#     font-family: 'Rajdhani', sans-serif; font-size: 0.58rem; font-weight: 700;
#     letter-spacing: 0.3em; text-transform: uppercase; color: var(--accent-cyan);
#     background: rgba(0,255,240,0.06); border: 1px solid rgba(0,255,240,0.25);
#     border-radius: 3px; padding: 0.15rem 0.7rem;
# }

# /* HERO */
# .hero-wrap {
#     display: flex; flex-direction: column; align-items: center; justify-content: center;
#     min-height: 92vh; padding: 3rem 1rem; position: relative; text-align: center;
# }
# .hero-wrap::before {
#     content: ''; position: absolute; width: 800px; height: 800px;
#     background: radial-gradient(circle, rgba(0,180,255,0.06) 0%, transparent 68%);
#     border-radius: 50%; top: 50%; left: 50%; transform: translate(-50%,-50%);
#     animation: pulse-bg 6s ease-in-out infinite;
# }
# @keyframes pulse-bg {
#     0%,100% { transform: translate(-50%,-50%) scale(1); opacity: 0.4; }
#     50% { transform: translate(-50%,-50%) scale(1.15); opacity: 0.9; }
# }
# .hero-eye {
#     font-size: 5.5rem; margin-bottom: 1.5rem;
#     animation: float 6s ease-in-out infinite;
#     filter: drop-shadow(0 0 50px rgba(0,180,255,1));
# }
# @keyframes float {
#     0%,100% { transform: translateY(0) rotate(-2deg); }
#     50% { transform: translateY(-24px) rotate(2deg); }
# }
# .hero-title {
#     font-family: 'Exo 2', sans-serif;
#     font-size: clamp(3.5rem, 8.5vw, 8rem); font-weight: 900; letter-spacing: 0.1em;
#     background: linear-gradient(140deg, #ffffff 0%, #60c8ff 35%, var(--accent-cyan) 100%);
#     -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
#     margin-bottom: 0.6rem; line-height: 0.9;
#     animation: reveal 0.8s ease-out both;
# }
# @keyframes reveal { from { opacity: 0; transform: translateY(32px); } to { opacity: 1; transform: translateY(0); } }
# .hero-sub {
#     font-family: 'Rajdhani', sans-serif; font-size: clamp(0.78rem, 1.8vw, 1rem);
#     font-weight: 300; letter-spacing: 0.5em; color: var(--text-dim);
#     margin-bottom: 0.6rem; text-transform: uppercase;
# }
# .hero-status {
#     font-family: 'IBM Plex Mono', monospace;
#     font-size: 0.65rem; color: var(--accent-green);
#     letter-spacing: 0.28em; margin-bottom: 2.5rem; opacity: 0.9;
# }
# .hero-status::before { content: '● '; animation: blink 1.5s infinite; }
# @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.1; } }

# /* STATS */
# .stats-row { display: flex; gap: 1.5rem; margin-bottom: 2.5rem; justify-content: center; flex-wrap: wrap; }
# .stat-item {
#     text-align: center; background: rgba(6,18,32,0.7);
#     border: 1px solid rgba(0,180,255,0.12); border-radius: 10px;
#     padding: 1rem 2rem; backdrop-filter: blur(20px); min-width: 105px;
#     transition: border-color 0.3s, box-shadow 0.3s;
# }
# .stat-item:hover { border-color: rgba(0,180,255,0.3); box-shadow: 0 0 20px rgba(0,180,255,0.1); }
# .stat-value { font-family: 'Exo 2', sans-serif; font-size: 1.7rem; font-weight: 900; color: var(--accent-blue); display: block; }
# .stat-label { font-size: 0.58rem; letter-spacing: 0.28em; color: var(--text-dim); text-transform: uppercase; margin-top: 0.3rem; display: block; }

# /* MODULE GRID */
# .module-grid {
#     display: grid; grid-template-columns: repeat(auto-fit, minmax(255px, 1fr));
#     gap: 1.25rem; width: 100%; max-width: 1240px; margin: 0 auto 3rem;
# }
# .mod-card {
#     background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 14px;
#     padding: 1.8rem 1.6rem; position: relative; overflow: hidden;
#     transition: all 0.38s cubic-bezier(0.23,1,0.32,1); backdrop-filter: blur(24px);
# }
# .mod-card::before {
#     content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1.5px;
#     background: linear-gradient(90deg, transparent 0%, var(--accent-blue) 30%, var(--accent-cyan) 70%, transparent 100%);
#     opacity: 0; transition: opacity 0.35s;
# }
# .mod-card::after {
#     content: ''; position: absolute; inset: 0;
#     background: radial-gradient(ellipse at 0% 0%, rgba(0,180,255,0.08) 0%, transparent 60%);
#     opacity: 0; transition: opacity 0.38s;
# }
# .mod-card:hover { border-color: rgba(0,180,255,0.32); transform: translateY(-6px) scale(1.005); box-shadow: var(--shadow-blue), var(--shadow-card); }
# .mod-card:hover::before { opacity: 1; }
# .mod-card:hover::after  { opacity: 1; }
# .mod-card.research-card { border-color: rgba(255,51,85,0.15); }
# .mod-card.research-card::before { background: linear-gradient(90deg, transparent, var(--accent-red), #ff8800, transparent); }
# .mod-card.research-card::after  { background: radial-gradient(ellipse at 0% 0%, rgba(255,51,85,0.07) 0%, transparent 60%); }
# .mod-card.research-card:hover { border-color: rgba(255,51,85,0.45); box-shadow: 0 0 50px rgba(255,51,85,0.1), var(--shadow-card); }

# .mod-icon { font-size: 1.9rem; margin-bottom: 0.9rem; display: block; line-height: 1; }
# .mod-name {
#     font-family: 'Rajdhani', sans-serif; font-size: 0.9rem; font-weight: 700;
#     letter-spacing: 0.22em; color: var(--accent-blue); text-transform: uppercase; margin-bottom: 0.45rem;
# }
# .mod-name.red { color: var(--accent-red); }
# .mod-tag {
#     display: inline-block; font-size: 0.55rem; letter-spacing: 0.15em;
#     color: var(--accent-cyan); background: rgba(0,255,240,0.06);
#     border: 1px solid rgba(0,255,240,0.18); border-radius: 3px;
#     padding: 0.12rem 0.55rem; margin-bottom: 0.65rem; text-transform: uppercase;
# }
# .mod-tag.red { color: var(--accent-red); background: rgba(255,51,85,0.06); border-color: rgba(255,51,85,0.22); }
# .mod-desc { font-size: 0.74rem; color: var(--text-secondary); line-height: 1.72; }
# .mod-meta {
#     font-size: 0.6rem; color: var(--text-dim); margin-top: 0.85rem;
#     border-top: 1px solid rgba(0,180,255,0.07); padding-top: 0.65rem;
#     letter-spacing: 0.03em; line-height: 1.5;
# }

# /* SCAN LINE */
# .scan-line {
#     width: 100%; max-width: 860px; height: 1px;
#     background: linear-gradient(90deg, transparent, rgba(0,180,255,0.3), rgba(0,255,240,0.5), rgba(0,180,255,0.3), transparent);
#     margin: 2rem auto; position: relative; overflow: hidden;
# }
# .scan-line::after {
#     content: ''; position: absolute; width: 90px; height: 100%;
#     background: linear-gradient(90deg, transparent, rgba(0,255,240,1), transparent);
#     animation: scan 3.5s linear infinite;
# }
# @keyframes scan { from { left: -90px; } to { left: 100%; } }

# /* APP HEADER */
# .app-header {
#     font-family: 'Exo 2', sans-serif; font-size: 1.6rem; font-weight: 800;
#     letter-spacing: 0.35em;
#     background: linear-gradient(135deg, #ffffff 0%, #70d4ff 60%, var(--accent-cyan) 100%);
#     -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
#     text-align: center; padding: 1.5rem 0 0.35rem;
# }
# .app-sub {
#     font-family: 'Rajdhani', sans-serif; font-size: 0.68rem; color: var(--text-dim);
#     letter-spacing: 0.5em; text-align: center; margin-bottom: 1.5rem; text-transform: uppercase;
# }

# /* NAV DIVIDER */
# .nav-divider {
#     display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;
# }
# .nav-divider-line { flex: 1; height: 1px; background: var(--border-subtle); }
# .nav-divider-label {
#     font-family: 'IBM Plex Mono', monospace; font-size: 0.55rem; color: var(--text-dim);
#     letter-spacing: 0.25em; text-transform: uppercase; white-space: nowrap;
# }

# /* BUTTONS */
# .stButton > button {
#     font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
#     letter-spacing: 0.1em !important; font-size: 0.78rem !important;
#     background: rgba(6,18,32,0.9) !important; color: var(--accent-blue) !important;
#     border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 7px !important;
#     padding: 0.65rem 0.8rem !important; transition: all 0.25s ease !important;
#     text-transform: uppercase !important; width: 100% !important;
#     white-space: nowrap !important;
# }
# .stButton > button:hover {
#     background: rgba(0,180,255,0.1) !important; border-color: rgba(0,180,255,0.4) !important;
#     color: var(--accent-cyan) !important;
#     box-shadow: 0 0 20px rgba(0,180,255,0.2), inset 0 0 15px rgba(0,180,255,0.05) !important;
#     transform: translateY(-2px) !important;
# }
# .stButton > button[kind="primary"] {
#     background: linear-gradient(135deg, rgba(0,90,180,0.5), rgba(0,180,255,0.25)) !important;
#     border-color: var(--accent-blue) !important; color: #fff !important;
#     box-shadow: 0 0 30px rgba(0,180,255,0.3) !important;
# }
# .stButton > button[kind="primary"]:hover {
#     box-shadow: 0 0 40px rgba(0,180,255,0.5) !important;
# }

# /* SECTION HEADERS */
# .section-hdr {
#     font-family: 'Exo 2', sans-serif; font-size: 1.2rem; font-weight: 700;
#     letter-spacing: 0.28em; color: var(--accent-blue); text-transform: uppercase;
#     padding: 0.5rem 0; border-bottom: 1px solid rgba(0,180,255,0.1);
#     margin-bottom: 0.5rem; position: relative;
# }
# .section-hdr::after {
#     content: ''; position: absolute; bottom: -1px; left: 0; width: 70px; height: 1.5px;
#     background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
# }
# .section-hdr.red { color: var(--accent-red); }
# .section-hdr.red::after { background: linear-gradient(90deg, var(--accent-red), #ff8800); }
# .section-sub { font-size: 0.7rem; color: var(--text-secondary); letter-spacing: 0.18em; margin-bottom: 1.8rem; text-transform: uppercase; }

# /* TERMINAL */
# .terminal {
#     background: rgba(0,6,16,0.95); border: 1px solid rgba(0,180,255,0.1);
#     border-left: 2px solid var(--accent-blue); border-radius: 0 5px 5px 0;
#     padding: 0.75rem 1.2rem; font-size: 0.7rem; color: var(--accent-green);
#     letter-spacing: 0.12em; margin-top: 1.5rem; position: relative; overflow: hidden;
#     box-shadow: inset 0 0 30px rgba(0,0,0,0.5);
# }
# .terminal::before {
#     content: ''; position: absolute; inset: 0;
#     background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,136,0.008) 2px, rgba(0,255,136,0.008) 4px);
#     pointer-events: none;
# }

# /* INFO BOX */
# .info-box {
#     background: rgba(0,8,20,0.8); border: 1px solid rgba(0,180,255,0.09);
#     border-radius: 8px; padding: 1.1rem 1.4rem; margin-bottom: 1.5rem;
#     font-size: 0.74rem; color: var(--text-secondary); line-height: 1.85;
#     box-shadow: inset 0 1px 0 rgba(0,180,255,0.05);
# }
# .info-box strong { color: var(--accent-blue); font-weight: 500; }

# /* STREAMLIT WIDGET OVERRIDES */
# .stFileUploader { background: var(--bg-card) !important; border: 1px dashed rgba(0,180,255,0.25) !important; border-radius: 10px !important; padding: 1rem !important; }
# .stTextInput > div > div { background: rgba(4,12,24,0.9) !important; border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 7px !important; color: var(--text-primary) !important; font-family: 'IBM Plex Mono', monospace !important; }
# .stTextInput > div > div:focus-within { border-color: rgba(0,180,255,0.4) !important; box-shadow: 0 0 12px rgba(0,180,255,0.12) !important; }
# .stSelectbox > div > div { background: rgba(4,12,24,0.9) !important; border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 7px !important; color: var(--text-primary) !important; }
# .stNumberInput > div > div { background: rgba(4,12,24,0.9) !important; border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 7px !important; }
# .stSlider > div > div > div { background: var(--accent-blue) !important; }

# div[data-testid="metric-container"] {
#     background: rgba(4,14,28,0.85) !important; border: 1px solid rgba(0,180,255,0.1) !important;
#     border-radius: 10px !important; padding: 1rem !important; transition: all 0.25s;
# }
# div[data-testid="metric-container"]:hover { border-color: rgba(0,180,255,0.28) !important; box-shadow: 0 0 16px rgba(0,180,255,0.08) !important; }
# div[data-testid="metric-container"] label { color: var(--text-dim) !important; font-size: 0.62rem !important; letter-spacing: 0.22em !important; font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important; }
# div[data-testid="metric-container"] div[data-testid="metric-value"] { color: var(--accent-blue) !important; font-family: 'Exo 2', sans-serif !important; font-weight: 800 !important; }

# div[data-testid="stDataFrame"] { background: rgba(4,14,28,0.85) !important; border: 1px solid rgba(0,180,255,0.1) !important; border-radius: 10px !important; overflow: hidden !important; }

# .stSuccess { background: rgba(0,255,136,0.06) !important; border: 1px solid rgba(0,255,136,0.25) !important; border-radius: 7px !important; color: var(--accent-green) !important; }
# .stError, .stWarning { background: rgba(255,51,85,0.06) !important; border: 1px solid rgba(255,51,85,0.25) !important; border-radius: 7px !important; }
# .stInfo { background: rgba(0,180,255,0.06) !important; border: 1px solid rgba(0,180,255,0.18) !important; border-radius: 7px !important; color: var(--accent-blue) !important; }

# hr { border-color: rgba(0,180,255,0.08) !important; margin: 1.5rem 0 !important; }
# ::-webkit-scrollbar { width: 3px; }
# ::-webkit-scrollbar-track { background: var(--bg-primary); }
# ::-webkit-scrollbar-thumb { background: rgba(0,180,255,0.4); border-radius: 2px; }
# .stSpinner > div { border-color: var(--accent-blue) transparent transparent transparent !important; }
# section[data-testid="stSidebar"] { display: none !important; }
# #MainMenu { visibility: hidden; }
# footer { visibility: hidden; }
# header { visibility: hidden; }

# @keyframes fadeInUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
# .stMarkdown, .stButton, .stFileUploader { animation: fadeInUp 0.38s ease-out both; }
# </style>
# """, unsafe_allow_html=True)


# @st.cache_resource
# def load_detector():
#     return PersonDetector()

# @st.cache_resource
# def load_osint():
#     return OSINTAudit()

# @st.cache_resource
# def load_emotion_model():
#     from core.emotion import process_frame_emotion
#     return process_frame_emotion

# @st.cache_resource
# def load_weapon_model_cached():
#     from core.weapon import load_weapon_model
#     return load_weapon_model()


# def render_session_bar():
#     sid = st.session_state.get("session_id", "PE-XXXXXXXX")
#     st.markdown(f"""
#     <div class="session-bar">
#         <div class="sid"><span>●</span>&nbsp;&nbsp;SESSION: <span>{sid}</span></div>
#         <div class="status">ALL SYSTEMS ONLINE</div>
#         <div class="badge">OPEN ACCESS</div>
#     </div>
#     """, unsafe_allow_html=True)


# def back_button():
#     if st.button("← BACK TO MODULES"):
#         st.session_state.page = "home"
#         st.rerun()


# def landing():
#     st.markdown("""
#     <div class="hero-wrap">
#       <div class="hero-eye">👁</div>
#       <div class="hero-title">PHANTOMEYE</div>
#       <div class="hero-sub">AI-Powered Surveillance Intelligence System</div>
#       <div class="hero-status">[ SYSTEM ONLINE ] · OPEN ACCESS · BUILD v3.4</div>

#       <div class="stats-row">
#         <div class="stat-item"><span class="stat-value">13</span><span class="stat-label">Modules</span></div>
#         <div class="stat-item"><span class="stat-value">4</span><span class="stat-label">Novel Algorithms</span></div>
#         <div class="stat-item"><span class="stat-value">9</span><span class="stat-label">Weapon Classes</span></div>
#         <div class="stat-item"><span class="stat-value">CPU</span><span class="stat-label">No GPU Required</span></div>
#       </div>

#       <div class="scan-line"></div>

#       <div class="module-grid">
#         <div class="mod-card">
#           <div class="mod-icon">🎯</div>
#           <div class="mod-name">Person Detection</div>
#           <div class="mod-tag">YOLOv8-nano</div>
#           <div class="mod-desc">Real-time person detection on any uploaded image. Returns bounding boxes and per-person confidence scores. Runs entirely on CPU — no GPU required.</div>
#           <div class="mod-meta">Model: yolov8n.pt · Class 0 only · Confidence: 0.4 · CPU optimized</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">🔥</div>
#           <div class="mod-name">Behavioral Analytics</div>
#           <div class="mod-tag">ByteTrack · OpenCV</div>
#           <div class="mod-desc">Persistent person IDs across frames, live behavioral heatmap showing movement density, per-person dwell times, and automated loitering alerts from any video.</div>
#           <div class="mod-meta">Tracker: ByteTrack IOU · Heatmap: NumPy · Alert threshold: 60s · Max: 15s</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">🕵️</div>
#           <div class="mod-name">OSINT Audit</div>
#           <div class="mod-tag">LBPH Face Recognition</div>
#           <div class="mod-desc">Upload a face and receive a Privacy Exposure Score from 0 to 100. LBPH embeddings matched against a reference gallery. Risk classified as LOW, MEDIUM, or HIGH.</div>
#           <div class="mod-meta">Engine: OpenCV LBPH · Similarity: cosine · Score: 0–100 · No data stored</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">🧠</div>
#           <div class="mod-name">Emotion Intelligence</div>
#           <div class="mod-tag">DeepFace · TensorFlow</div>
#           <div class="mod-desc">Multi-face emotion analysis. Returns dominant emotion, estimated age, and gender per face. False-positive filter rejects faces smaller than 15% of frame area.</div>
#           <div class="mod-meta">Backend: DeepFace · Detector: OpenCV · Min face: 15% · 7 emotion classes</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">💬</div>
#           <div class="mod-name">NL Query Engine</div>
#           <div class="mod-tag">Groq LLaMA 3</div>
#           <div class="mod-desc">Type a surveillance query in plain English or Roman Urdu. LLaMA 3 extracts structured filters — emotion, gender, age, dwell time, loitering — then matches against records.</div>
#           <div class="mod-meta">Model: llama-3.1-8b-instant · English + Roman Urdu · Output: JSON filters</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">⚠️</div>
#           <div class="mod-name">Weapon Detection</div>
#           <div class="mod-tag">YOLOv8 Custom · 9 Classes</div>
#           <div class="mod-desc">Custom YOLOv8 trained on 714 real weapon images. Handgun 89.5%, Shotgun 96.3%, SMG 98.6% average precision. Immediate threat alert fires on any detection.</div>
#           <div class="mod-meta">Classes: Handgun · Knife · Shotgun · Sniper · AR · SMG · Sword · Bazooka · GL</div>
#         </div>
#         <div class="mod-card research-card">
#           <div class="mod-icon">📊</div>
#           <div class="mod-name red">Threat Momentum Score</div>
#           <div class="mod-tag red">Novel Algorithm · TMS v1.0</div>
#           <div class="mod-desc">Original research. Accumulates threat signals over time using a compound interest model — loitering, stress emotion, rapid movement, restricted zone, gaze anomaly, group formation.</div>
#           <div class="mod-meta">6 signals · Decay: 45s half-life · Amplifier: score/200 · 5 threat levels</div>
#         </div>
#         <div class="mod-card research-card">
#           <div class="mod-icon">🧬</div>
#           <div class="mod-name red">Behavioral DNA</div>
#           <div class="mod-tag red">Novel Algorithm · BDF v1.0</div>
#           <div class="mod-desc">Camera-agnostic person re-identification using behavioral signature alone. Identifies the same person across cameras without face recognition — works through masks, hats, distance.</div>
#           <div class="mod-meta">5 components: gait · velocity · spatial · social distance · dwell zones · Threshold: 82%</div>
#         </div>
#         <div class="mod-card research-card">
#           <div class="mod-icon">🕸️</div>
#           <div class="mod-name red">Social Graph</div>
#           <div class="mod-tag red">Novel Algorithm · SGI v1.0</div>
#           <div class="mod-desc">Detects who is associated with whom from movement correlation alone — no prior information needed. Three people entering separately but coordinating get flagged before any overt action.</div>
#           <div class="mod-meta">Proximity · velocity sync · dwell overlap · BFS connected-component group detection</div>
#         </div>
#         <div class="mod-card research-card">
#           <div class="mod-icon">🚀</div>
#           <div class="mod-name red">Predictive Exit Vector</div>
#           <div class="mod-tag red">Novel Algorithm · PEV v1.0</div>
#           <div class="mod-desc">Predicts which frame boundary a person will cross and how many seconds remain — 3 to 5 seconds before actual exit. Velocity smoothing plus linear trajectory extrapolation. Designed for camera handoff in multi-camera surveillance grids.</div>
#           <div class="mod-meta">Trajectory extrapolation · velocity smoothing · boundary proximity · confidence scoring · no open-source equivalent</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">📄</div>
#           <div class="mod-name">Intel Report</div>
#           <div class="mod-tag">fpdf2 · PDF Export</div>
#           <div class="mod-desc">Generate a classified PDF intelligence report from any session. Session overview, weapon threat log in red, per-subject behavioral records, and NL query history.</div>
#           <div class="mod-meta">fpdf2 · Dark bg + green text · CLASSIFIED header · Threat sections in red</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">⚡</div>
#           <div class="mod-name">System Intel</div>
#           <div class="mod-tag">Live Status</div>
#           <div class="mod-desc">Live system dashboard with all active modules, tech stack, benchmark results, API endpoint reference, and full deployment metadata for complete transparency.</div>
#           <div class="mod-meta">v3.4.0 · HuggingFace Spaces · FastAPI OAS 3.1 · GitHub open source</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">📖</div>
#           <div class="mod-name">User Guide</div>
#           <div class="mod-tag">Complete Documentation</div>
#           <div class="mod-desc">Complete interactive user guide — Quick Start in 5 minutes, per-module step-by-step walkthroughs, novel algorithm deep dives with math, API reference, and FAQ. Re-run the onboarding tour anytime.</div>
#           <div class="mod-meta">5-step onboarding · 12 module walkthroughs · 4 algorithm deep dives · API reference · FAQ</div>
#         </div>
#       </div>
#     </div>
#     """, unsafe_allow_html=True)

#     cols = st.columns([1, 2, 1])
#     with cols[1]:
#         if st.button("INITIALIZE SYSTEM  →", key="enter_btn"):
#             if not st.session_state.get("first_visit_done", False):
#                 st.session_state.page = "welcome"
#             else:
#                 st.session_state.page = "home"
#             st.rerun()


# def home():
#     render_session_bar()
#     st.markdown('<div class="app-header">👁 PHANTOMEYE</div>', unsafe_allow_html=True)
#     st.markdown('<div class="app-sub">SELECT INTELLIGENCE MODULE · ALL SYSTEMS ACTIVE</div>', unsafe_allow_html=True)

#     # Row 1 — Core modules
#     st.markdown("""
#     <div class="nav-divider">
#         <div class="nav-divider-line"></div>
#         <div class="nav-divider-label">Core Intelligence</div>
#         <div class="nav-divider-line"></div>
#     </div>
#     """, unsafe_allow_html=True)

#     row1 = [
#         ("DETECTION", "Detection"),
#         ("ANALYTICS", "Analytics"),
#         ("OSINT",     "OSINT"),
#         ("EMOTION",   "Emotion"),
#         ("NL QUERY",  "NL Query"),
#         ("WEAPON",    "Weapon"),
#     ]
#     cols1 = st.columns(6)
#     for i, (key, label) in enumerate(row1):
#         with cols1[i]:
#             if st.button(label, key=f"mod_{key}"):
#                 st.session_state.page = key
#                 st.rerun()

#     # Row 2 — Research + utility
#     st.markdown("""
#     <div class="nav-divider" style="margin-top:0.75rem;">
#         <div class="nav-divider-line"></div>
#         <div class="nav-divider-label">Novel Research · Utility</div>
#         <div class="nav-divider-line"></div>
#     </div>
#     """, unsafe_allow_html=True)

#     row2 = [
#         ("THREAT", "Threat Score"),
#         ("BDF",    "Behavioral DNA"),
#         ("SGI",    "Social Graph"),
#         ("PEV",    "Predictive Exit"),
#         ("REPORT", "Report"),
#         ("INTEL",  "System"),
#         ("ZONE",   "Zone Intel"),
#         ("GUIDE",  "Guide"),
#     ]
#     cols2 = st.columns(8)
#     for i, (key, label) in enumerate(row2):
#         with cols2[i]:
#             if st.button(label, key=f"mod2_{key}"):
#                 st.session_state.page = key
#                 st.rerun()

#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown(
#         '<div class="terminal">[ PHANTOMEYE v3.4 ] · YOLOv8 loaded · ByteTrack active · '
#         'DeepFace online · Groq LLaMA connected · Weapon model ready · '
#         'TMS v1.0 active · BDF v1.0 active · SGI v1.0 active · PEV v1.0 active · '
#         'User Guide active · All 13 modules ONLINE</div>',
#         unsafe_allow_html=True
#     )


# def detection_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">Person Detection</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">YOLOv8-nano · CPU inference · class 0 persons only · confidence threshold 0.4</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> Upload any image and PhantomEye runs YOLOv8-nano inference entirely on CPU. Configured for class 0 (person) detection only at a confidence threshold of 0.4. Each detected person receives a bounding box and confidence score. Expand the detection log below the output image to inspect raw bbox coordinates and confidence per subject. No GPU required at any point.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">yolov8n.pt · device: cpu · class 0 only · confidence threshold: 0.4</div>', unsafe_allow_html=True)

#     uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], key="det_up")
#     if uploaded:
#         data  = np.frombuffer(uploaded.read(), np.uint8)
#         image = cv2.imdecode(data, cv2.IMREAD_COLOR)
#         if image is None:
#             st.error("Cannot decode image.")
#             return
#         with st.spinner("Running inference..."):
#             detector   = load_detector()
#             t0         = time.time()
#             detections = detector.detect(image)
#             elapsed    = round(time.time() - t0, 3)
#             annotated  = detector.draw(image, detections)
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("PERSONS DETECTED", len(detections))
#         c2.metric("INFERENCE TIME",   f"{elapsed}s")
#         c3.metric("MODEL",            "YOLOv8n")
#         c4.metric("DEVICE",           "CPU")
#         st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection output", use_container_width=True)
#         if detections:
#             st.markdown('<div class="section-hdr">Detection Log</div>', unsafe_allow_html=True)
#             st.markdown('<div class="section-sub">Expand each entry to inspect bounding box coordinates and confidence score</div>', unsafe_allow_html=True)
#             for i, d in enumerate(detections):
#                 with st.expander(f"PERSON_{i+1:03d}  ·  CONF: {d['confidence']}"):
#                     st.json({"id": i+1, "bbox": list(d["bbox"]), "confidence": d["confidence"]})


# def analytics_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">Behavioral Analytics</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">ByteTrack · behavioral heatmap · dwell time · loitering alerts</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> Upload a video and PhantomEye processes up to 15 seconds of footage. ByteTrack assigns a persistent ID to each person and maintains it across frames, including through brief occlusion. A NumPy heatmap accumulates every pixel position each person visits — high-activity zones appear red. Dwell time is tracked per ID in seconds. If any person remains in one area beyond the loitering threshold, an alert fires listing their tracked ID.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">ByteTrack IOU matching · heatmap: NumPy accumulation · loitering threshold: 60s · max analysis window: 15s</div>', unsafe_allow_html=True)

#     uploaded = st.file_uploader("", type=["mp4", "avi", "mov"], key="ana_up")
#     if uploaded:
#         tmp = Path("outputs") / f"tmp_{int(time.time())}.mp4"
#         tmp.parent.mkdir(exist_ok=True)
#         with open(tmp, "wb") as f:
#             f.write(uploaded.read())
#         cap   = cv2.VideoCapture(str(tmp))
#         fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
#         w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         cap.release()
#         st.markdown(f'<div class="terminal">{w}x{h} @ {fps}fps · {total} total frames · analysis cap: {min(total, fps*15)} frames</div>', unsafe_allow_html=True)
#         if st.button("RUN BEHAVIORAL ANALYSIS"):
#             detector = load_detector()
#             tracker  = ByteTracker()
#             analyzer = BehavioralAnalyzer(w, h, fps)
#             cap      = cv2.VideoCapture(str(tmp))
#             limit    = min(total, fps * 15)
#             prog     = st.progress(0)
#             stat     = st.empty()
#             for i in range(limit):
#                 ret, frame = cap.read()
#                 if not ret: break
#                 dets   = detector.detect(frame)
#                 active = tracker.update(dets)
#                 analyzer.update(active)
#                 prog.progress(int((i / limit) * 100))
#                 if i % 25 == 0:
#                     stat.markdown(f'<div class="terminal">Processing frame {i}/{limit} · active persons: {len(active)}</div>', unsafe_allow_html=True)
#             cap.release()
#             tmp.unlink(missing_ok=True)
#             prog.progress(100)
#             stat.empty()
#             s = analyzer.summary()
#             st.success("Analysis complete")
#             c1, c2, c3, c4 = st.columns(4)
#             c1.metric("TOTAL PERSONS", s.get("total_persons", 0))
#             c2.metric("AVG DWELL",     f"{s.get('avg_dwell_sec', 0)}s")
#             c3.metric("MAX DWELL",     f"{s.get('max_dwell_sec', 0)}s")
#             c4.metric("LOITER ALERTS", s.get("total_alerts", 0))
#             if s.get("total_alerts", 0) > 0:
#                 st.warning(f"Loitering detected — Subject IDs: {s.get('loiterers', [])}")
#             heat = analyzer.get_heatmap_overlay(np.zeros((h, w, 3), dtype=np.uint8))
#             st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), caption="Behavioral heatmap — red zones indicate highest activity density", use_container_width=True)


# def osint_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">OSINT Privacy Audit</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">LBPH face embedding · gallery match · exposure score 0–100 · risk classification</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> Upload a face photo and PhantomEye extracts an LBPH (Local Binary Pattern Histogram) embedding from the detected face region. This is compared against every person in the reference gallery using cosine similarity. The Privacy Exposure Score (0–100) reflects recognition confidence — higher score means stronger match. Risk level: LOW (score &lt; 40), MEDIUM (40–70), HIGH (&gt; 70). All processing in-session only — nothing stored server-side at any point.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">Engine: OpenCV LBPH · Similarity: cosine distance · Score: 0–100 · Risk: LOW / MEDIUM / HIGH · No data retention</div>', unsafe_allow_html=True)

#     c1, c2 = st.columns([1, 1])
#     with c1:
#         query_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="osint_up")
#     with c2:
#         osint = load_osint()
#         st.metric("GALLERY SIZE", f"{len(osint.gallery)} persons")
#         st.metric("ENGINE",       "LBPH Face Recognition")
#     if query_file and st.button("EXECUTE AUDIT"):
#         data  = np.frombuffer(query_file.read(), np.uint8)
#         image = cv2.imdecode(data, cv2.IMREAD_COLOR)
#         if image is None:
#             st.error("Cannot decode image.")
#             return
#         with st.spinner("Running audit..."):
#             result = osint.audit(image, query_id=Path(query_file.name).stem)
#         c1, c2, c3 = st.columns(3)
#         c1.metric("RISK LEVEL",     result["risk_level"])
#         c2.metric("EXPOSURE SCORE", f"{result['exposure_score']}/100")
#         c3.metric("MATCHES FOUND",  len(result["matches"]))
#         st.markdown(f'<div class="terminal">{result["message"]}</div>', unsafe_allow_html=True)
#         if result["matches"]:
#             st.markdown('<div class="section-hdr">Match Log</div>', unsafe_allow_html=True)
#             for m in result["matches"]:
#                 st.markdown(f'<div class="terminal">MATCH: {m["matched_id"]} · CONF: {m["confidence"]}% · SOURCE: {m["source"]}</div>', unsafe_allow_html=True)
#         vis = osint.visualize(image, result)
#         st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="OSINT visualization output", use_container_width=True)


# def emotion_page():
#     render_session_bar()
#     process_frame_emotion = load_emotion_model()
#     back_button()
#     st.markdown('<div class="section-hdr">Emotion Intelligence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">DeepFace · TensorFlow · dominant emotion · age · gender per face</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> PhantomEye runs DeepFace analysis on every detected face in the uploaded image. Returns dominant emotion from 7 classes (angry, fear, sad, happy, surprise, neutral, disgust), an estimated age, and gender classification. A false-positive filter discards any face region smaller than 15% of the frame area — prevents noise from distant or partially visible faces. Multiple faces in a single image are processed independently.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">DeepFace + TensorFlow · OpenCV face detector · min face size: 15% of frame · 7 emotion classes · multi-subject</div>', unsafe_allow_html=True)

#     uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
#     if uploaded:
#         from PIL import Image
#         img       = Image.open(uploaded).convert("RGB")
#         frame     = np.array(img)
#         frame_bgr = frame[:, :, ::-1].copy()
#         with st.spinner("Analyzing faces..."):
#             annotated, results = process_frame_emotion(frame_bgr)
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(frame, caption="Original", use_container_width=True)
#         with col2:
#             st.image(annotated[:, :, ::-1], caption="Emotion analysis output", use_container_width=True)
#         if results:
#             st.markdown("<hr>")
#             st.markdown('<div class="section-hdr">Detected Subjects</div>', unsafe_allow_html=True)
#             for i, r in enumerate(results):
#                 emotion = r.get("dominant_emotion", "N/A").upper()
#                 age     = int(r.get("age", 0))
#                 gender  = r.get("dominant_gender", r.get("gender", "N/A"))
#                 if isinstance(gender, dict):
#                     gender = max(gender, key=gender.get)
#                 c1, c2, c3 = st.columns(3)
#                 c1.metric(f"SUBJECT {i+1} EMOTION", emotion)
#                 c2.metric("AGE ESTIMATE",            f"{age} yrs")
#                 c3.metric("GENDER",                  gender.upper())
#         else:
#             st.warning("No faces detected in this image.")
#     else:
#         st.info("Upload a face image to begin analysis.")


# def nlquery_page():
#     render_session_bar()
#     from core.nlquery import parse_nl_query, apply_filters
#     back_button()
#     st.markdown('<div class="section-hdr">NL Query Engine</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Groq LLaMA 3 · English + Roman Urdu · structured filter extraction</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> Type any surveillance query in natural language — English or Roman Urdu both work. Groq's LLaMA 3 (llama-3.1-8b-instant) parses the intent and extracts structured filters: emotion type, gender, age range, minimum dwell time, and loitering status. Filters are applied against person records and matching subjects are returned in a filterable table. This is the first open-source surveillance system with multilingual NL query support including Roman Urdu.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">llama-3.1-8b-instant via Groq · JSON structured filter extraction · Roman Urdu supported · apply_filters() on person records</div>', unsafe_allow_html=True)

#     query = st.text_input("Enter your query", placeholder="show me angry men who were loitering  |  log jo loiter kar rahy thy")
#     if query:
#         with st.spinner("Parsing query..."):
#             result = parse_nl_query(query)
#         if result['success']:
#             filters = result['filters']
#             st.success(f"Understood: {filters['summary']}")
#             col1, col2, col3 = st.columns(3)
#             col1.metric("EMOTION",   filters['emotion']  or "ANY")
#             col2.metric("GENDER",    filters['gender']   or "ANY")
#             col3.metric("MAX AGE",   filters['max_age']  or "ANY")
#             col4, col5 = st.columns(2)
#             col4.metric("LOITERING", "YES" if filters['loitering'] else "ANY")
#             col5.metric("MIN DWELL", f"{filters['min_dwell_seconds']}s" if filters['min_dwell_seconds'] else "ANY")
#             st.markdown("<hr>")
#             st.markdown('<div class="section-hdr">Filter Results — Sample Dataset</div>', unsafe_allow_html=True)
#             sample_records = [
#                 {"id": 1, "emotion": "angry",   "gender": "Man",   "age": 28, "dwell_seconds": 45,  "loitering": False},
#                 {"id": 2, "emotion": "neutral",  "gender": "Woman", "age": 22, "dwell_seconds": 180, "loitering": True},
#                 {"id": 3, "emotion": "happy",    "gender": "Man",   "age": 35, "dwell_seconds": 20,  "loitering": False},
#                 {"id": 4, "emotion": "angry",    "gender": "Man",   "age": 41, "dwell_seconds": 200, "loitering": True},
#                 {"id": 5, "emotion": "sad",      "gender": "Woman", "age": 19, "dwell_seconds": 90,  "loitering": False},
#                 {"id": 6, "emotion": "fear",     "gender": "Man",   "age": 26, "dwell_seconds": 310, "loitering": True},
#             ]
#             matched = apply_filters(sample_records, filters)
#             if matched:
#                 st.success(f"{len(matched)} subject(s) matched from {len(sample_records)} records")
#                 import pandas as pd
#                 st.dataframe(pd.DataFrame(matched), use_container_width=True)
#             else:
#                 st.warning("No subjects matched this query.")
#         else:
#             st.error(f"Parse failed: {result['error']}")
#     else:
#         st.info("Type a query above — English or Roman Urdu both work.")


# def weapon_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">Weapon Detection</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">YOLOv8 custom trained · 9 weapon classes · real-time threat alert</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> A custom YOLOv8 model trained from scratch on 714 real-world weapon images across 9 classes — trained on Kaggle T4 GPU. Achieves Handgun 89.5%, Shotgun 96.3%, SMG 98.6% average precision at mAP50 53.2%. Upload any image — detected weapons are highlighted with red bounding boxes and an immediate threat alert fires with the weapon class and confidence score. A clean result confirms the scene is clear.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">weapon_detector.pt · mAP50: 53.2% · Handgun: 89.5% · Shotgun: 96.3% · SMG: 98.6% · 714 training images · 9 classes</div>', unsafe_allow_html=True)

#     uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
#     if uploaded:
#         from PIL import Image
#         from core.weapon import detect_weapons
#         img       = Image.open(uploaded).convert("RGB")
#         frame     = np.array(img)
#         frame_bgr = frame[:, :, ::-1].copy()
#         model     = load_weapon_model_cached()
#         with st.spinner("Scanning for threats..."):
#             annotated, detections = detect_weapons(frame_bgr, model)
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(frame, caption="Original", use_container_width=True)
#         with col2:
#             st.image(annotated[:, :, ::-1], caption="Threat analysis output", use_container_width=True)
#         st.markdown("<hr>")
#         if detections:
#             st.error(f"THREAT DETECTED — {len(detections)} weapon(s) identified")
#             st.markdown('<div class="section-hdr red">Detected Threats</div>', unsafe_allow_html=True)
#             for d in detections:
#                 c1, c2 = st.columns(2)
#                 c1.metric("WEAPON CLASS", d['class_name'])
#                 c2.metric("CONFIDENCE",   f"{d['confidence']:.0%}")
#         else:
#             st.success("No weapons detected — scene clear")
#     else:
#         st.info("Upload an image to begin weapon scan.")


# def threat_page():
#     render_session_bar()
#     back_button()
#     from core.threat_momentum import ThreatMomentumEngine

#     st.markdown('<div class="section-hdr red">Threat Momentum Score</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Novel temporal threat accumulation · compound behavioral signal model · TMS v1.0</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>Research contribution:</strong> Unlike binary threat detection systems that output a single yes/no result, TMS accumulates behavioral signals over time using a compound interest model. Each new signal contributes to the score weighted by importance. When the score is already elevated, new signals contribute proportionally more — the amplifier effect. The score decays with a 45-second half-life when no signals arrive, modeling how real threat situations escalate gradually, not instantaneously.<br><br><strong>6 signals and weights:</strong> loitering (0.28) · stress emotion (0.22) · rapid movement (0.18) · proximity violation (0.15) · gaze anomaly (0.10) · group formation (0.07)</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">TMS v1.0 · decay half-life: 45s · amplifier: 1 + score/200 · 5 levels: CLEAR / LOW / MEDIUM / HIGH / CRITICAL</div>', unsafe_allow_html=True)

#     if "tms_engine" not in st.session_state:
#         st.session_state.tms_engine = ThreatMomentumEngine()
#     engine = st.session_state.tms_engine

#     st.markdown("### Subject Input")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         person_id     = st.number_input("Person ID", min_value=1, value=1)
#         dwell_seconds = st.number_input("Dwell Time (seconds)", min_value=0.0, value=0.0, step=5.0)
#         is_loitering  = st.checkbox("Loitering detected")
#     with c2:
#         emotion       = st.selectbox("Detected Emotion", ["none", "neutral", "angry", "fear", "disgust", "sad", "happy", "surprise"])
#         in_restricted = st.checkbox("In restricted zone")
#         group_anomaly = st.checkbox("Group anomaly detected")
#     with c3:
#         px = st.number_input("Position X (px)", min_value=0, value=320)
#         py = st.number_input("Position Y (px)", min_value=0, value=240)

#     col_a, col_b = st.columns(2)
#     with col_a:
#         if st.button("UPDATE THREAT SCORE", type="primary"):
#             result = engine.update_person(
#                 person_id=person_id, position=(px, py),
#                 emotion=None if emotion == "none" else emotion,
#                 dwell_seconds=dwell_seconds, is_loitering=is_loitering,
#                 in_restricted_zone=in_restricted, group_anomaly=group_anomaly,
#             )
#             st.session_state.last_tms = result
#     with col_b:
#         if st.button("RESET THIS PERSON"):
#             engine.reset_person(person_id)
#             if "last_tms" in st.session_state:
#                 del st.session_state.last_tms
#             st.success(f"Person {person_id} profile cleared.")

#     if "last_tms" in st.session_state:
#         r = st.session_state.last_tms
#         level_colors = {"CLEAR": "#10b981", "LOW": "#3b82f6", "MEDIUM": "#f59e0b", "HIGH": "#ef4444", "CRITICAL": "#ff0033"}
#         color = level_colors.get(r.threat_level, "#ffffff")
#         st.markdown(f"""
#         <div style="text-align:center; padding:2.5rem; margin:1.5rem 0;
#             background:rgba(0,4,12,0.97); border:2px solid {color};
#             border-radius:12px; box-shadow: 0 0 60px {color}18;">
#             <div style="font-family:'IBM Plex Mono',monospace; font-size:0.58rem; color:#2a4060; letter-spacing:0.4em; margin-bottom:0.8rem; text-transform:uppercase;">
#                 Threat Momentum Score · Person {r.person_id}
#             </div>
#             <div style="font-size:5.5rem; font-weight:900; color:{color}; font-family:'Exo 2',sans-serif; line-height:0.9; letter-spacing:-0.02em;">{r.tms_score:.1f}</div>
#             <div style="font-size:1rem; font-weight:700; color:{color}; letter-spacing:0.5em; margin-top:0.7rem; font-family:'Rajdhani',sans-serif;">{r.threat_level}</div>
#             <div style="font-size:0.62rem; color:#2a4060; margin-top:0.8rem; font-family:'IBM Plex Mono',monospace; letter-spacing:0.08em;">
#                 Momentum: {r.momentum:+.2f}/frame &nbsp;&nbsp;|&nbsp;&nbsp; Time in system: {r.time_in_system}s
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
#         if r.alert:
#             st.error(r.alert_message)
#         c1, c2, c3 = st.columns(3)
#         c1.metric("ACTIVE SIGNALS", len(r.active_signals))
#         c2.metric("MOMENTUM",       f"{r.momentum:+.3f}")
#         c3.metric("TIME IN SYSTEM", f"{r.time_in_system}s")
#         if r.signal_breakdown:
#             st.markdown('<div class="section-hdr">Signal Breakdown</div>', unsafe_allow_html=True)
#             import pandas as pd
#             df = pd.DataFrame([{"Signal": k.replace("_", " ").upper(), "Score Contribution": round(v, 3)} for k, v in r.signal_breakdown.items()])
#             st.dataframe(df, use_container_width=True)

#     st.markdown("<hr>")
#     st.markdown('<div class="section-hdr">Session Summary</div>', unsafe_allow_html=True)
#     summary = engine.summary()
#     s1, s2, s3, s4 = st.columns(4)
#     s1.metric("PERSONS TRACKED", summary["total_persons_tracked"])
#     s2.metric("TOTAL ALERTS",    summary["total_alerts"])
#     s3.metric("HIGHEST TMS",     summary["highest_tms"])
#     s4.metric("AVG TMS",         summary["avg_tms"])
#     if summary["level_distribution"]:
#         st.markdown('<div class="terminal">Distribution: ' + ' · '.join(f"{k}: {v}" for k, v in summary["level_distribution"].items()) + '</div>', unsafe_allow_html=True)
#     if st.button("RESET ALL PROFILES"):
#         engine.reset_all()
#         if "last_tms" in st.session_state:
#             del st.session_state.last_tms
#         st.success("All threat profiles cleared.")


# def bdf_page():
#     render_session_bar()
#     back_button()
#     from core.behavioral_dna import BehavioralDNAEngine

#     st.markdown('<div class="section-hdr red">Behavioral DNA Fingerprint</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Camera-agnostic re-identification · no face required · pure movement signature · BDF v1.0</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>Research contribution:</strong> Identifies the same person across cameras using behavioral signature alone — gait rhythm, velocity profile, spatial preference zones, social distance pattern, and dwell locations. Works with masks, hats, and at distances where face recognition completely fails. When a person re-enters the scene with a new tracking ID, BDF matches them to their previous identity using cosine similarity on a 5-component behavioral feature vector. Match threshold: 82%.<br><br><strong>5 behavioral components:</strong> gait signature (stride rhythm histogram) · velocity profile (speed distribution) · spatial preference (normalized grid heatmap) · social distance average · dwell zone signature (stopping locations)</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">BDF v1.0 · 5 behavioral signals · cosine similarity · match threshold: 82% · min observations: 15 frames</div>', unsafe_allow_html=True)

#     if "bdf_engine" not in st.session_state:
#         st.session_state.bdf_engine = BehavioralDNAEngine(640, 480)
#     engine = st.session_state.bdf_engine

#     st.markdown("### Add Observations")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         obs_id   = st.number_input("Person ID", min_value=1, value=1)
#         pos_x    = st.number_input("Position X", min_value=0, max_value=640, value=320)
#     with c2:
#         pos_y    = st.number_input("Position Y", min_value=0, max_value=480, value=240)
#         soc_dist = st.number_input("Nearest person distance (px)", min_value=0.0, value=100.0)
#     with c3:
#         n_obs = st.number_input("Observations to simulate", min_value=1, max_value=100, value=30)

#     if st.button("SIMULATE OBSERVATIONS"):
#         for i in range(int(n_obs)):
#             x = int(pos_x + i * 2 + np.random.randn() * 3)
#             y = int(pos_y + np.sin(i * 0.3) * 15 + np.random.randn() * 2)
#             engine.observe(obs_id, (max(0, x), max(0, y)), soc_dist)
#         st.success(f"Added {n_obs} observations for Person {obs_id}")

#     col_a, col_b = st.columns(2)
#     with col_a:
#         if st.button("REGISTER TO GALLERY", type="primary"):
#             bdf = engine.extract_and_register(obs_id)
#             if bdf:
#                 st.success(f"Person {obs_id} registered — confidence: {bdf.confidence:.2f} | observations: {bdf.observation_count}")
#             else:
#                 st.warning(f"Insufficient data. Need at least 15 observations for Person {obs_id}.")
#     with col_b:
#         if st.button("MATCH AGAINST GALLERY"):
#             result = engine.match_against_gallery(obs_id)
#             st.session_state.last_bdf = result

#     if "last_bdf" in st.session_state:
#         r = st.session_state.last_bdf
#         color = "#00b4ff" if r.is_match else "#10b981"
#         st.markdown(f"""
#         <div style="padding:2rem; margin:1rem 0; background:rgba(0,4,12,0.97);
#             border:2px solid {color}; border-radius:10px; box-shadow: 0 0 40px {color}18;">
#             <div style="font-family:'IBM Plex Mono',monospace; font-size:0.58rem; color:#2a4060; letter-spacing:0.35em; margin-bottom:0.6rem; text-transform:uppercase;">
#                 Behavioral DNA Match · Person {r.query_id}
#             </div>
#             <div style="font-size:2.2rem; font-weight:900; color:{color}; font-family:'Exo 2',sans-serif; letter-spacing:0.05em;">{"MATCH FOUND" if r.is_match else "NO MATCH"}</div>
#             <div style="font-size:0.72rem; color:#5a8090; margin-top:0.8rem; font-family:'IBM Plex Mono',monospace; line-height:1.65;">{r.explanation}</div>
#         </div>
#         """, unsafe_allow_html=True)
#         c1, c2, c3 = st.columns(3)
#         c1.metric("SIMILARITY",  f"{r.similarity:.1%}")
#         c2.metric("MATCHED ID",  str(r.matched_id) if r.matched_id else "None")
#         c3.metric("CONFIDENCE",  f"{r.confidence:.2f}")

#     st.markdown("<hr>")
#     st.markdown('<div class="section-hdr">Gallery & Session</div>', unsafe_allow_html=True)
#     summary = engine.summary()
#     s1, s2, s3, s4 = st.columns(4)
#     s1.metric("TRACKED",   summary["persons_tracked"])
#     s2.metric("BDF READY", summary["bdf_ready"])
#     s3.metric("GALLERY",   summary["gallery_size"])
#     s4.metric("MATCHES",   summary["matches_detected"])
#     if st.button("RESET ALL"):
#         engine.reset_all()
#         if "last_bdf" in st.session_state:
#             del st.session_state.last_bdf
#         st.success("BDF engine reset.")


# def sgi_page():
#     render_session_bar()
#     back_button()
#     from core.social_graph import SocialGraphEngine

#     st.markdown('<div class="section-hdr red">Social Graph Intelligence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Real-time group detection · no prior information · pure behavioral correlation · SGI v1.0</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>Research contribution:</strong> Detects "who is with whom" from surveillance footage without any prior information — no face recognition, no name lists, no pre-registration. Three bank robbers entering a building separately — SGI detects their association before any overt action occurs, purely from movement correlation. Uses three behavioral signals: spatial proximity, velocity synchronization (do they accelerate and decelerate together?), and shared dwell zones. Connected-component BFS then extracts groups from the link graph.<br><br><strong>Link strength formula:</strong> proximity score (0.40) + Pearson velocity correlation (0.35) + dwell zone overlap (0.25)</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">SGI v1.0 · proximity threshold: 150px · Pearson velocity correlation · group detection: BFS connected-component analysis</div>', unsafe_allow_html=True)

#     if "sgi_engine" not in st.session_state:
#         st.session_state.sgi_engine = SocialGraphEngine(proximity_px=150)
#     engine = st.session_state.sgi_engine

#     st.markdown("### Simulate Person Movement")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         obs_id = st.number_input("Person ID", min_value=1, value=1)
#         pos_x  = st.number_input("Start Position X", min_value=0, max_value=1920, value=320)
#     with c2:
#         pos_y  = st.number_input("Start Position Y", min_value=0, max_value=1080, value=240)
#         n_obs  = st.number_input("Frames to simulate", min_value=1, max_value=200, value=50)
#     with c3:
#         move_x = st.number_input("Movement X per frame", min_value=-10, max_value=10, value=2)
#         move_y = st.number_input("Movement Y per frame", min_value=-10, max_value=10, value=0)

#     if st.button("SIMULATE MOVEMENT"):
#         for i in range(int(n_obs)):
#             x = int(pos_x + i * move_x + np.random.randn() * 2)
#             y = int(pos_y + i * move_y + np.random.randn() * 2)
#             engine.observe(obs_id, (max(0, x), max(0, y)))
#         engine._update_links()
#         st.success(f"Simulated {n_obs} frames for Person {obs_id}")

#     if st.button("DETECT GROUPS", type="primary"):
#         st.session_state.sgi_result = {
#             "groups":  engine.detect_groups(),
#             "links":   engine.get_all_links(),
#             "summary": engine.summary(),
#         }

#     if "sgi_result" in st.session_state:
#         res     = st.session_state.sgi_result
#         groups  = res["groups"]
#         links   = res["links"]
#         summary = res["summary"]

#         s1, s2, s3, s4 = st.columns(4)
#         s1.metric("PERSONS TRACKED",  summary["persons_tracked"])
#         s2.metric("ACTIVE LINKS",     summary["active_links"])
#         s3.metric("GROUPS DETECTED",  summary["groups_detected"])
#         s4.metric("TOTAL ALERTS",     summary["total_alerts"])

#         if groups:
#             st.markdown('<div class="section-hdr">Detected Groups</div>', unsafe_allow_html=True)
#             for g in groups:
#                 color = "#ef4444" if g.alert else "#00b4ff"
#                 st.markdown(f"""
#                 <div style="padding:1rem 1.5rem; margin:0.5rem 0; background:rgba(0,4,12,0.92); border:1px solid {color}; border-radius:8px;">
#                     <div style="font-family:'Rajdhani',sans-serif; font-size:0.82rem; font-weight:700; color:{color}; letter-spacing:0.18em; margin-bottom:0.4rem;">
#                         GROUP {g.group_id} · {g.formation.upper()} · Cohesion: {g.cohesion:.3f}
#                     </div>
#                     <div style="font-family:'IBM Plex Mono',monospace; font-size:0.66rem; color:#5a8090; line-height:1.5;">
#                         Members: {g.members}{"  ·  ALERT: " + g.alert_reason if g.alert else ""}
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
#         else:
#             st.info("No groups detected yet. Simulate matching movement patterns for multiple persons, then detect groups.")

#         if links:
#             st.markdown('<div class="section-hdr">Social Link Graph</div>', unsafe_allow_html=True)
#             st.markdown('<div class="section-sub">All pairwise behavioral associations detected between tracked persons</div>', unsafe_allow_html=True)
#             import pandas as pd
#             df = pd.DataFrame([{
#                 "Persons":         f"{l.person_a} -- {l.person_b}",
#                 "Strength":        l.strength,
#                 "Type":            l.link_type,
#                 "Frames Observed": l.frame_count,
#                 "Proximity (px)":  l.evidence.get("proximity_px", 0),
#                 "Velocity Corr":   l.evidence.get("velocity_corr", 0),
#             } for l in links])
#             st.dataframe(df, use_container_width=True)

#     st.markdown("<hr>")
#     if st.button("RESET ENGINE"):
#         engine.reset_all()
#         if "sgi_result" in st.session_state:
#             del st.session_state.sgi_result
#         st.success("Social graph engine reset.")


# def pev_page():
#     render_session_bar()
#     back_button()
#     from core.predictive_exit import PredictiveExitEngine

#     st.markdown('<div class="section-hdr red">Predictive Exit Vector</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Frame boundary exit prediction · 3–5 seconds ahead · camera handoff intelligence · PEV v1.0</div>', unsafe_allow_html=True)
#     st.markdown("""
#     <div class='info-box'>
#         <strong>Research contribution:</strong> PEV v1.0 tracks each person's position history and
#         computes a smoothed velocity vector using a sliding window over recent frames.
#         Linear trajectory extrapolation then determines which frame boundary — LEFT, RIGHT, TOP, or BOTTOM —
#         the person will cross and how many seconds remain before exit.
#         Prediction fires <strong>3 to 5 seconds before actual exit</strong>, enabling downstream camera handoff
#         in multi-camera surveillance grids. Confidence is composed from three factors: velocity stability
#         (is the direction consistent?), boundary proximity (how close are they?), and history depth
#         (how many frames observed?). No equivalent open-source implementation exists for
#         real-time multi-person exit prediction in surveillance systems.
#         <br><br>
#         <strong>Algorithm:</strong> position history → sliding-window velocity smoothing →
#         linear trajectory extrapolation → boundary intersection detection →
#         confidence scoring → ExitPrediction output
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown('<div class="terminal">PEV v1.0 · velocity smoothing window: 5 frames · prediction horizon: 4s · confidence: stability × proximity × depth · IUB AI Research Lab</div>', unsafe_allow_html=True)

#     st.markdown("### Simulate Exit Prediction")
#     st.markdown('<div class="section-sub">Manually feed person positions to test the prediction engine in real time</div>', unsafe_allow_html=True)

#     if "pev_engine" not in st.session_state:
#         st.session_state.pev_engine = PredictiveExitEngine(frame_width=640, frame_height=480, fps=25)
#     engine = st.session_state.pev_engine

#     c1, c2, c3 = st.columns(3)
#     with c1:
#         sim_person_id = st.number_input("Person ID", min_value=1, value=1, key="pev_pid")
#         bbox_x1       = st.number_input("BBox X1", min_value=0, max_value=620, value=400, key="pev_x1")
#     with c2:
#         bbox_y1       = st.number_input("BBox Y1", min_value=0, max_value=460, value=200, key="pev_y1")
#         bbox_w        = st.number_input("BBox Width", min_value=20, max_value=200, value=50, key="pev_w")
#     with c3:
#         bbox_h        = st.number_input("BBox Height", min_value=20, max_value=300, value=100, key="pev_h")
#         n_auto_frames = st.number_input("Auto-simulate frames", min_value=1, max_value=50, value=20, key="pev_nf")

#     col_a, col_b, col_c = st.columns(3)

#     with col_a:
#         if st.button("FEED SINGLE FRAME"):
#             dets = [{"person_id": sim_person_id, "bbox": [bbox_x1, bbox_y1, bbox_x1 + bbox_w, bbox_y1 + bbox_h]}]
#             preds = engine.update(dets)
#             st.session_state.pev_result = preds

#     with col_b:
#         if st.button("AUTO-SIMULATE →RIGHT", type="primary"):
#             preds_last = []
#             for i in range(int(n_auto_frames)):
#                 x1 = bbox_x1 + i * 12
#                 dets = [{"person_id": sim_person_id, "bbox": [x1, bbox_y1, x1 + bbox_w, bbox_y1 + bbox_h]}]
#                 preds_last = engine.update(dets)
#             st.session_state.pev_result = preds_last
#             st.success(f"Simulated {n_auto_frames} frames — person moving RIGHT")

#     with col_c:
#         if st.button("RESET ENGINE"):
#             engine.reset()
#             if "pev_result" in st.session_state:
#                 del st.session_state.pev_result
#             st.success("PEV engine reset.")

#     if "pev_result" in st.session_state:
#         preds = st.session_state.pev_result
#         st.markdown("<hr>")
#         st.markdown('<div class="section-hdr">Live Predictions</div>', unsafe_allow_html=True)

#         if not preds or all(p.exit_side == "NONE" for p in preds):
#             st.info("No exit predicted yet — feed more frames or simulate movement toward a boundary.")
#         else:
#             for pred in preds:
#                 if pred.exit_side == "NONE":
#                     continue

#                 side_colors = {
#                     "LEFT":   "#00b4ff",
#                     "RIGHT":  "#00ff88",
#                     "TOP":    "#f0b429",
#                     "BOTTOM": "#ff3355",
#                 }
#                 color  = side_colors.get(pred.exit_side, "#ffffff")
#                 alert_html = (
#                     '<div style="color:#ff3355;font-weight:700;font-size:0.75rem;'
#                     'letter-spacing:0.2em;margin-top:0.5rem;">⚠ EXIT IMMINENT</div>'
#                     if pred.alert else ""
#                 )

#                 st.markdown(f"""
#                 <div style="padding:1.8rem 2rem; margin:0.75rem 0;
#                     background:rgba(0,4,12,0.97); border:2px solid {color};
#                     border-radius:10px; box-shadow: 0 0 40px {color}18;
#                     font-family:'IBM Plex Mono',monospace;">
#                     <div style="font-size:0.55rem; color:#2a4060; letter-spacing:0.4em;
#                         margin-bottom:0.7rem; text-transform:uppercase;">
#                         Predictive Exit Vector · Person {pred.person_id}
#                     </div>
#                     <div style="display:flex; align-items:center; gap:2rem; flex-wrap:wrap;">
#                         <div>
#                             <div style="font-size:2.5rem; font-weight:900; color:{color};
#                                 font-family:'Exo 2',sans-serif; line-height:1;">
#                                 {pred.exit_side}
#                             </div>
#                             <div style="font-size:0.6rem; color:#3a6080; margin-top:0.2rem;
#                                 letter-spacing:0.15em;">EXIT SIDE</div>
#                         </div>
#                         <div>
#                             <div style="font-size:2.5rem; font-weight:900; color:#fff;
#                                 font-family:'Exo 2',sans-serif; line-height:1;">
#                                 {pred.seconds_to_exit:.2f}s
#                             </div>
#                             <div style="font-size:0.6rem; color:#3a6080; margin-top:0.2rem;
#                                 letter-spacing:0.15em;">TIME TO EXIT</div>
#                         </div>
#                         <div>
#                             <div style="font-size:2.5rem; font-weight:900; color:#00fff0;
#                                 font-family:'Exo 2',sans-serif; line-height:1;">
#                                 {pred.confidence:.3f}
#                             </div>
#                             <div style="font-size:0.6rem; color:#3a6080; margin-top:0.2rem;
#                                 letter-spacing:0.15em;">CONFIDENCE</div>
#                         </div>
#                         <div>
#                             <div style="font-size:1.1rem; font-weight:700; color:#7ab3d4;
#                                 font-family:'Rajdhani',sans-serif; line-height:1.3;">
#                                 ({pred.predicted_exit_point[0]}, {pred.predicted_exit_point[1]})
#                             </div>
#                             <div style="font-size:0.6rem; color:#3a6080; margin-top:0.2rem;
#                                 letter-spacing:0.15em;">EXIT POINT (px)</div>
#                         </div>
#                     </div>
#                     <div style="margin-top:0.85rem; font-size:0.62rem; color:#2a4060;">
#                         Velocity: vx={pred.current_velocity[0]:+.2f} · vy={pred.current_velocity[1]:+.2f} px/frame
#                     </div>
#                     {alert_html}
#                 </div>
#                 """, unsafe_allow_html=True)

#         st.markdown("<hr>")
#         st.markdown('<div class="section-hdr">Engine Status</div>', unsafe_allow_html=True)
#         s1, s2 = st.columns(2)
#         s1.metric("ACTIVE TRACKS", len(engine.tracks))
#         s2.metric("FRAMES PROCESSED", engine.frame_counter)

#         if engine.tracks:
#             st.markdown('<div class="section-sub">Tracked persons — history depth per ID</div>', unsafe_allow_html=True)
#             import pandas as pd
#             rows = [{"Person ID": pid, "History Frames": len(t.positions)} for pid, t in engine.tracks.items()]
#             st.dataframe(pd.DataFrame(rows), use_container_width=True)


# def report_page():
#     render_session_bar()
#     from core.reporter import generate_report
#     back_button()
#     st.markdown('<div class="section-hdr">Intelligence Report</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Classified PDF · session data · threat log · subject records · one-click download</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How it works:</strong> Fill in session data — total persons detected, loitering alerts, per-subject behavioral records with emotion and dwell time, and any weapon detections from the session. Click Generate and PhantomEye produces a classified PDF using fpdf2. Dark background with green terminal-style text. Weapon threat sections highlighted in red. CLASSIFIED header on the first page. The file is immediately available for download — nothing is stored server-side at any point.</div>""", unsafe_allow_html=True)
#     st.markdown('<div class="terminal">fpdf2 · dark theme · CLASSIFIED header · weapon threat sections in red · immediate download · zero server-side storage</div>', unsafe_allow_html=True)

#     st.markdown("### Session Data")
#     col1, col2 = st.columns(2)
#     with col1:
#         session_id       = st.text_input("Session ID",           value=st.session_state.get("session_id", "PE-SESSION-001"))
#         total_persons    = st.number_input("Total Persons",      min_value=0, value=5)
#         duration         = st.number_input("Duration (seconds)", min_value=0, value=300)
#     with col2:
#         loitering_alerts = st.number_input("Loitering Alerts",   min_value=0, value=1)
#         nl_query         = st.text_input("NL Query (optional)",  value="")
#         nl_result        = st.text_input("NL Result (optional)", value="")

#     st.markdown("### Detected Subjects")
#     num_subjects = st.slider("Number of subjects", 1, 10, 3)
#     detections = []
#     for i in range(num_subjects):
#         c1, c2, c3, c4, c5, _ = st.columns(6)
#         detections.append({
#             "id":            i + 1,
#             "emotion":       c1.selectbox(f"Emotion {i+1}", ["neutral","angry","happy","sad","fear","surprise"], key=f"em_{i}"),
#             "gender":        c2.selectbox(f"Gender {i+1}",  ["Man","Woman"], key=f"gen_{i}"),
#             "age":           c3.number_input(f"Age {i+1}",  10, 80, 25, key=f"age_{i}"),
#             "dwell_seconds": c4.number_input(f"Dwell {i+1}", 0, 600, 60, key=f"dw_{i}"),
#             "loitering":     c5.checkbox(f"Loiter {i+1}",   key=f"lo_{i}"),
#         })

#     st.markdown("### Weapon Detections")
#     has_weapon = st.checkbox("Weapon detected in session?")
#     weapon_detections = []
#     if has_weapon:
#         wc1, wc2 = st.columns(2)
#         weapon_class = wc1.selectbox("Weapon Class", ["Handgun","Knife","Shotgun","SMG","Automatic Rifle","Sniper","Sword"])
#         weapon_conf  = wc2.slider("Confidence", 0.3, 1.0, 0.85)
#         weapon_detections.append({"class_name": weapon_class, "confidence": weapon_conf})

#     st.markdown("<hr>")
#     if st.button("GENERATE PDF REPORT", type="primary"):
#         data = {
#             "session_id":        session_id,
#             "total_persons":     total_persons,
#             "duration_seconds":  duration,
#             "loitering_alerts":  loitering_alerts,
#             "weapon_detections": weapon_detections,
#             "detections":        detections,
#             "heatmap_img":       None,
#             "frame_sample":      None,
#             "nl_query":          nl_query,
#             "nl_result":         nl_result,
#         }
#         with st.spinner("Generating classified report..."):
#             path = generate_report(data)
#         with open(path, "rb") as f:
#             pdf_bytes = f.read()
#         st.success("Report generated.")
#         st.download_button(
#             label="Download PDF Report",
#             data=pdf_bytes,
#             file_name=f"phantomeye_report_{session_id}.pdf",
#             mime="application/pdf"
#         )


# def intel_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">System Intelligence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Module registry · model benchmarks · deployment info · novel contributions</div>', unsafe_allow_html=True)

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("SYSTEM",  "PhantomEye")
#     c2.metric("VERSION", "v3.4.0")
#     c3.metric("STATUS",  "ONLINE")
#     c4.metric("MODULES", "13 ACTIVE")

#     st.markdown("<br>", unsafe_allow_html=True)
#     modules_info = [
#         ("DETECTION",         "YOLOv8-nano",   "yolov8n.pt · class 0 · confidence 0.4+ · CPU only"),
#         ("ANALYTICS",         "ByteTrack",     "IOU matching · NumPy heatmap · dwell time tracking · loitering threshold: 60s"),
#         ("OSINT",             "LBPH Face",     "LBPH embedding · cosine gallery search · score 0–100 · LOW/MEDIUM/HIGH risk"),
#         ("EMOTION",           "DeepFace + TF", "7 emotion classes · age + gender · OpenCV detector · 15% min face size filter"),
#         ("NL QUERY",          "Groq LLaMA 3",  "llama-3.1-8b-instant · English + Roman Urdu · JSON structured filter extraction"),
#         ("WEAPON",            "YOLOv8 Custom", "9 classes · mAP50 53.2% · Handgun 89.5% · Shotgun 96.3% · SMG 98.6%"),
#         ("THREAT MOMENTUM",   "TMS v1.0",      "Novel · 6 behavioral signals · compound amplifier · 45s decay · 5 threat levels"),
#         ("BEHAVIORAL DNA",    "BDF v1.0",      "Novel · 5 behavioral components · cosine similarity · 82% match threshold"),
#         ("SOCIAL GRAPH",      "SGI v1.0",      "Novel · proximity + velocity sync + dwell overlap · BFS group detection"),
#         ("PREDICTIVE EXIT",   "PEV v1.0",      "Novel · velocity smoothing · linear trajectory extrapolation · boundary prediction 3–5s ahead · camera handoff"),
#         ("REPORT",            "fpdf2",         "Classified PDF · dark theme · CLASSIFIED header · threat sections in red"),
#         ("API",               "FastAPI",       "OAS 3.1 · CORS enabled · uvicorn · modular route handlers"),
#         ("USER GUIDE",        "Interactive",   "5-step onboarding · 12 module walkthroughs · 4 novel algo deep dives · API reference · FAQ · re-runnable tour"),
#     ]
#     for name, tech, desc in modules_info:
#         with st.expander(f"{name}  ·  {tech}  ·  ACTIVE"):
#             st.markdown(f'<div class="terminal">{desc}</div>', unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.json({
#         "author":              "Abu-Sameer-66",
#         "github":              "https://github.com/Abu-Sameer-66/PhantomEye",
#         "huggingface":         "https://abu-sameer-66-phantomeye.hf.space",
#         "stack":               ["Python 3.10", "YOLOv8", "DeepFace", "ByteTrack", "FastAPI", "Streamlit", "Groq", "fpdf2"],
#         "novel_contributions": [
#             "Threat Momentum Score (TMS v1.0) — temporal compound threat accumulation",
#             "Behavioral DNA Fingerprint (BDF v1.0) — camera-agnostic behavioral re-ID",
#             "Social Graph Intelligence (SGI v1.0) — implicit group detection from movement",
#             "Predictive Exit Vector (PEV v1.0) — 3-5s ahead frame boundary exit prediction",
#         ],
#         "paper_status":        "in progress",
#         "status":              "online",
#         "access":              "open",
#         "user_guide":          "active — 5-step onboarding + 13 module walkthroughs",
#     })


# def welcome_flow():
#     """First-visit onboarding — 5 steps."""
#     if "welcome_step" not in st.session_state:
#         st.session_state.welcome_step = 1
#     step = st.session_state.welcome_step
#     progress = (step - 1) / 4
#     st.markdown(f"""
#     <div style="position:fixed;top:0;left:0;right:0;height:2px;z-index:9999;
#          background:linear-gradient(90deg,#00b4ff {int(progress*100)}%,rgba(0,180,255,0.1) {int(progress*100)}%);"></div>
#     """, unsafe_allow_html=True)
#     render_session_bar()
#     steps_html = ""
#     for i in range(1, 6):
#         if i < step:   color, border, txt = "#00ff88","#00ff88","#020408"
#         elif i == step: color, border, txt = "transparent","#00b4ff","#fff"
#         else:           color, border, txt = "transparent","#1a3a5c","#1a3a5c"
#         steps_html += f'<div style="width:28px;height:28px;border-radius:50%;border:2px solid {border};background:{color};display:flex;align-items:center;justify-content:center;font-family:\'IBM Plex Mono\',monospace;font-size:11px;font-weight:700;color:{txt};">{"✓" if i < step else i}</div>'
#         if i < 5: steps_html += f'<div style="flex:1;height:1px;background:{"#00ff88" if i < step else "#1a3a5c"};margin:0 4px;"></div>'
#     st.markdown(f'<div style="display:flex;align-items:center;gap:0;max-width:360px;margin:2rem auto 0;">{steps_html}</div>', unsafe_allow_html=True)

#     if step == 1:
#         st.markdown("""
#         <div style="text-align:center;padding:4rem 2rem 2rem;">
#             <div style="font-size:5rem;margin-bottom:1.5rem;filter:drop-shadow(0 0 40px rgba(0,180,255,0.8));">👁</div>
#             <div style="font-family:'Exo 2',sans-serif;font-size:clamp(2.5rem,6vw,5rem);font-weight:900;letter-spacing:0.1em;
#                  background:linear-gradient(135deg,#fff 0%,#60c8ff 50%,#00fff0 100%);
#                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:0.75rem;">
#                 PHANTOMEYE</div>
#             <div style="font-family:'Rajdhani',sans-serif;font-size:0.9rem;letter-spacing:0.4em;color:#3a6080;text-transform:uppercase;margin-bottom:0.5rem;">
#                 AI-Powered Surveillance Intelligence System</div>
#             <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#00ff88;letter-spacing:0.25em;margin-bottom:3rem;">
#                 ● BUILD v3.4 · 13 MODULES · 4 NOVEL ALGORITHMS · OPEN ACCESS</div>
#             <div style="max-width:600px;margin:0 auto 3rem;background:rgba(0,8,20,0.8);border:1px solid rgba(0,180,255,0.1);
#                  border-radius:12px;padding:1.5rem 2rem;font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:#7ab3d4;line-height:1.8;text-align:left;">
#                 PhantomEye is a production-grade AI surveillance system built entirely on CPU — no GPU required.
#                 Upload any image or video and get instant intelligent analysis across 13 specialized modules.
#                 Four original research algorithms <span style="color:#00b4ff;font-weight:700;">TMS · BDF · SGI · PEV</span>
#                 have no open-source equivalents anywhere.</div>
#         </div>""", unsafe_allow_html=True)

#     elif step == 2:
#         st.markdown("""
#         <div style="text-align:center;padding:2rem 1rem 1rem;">
#             <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;letter-spacing:0.2em;color:#00b4ff;margin-bottom:0.5rem;">13 INTELLIGENCE MODULES</div>
#             <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#3a6080;letter-spacing:0.2em;margin-bottom:2rem;">CORE INTELLIGENCE · NOVEL RESEARCH · UTILITY</div>
#         </div>""", unsafe_allow_html=True)
#         modules_overview = [
#             ("🎯","Person Detection","YOLOv8-nano · CPU · Conf 0.4",False),
#             ("🔥","Behavioral Analytics","ByteTrack · Heatmap · Dwell",False),
#             ("🕵️","OSINT Audit","LBPH Face · Score 0–100",False),
#             ("🧠","Emotion Intelligence","DeepFace · 7 Classes",False),
#             ("💬","NL Query Engine","Groq LLaMA 3 · Roman Urdu",False),
#             ("⚠️","Weapon Detection","YOLOv8 Custom · 9 Classes",False),
#             ("📊","Threat Momentum","Novel · TMS v1.0",True),
#             ("🧬","Behavioral DNA","Novel · BDF v1.0",True),
#             ("🕸️","Social Graph","Novel · SGI v1.0",True),
#             ("🚀","Predictive Exit","Novel · PEV v1.0",True),
#             ("📄","Intel Report","fpdf2 · Classified PDF",False),
#             ("⚡","System Intel","Live Status · API Ref",False),
#             ("📖","User Guide","Walkthroughs · FAQ",False),
#         ]
#         rows = [modules_overview[i:i+4] for i in range(0, len(modules_overview), 4)]
#         for row in rows:
#             cols = st.columns(len(row))
#             for idx, (icon, name, desc, novel) in enumerate(row):
#                 bc = "rgba(255,51,85,0.3)" if novel else "rgba(0,180,255,0.15)"
#                 nc = "#ff3355" if novel else "#00b4ff"
#                 with cols[idx]:
#                     st.markdown(f"""
#                     <div style="background:rgba(0,8,20,0.8);border:1px solid {bc};border-radius:10px;
#                          padding:1rem;text-align:center;margin-bottom:0.75rem;">
#                         <div style="font-size:1.4rem;margin-bottom:0.3rem;">{icon}</div>
#                         <div style="font-family:'Rajdhani',sans-serif;font-size:0.7rem;font-weight:700;
#                              letter-spacing:0.08em;color:{nc};margin-bottom:0.25rem;">{name}</div>
#                         <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;color:#3a6080;line-height:1.4;">{desc}</div>
#                     </div>""", unsafe_allow_html=True)

#     elif step == 3:
#         st.markdown("""
#         <div style="text-align:center;padding:2rem 1rem 1rem;">
#             <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;letter-spacing:0.2em;color:#ff3355;margin-bottom:0.5rem;">4 NOVEL ALGORITHMS</div>
#             <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#3a6080;letter-spacing:0.2em;margin-bottom:2rem;">NO OPEN-SOURCE EQUIVALENTS · IUB AI RESEARCH LAB · 2025</div>
#         </div>""", unsafe_allow_html=True)
#         algos = [
#             ("📊","TMS v1.0","Threat Momentum Score",
#              "Like compound interest for threat — each bad signal adds more than the last.",
#              "TMS(t) = TMS(t-1) × decay + Σ(signal × weight × amplifier)","CLEAR → LOW → MEDIUM → HIGH → CRITICAL"),
#             ("🧬","BDF v1.0","Behavioral DNA Fingerprint",
#              "Identifies same person across cameras using HOW they move — no face needed.",
#              "BDF = [gait(10) + velocity(10) + spatial(64) + social_dist(1) + dwell(64)]","cosine similarity > 0.82 → SAME PERSON"),
#             ("🕸️","SGI v1.0","Social Graph Intelligence",
#              "Detects who is with whom purely from movement — no prior info needed.",
#              "Link = proximity(0.40) + velocity_corr(0.35) + dwell_overlap(0.25)","BFS group detection → coordinated movement alert"),
#             ("🚀","PEV v1.0","Predictive Exit Vector",
#              "Predicts which exit and how many seconds — 3 to 5 seconds before it happens.",
#              "trajectory → boundary_intersect() → confidence = stability × proximity × depth","LEFT/RIGHT/TOP/BOTTOM · ETA seconds · camera handoff"),
#         ]
#         for icon, tag, name, simple, formula, output in algos:
#             st.markdown(f"""
#             <div style="background:rgba(0,4,12,0.95);border:1px solid rgba(255,51,85,0.2);border-left:3px solid #ff3355;
#                  border-radius:8px;padding:1.25rem 1.5rem;margin-bottom:1rem;">
#                 <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.6rem;">
#                     <span style="font-size:1.4rem;">{icon}</span>
#                     <span style="font-family:'Rajdhani',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;
#                          color:#ff3355;background:rgba(255,51,85,0.1);border:1px solid rgba(255,51,85,0.3);
#                          padding:2px 8px;border-radius:3px;">{tag}</span>
#                     <span style="font-family:'Exo 2',sans-serif;font-size:1rem;font-weight:700;color:#e8f4ff;">{name}</span>
#                 </div>
#                 <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#00b4ff;margin-bottom:0.4rem;">"{simple}"</div>
#                 <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#3a6080;background:rgba(0,0,0,0.3);
#                      border-radius:4px;padding:5px 10px;margin-bottom:0.35rem;">{formula}</div>
#                 <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#00ff88;">→ {output}</div>
#             </div>""", unsafe_allow_html=True)

#     elif step == 4:
#         st.markdown("""
#         <div style="text-align:center;padding:2rem 1rem 1rem;">
#             <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;letter-spacing:0.2em;color:#00ff88;margin-bottom:0.5rem;">QUICK START — 5 MINUTES</div>
#             <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#3a6080;letter-spacing:0.2em;margin-bottom:2rem;">FOLLOW THESE STEPS TO RUN YOUR FIRST ANALYSIS</div>
#         </div>""", unsafe_allow_html=True)
#         steps_qs = [
#             ("01","Person Detection","Upload any JPG/PNG image → instant bounding boxes + confidence per person. < 2 seconds on CPU."),
#             ("02","Behavioral Analytics","Upload any MP4 video → click RUN → heatmap + dwell times + loitering alerts. 15s video ≈ 30s processing."),
#             ("03","Weapon Detection","Upload any image → scans 9 weapon classes → THREAT DETECTED or scene clear. < 1 second."),
#             ("04","NL Query — Roman Urdu","Type: 'log jo loiter kar rahy thy aur angry thy' → extracted filters + matched subjects."),
#             ("05","Try TMS → Trigger CRITICAL","Threat Score → Person 1 → check Loitering + emotion angry + Restricted Zone → click UPDATE 4 times → CRITICAL fires."),
#         ]
#         for num, title, desc in steps_qs:
#             st.markdown(f"""
#             <div style="background:rgba(0,8,20,0.8);border:1px solid rgba(0,180,255,0.1);border-radius:10px;
#                  padding:1rem 1.4rem;margin-bottom:0.75rem;display:flex;gap:1rem;">
#                 <div style="font-family:'Exo 2',sans-serif;font-size:2rem;font-weight:900;color:rgba(0,180,255,0.2);
#                      line-height:1;flex-shrink:0;min-width:44px;">{num}</div>
#                 <div>
#                     <div style="font-family:'Rajdhani',sans-serif;font-size:0.82rem;font-weight:700;
#                          color:#00b4ff;letter-spacing:0.1em;margin-bottom:0.3rem;">{title}</div>
#                     <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#7ab3d4;line-height:1.6;">{desc}</div>
#                 </div>
#             </div>""", unsafe_allow_html=True)

#     elif step == 5:
#         st.markdown("""
#         <div style="text-align:center;padding:3rem 2rem 2rem;">
#             <div style="font-size:4rem;margin-bottom:1.5rem;filter:drop-shadow(0 0 30px rgba(0,255,136,0.8));">✓</div>
#             <div style="font-family:'Exo 2',sans-serif;font-size:2rem;font-weight:900;letter-spacing:0.2em;color:#00ff88;margin-bottom:0.75rem;">SYSTEM READY</div>
#             <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#7ab3d4;letter-spacing:0.15em;margin-bottom:3rem;">
#                 All 13 modules loaded · 4 novel algorithms active · CPU optimized · Open Access</div>
#             <div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;max-width:500px;margin:0 auto 2rem;">
#                 <div style="flex:1;min-width:110px;background:rgba(0,8,20,0.8);border:1px solid rgba(0,255,136,0.2);border-radius:8px;padding:1rem;text-align:center;">
#                     <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;color:#00ff88;">13</div>
#                     <div style="font-size:0.55rem;color:#3a6080;font-family:'IBM Plex Mono',monospace;letter-spacing:0.1em;">MODULES</div>
#                 </div>
#                 <div style="flex:1;min-width:110px;background:rgba(0,8,20,0.8);border:1px solid rgba(255,51,85,0.2);border-radius:8px;padding:1rem;text-align:center;">
#                     <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;color:#ff3355;">4</div>
#                     <div style="font-size:0.55rem;color:#3a6080;font-family:'IBM Plex Mono',monospace;letter-spacing:0.1em;">NOVEL ALGOS</div>
#                 </div>
#                 <div style="flex:1;min-width:110px;background:rgba(0,8,20,0.8);border:1px solid rgba(0,180,255,0.2);border-radius:8px;padding:1rem;text-align:center;">
#                     <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;color:#00b4ff;">CPU</div>
#                     <div style="font-size:0.55rem;color:#3a6080;font-family:'IBM Plex Mono',monospace;letter-spacing:0.1em;">NO GPU</div>
#                 </div>
#             </div>
#             <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#3a6080;">
#                 Access the <span style="color:#00b4ff;">GUIDE</span> module anytime from the navigation for walkthroughs, deep dives, and API reference.
#             </div>
#         </div>""", unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     col_l, col_m, col_r = st.columns([1, 2, 1])
#     with col_l:
#         if step > 1:
#             if st.button("← BACK", key="welcome_back"):
#                 st.session_state.welcome_step -= 1
#                 st.rerun()
#     with col_m:
#         labels = ["","WHAT IS PHANTOMEYE","13 MODULES","4 NOVEL ALGORITHMS","QUICK START","SYSTEM READY"]
#         st.markdown(f'<div style="text-align:center;font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;color:#3a6080;letter-spacing:0.15em;padding-top:0.65rem;">STEP {step} OF 5 · {labels[step]}</div>', unsafe_allow_html=True)
#     with col_r:
#         if step < 5:
#             if st.button("NEXT →", key="welcome_next"):
#                 st.session_state.welcome_step += 1
#                 st.rerun()
#         else:
#             if st.button("ENTER SYSTEM →", key="welcome_enter", type="primary"):
#                 st.session_state.page = "home"
#                 st.session_state.first_visit_done = True
#                 st.rerun()
#     if step < 5:
#         st.markdown("<br>", unsafe_allow_html=True)
#         cs = st.columns([3, 1])
#         with cs[1]:
#             if st.button("Skip Guide", key="welcome_skip"):
#                 st.session_state.page = "home"
#                 st.session_state.first_visit_done = True
#                 st.rerun()


# def zone_page():
#     render_session_bar()
#     back_button()
#     from core.zone_intelligence import ZoneIntelligenceEngine, ZoneType, AlertLevel

#     st.markdown('<div class="section-hdr red">Zone Intelligence Engine</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Novel · ZIE v1.0 · Restricted zones · Capacity limits · Breach alerts · TMS integration</div>', unsafe_allow_html=True)
#     st.markdown("""
#     <div class='info-box'>
#         <strong>Research contribution:</strong> ZIE v1.0 defines named surveillance zones
#         directly on the video frame. Four zone types:
#         <span style="color:#ff3355;font-weight:600;">RESTRICTED</span> (no entry — CRITICAL alert),
#         <span style="color:#f0b429;font-weight:600;">MONITORED</span> (entry logged),
#         <span style="color:#00fff0;font-weight:600;">CAPACITY LIMITED</span> (max occupancy enforced),
#         <span style="color:#00ff88;font-weight:600;">SAFE</span> (normal zone).
#         Every breach automatically feeds the <strong>proximity_violation</strong> signal into
#         TMS v1.0 — escalating the threat score of the offending person in real time.
#         Suspicious zone traversal sequences are also detected automatically.
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown('<div class="terminal">ZIE v1.0 · 4 zone types · CRITICAL breach alerts · capacity enforcement · TMS integration · suspicious sequence detection</div>', unsafe_allow_html=True)

#     if "zie_engine" not in st.session_state:
#         st.session_state.zie_engine = ZoneIntelligenceEngine()
#     engine = st.session_state.zie_engine

#     # ── Zone Definition ──────────────────────────────────
#     st.markdown("### Define Zones")
#     st.markdown('<div class="section-sub">Add zones by specifying name, type, and pixel coordinates on a 640×480 frame</div>', unsafe_allow_html=True)

#     c1, c2, c3 = st.columns(3)
#     with c1:
#         zone_name = st.text_input("Zone Name", value="Server Room", key="zie_name")
#         zone_type = st.selectbox("Zone Type", ["RESTRICTED", "MONITORED", "CAPACITY_LIMITED", "SAFE"], key="zie_type")
#     with c2:
#         zx1 = st.number_input("X1 (left)", min_value=0, max_value=639, value=400, key="zie_x1")
#         zy1 = st.number_input("Y1 (top)", min_value=0, max_value=479, value=100, key="zie_y1")
#     with c3:
#         zx2 = st.number_input("X2 (right)", min_value=1, max_value=640, value=600, key="zie_x2")
#         zy2 = st.number_input("Y2 (bottom)", min_value=1, max_value=480, value=350, key="zie_y2")
#         max_cap = st.number_input("Max Capacity", min_value=1, max_value=50, value=5, key="zie_cap")

#     col_add, col_clear = st.columns(2)
#     with col_add:
#         if st.button("ADD ZONE", type="primary", key="zie_add"):
#             zt = ZoneType(zone_type)
#             engine.add_zone(zone_name, zt, zx1, zy1, zx2, zy2, max_capacity=max_cap)
#             st.success(f"Zone '{zone_name}' added. Total zones: {len(engine.zones)}")
#     with col_clear:
#         if st.button("CLEAR ALL ZONES", key="zie_clear"):
#             engine.clear_zones()
#             st.success("All zones cleared.")

#     # ── Preset Scenarios ─────────────────────────────────
#     st.markdown("### Quick Presets")
#     p1, p2, p3 = st.columns(3)
#     with p1:
#         if st.button("🏦 Bank Scenario", key="preset_bank"):
#             engine.clear_zones()
#             engine.add_zone("Vault",        ZoneType.RESTRICTED,       420, 50,  620, 280)
#             engine.add_zone("Counter",      ZoneType.MONITORED,        50,  50,  400, 250)
#             engine.add_zone("Lobby",        ZoneType.CAPACITY_LIMITED, 50,  280, 640, 480, max_capacity=10)
#             engine.add_zone("Exit",         ZoneType.SAFE,             600, 280, 640, 480)
#             st.success("Bank scenario loaded — 4 zones defined.")
#     with p2:
#         if st.button("🏥 Hospital Scenario", key="preset_hospital"):
#             engine.clear_zones()
#             engine.add_zone("ICU",          ZoneType.RESTRICTED,       400, 50,  640, 300)
#             engine.add_zone("Ward",         ZoneType.MONITORED,        50,  50,  380, 300)
#             engine.add_zone("Waiting Area", ZoneType.CAPACITY_LIMITED, 50,  300, 640, 480, max_capacity=8)
#             engine.add_zone("Reception",    ZoneType.SAFE,             50,  300, 200, 480)
#             st.success("Hospital scenario loaded — 4 zones defined.")
#     with p3:
#         if st.button("🏢 Office Scenario", key="preset_office"):
#             engine.clear_zones()
#             engine.add_zone("Server Room",  ZoneType.RESTRICTED,       450, 50,  640, 250)
#             engine.add_zone("CEO Office",   ZoneType.MONITORED,        200, 50,  430, 250)
#             engine.add_zone("Open Floor",   ZoneType.CAPACITY_LIMITED, 50,  250, 640, 480, max_capacity=15)
#             engine.add_zone("Corridor",     ZoneType.SAFE,             50,  50,  180, 480)
#             st.success("Office scenario loaded — 4 zones defined.")

#     # ── Active Zones Display ─────────────────────────────
#     if engine.zones:
#         st.markdown("### Active Zones")
#         type_colors = {
#             "RESTRICTED":       "#ff3355",
#             "MONITORED":        "#f0b429",
#             "CAPACITY_LIMITED": "#00fff0",
#             "SAFE":             "#00ff88",
#         }
#         for zid, zone in engine.zones.items():
#             color = type_colors.get(zone.zone_type.value, "#00b4ff")
#             occ   = sum(1 for s in engine.person_states.values() if zid in s.current_zones)
#             st.markdown(f"""
#             <div style="background:rgba(0,8,20,0.8);border:1px solid {color}40;border-left:3px solid {color};
#                  border-radius:6px;padding:0.6rem 1rem;margin-bottom:0.5rem;
#                  font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
#                  display:flex;align-items:center;gap:1.5rem;">
#                 <span style="color:{color};font-weight:700;min-width:20px;">[{zid}]</span>
#                 <span style="color:#e8f4ff;font-weight:600;min-width:140px;">{zone.name}</span>
#                 <span style="color:{color};font-size:0.62rem;min-width:140px;">{zone.zone_type.value}</span>
#                 <span style="color:#3a6080;font-size:0.62rem;">BBox: ({zone.x1},{zone.y1}) → ({zone.x2},{zone.y2})</span>
#                 <span style="color:#00ff88;font-size:0.62rem;margin-left:auto;">Occupancy: {occ}</span>
#             </div>
#             """, unsafe_allow_html=True)

#     # ── Person Simulation ────────────────────────────────
#     st.markdown("### Simulate Person Movement")
#     st.markdown('<div class="section-sub">Feed person positions frame by frame to test zone detection</div>', unsafe_allow_html=True)

#     s1, s2, s3 = st.columns(3)
#     with s1:
#         sim_pid = st.number_input("Person ID", min_value=1, value=1, key="zie_pid")
#         sim_x1  = st.number_input("BBox X1", min_value=0, max_value=639, value=410, key="zie_sx1")
#     with s2:
#         sim_y1  = st.number_input("BBox Y1", min_value=0, max_value=479, value=110, key="zie_sy1")
#         sim_x2  = st.number_input("BBox X2", min_value=1, max_value=640, value=460, key="zie_sx2")
#     with s3:
#         sim_y2  = st.number_input("BBox Y2", min_value=1, max_value=480, value=210, key="zie_sy2")

#     if st.button("FEED FRAME", type="primary", key="zie_feed"):
#         if not engine.zones:
#             st.warning("Define at least one zone first — or use a preset above.")
#         else:
#             dets   = [{"person_id": sim_pid, "bbox": [sim_x1, sim_y1, sim_x2, sim_y2]}]
#             result = engine.update(dets)
#             st.session_state.zie_last_result = result

#     if "zie_last_result" in st.session_state:
#         result = st.session_state.zie_last_result
#         events = result.get("events", [])

#         if events:
#             st.markdown("### Events This Frame")
#             level_colors = {
#                 "CRITICAL": "#ff3355",
#                 "HIGH":     "#f0b429",
#                 "MEDIUM":   "#00b4ff",
#                 "LOW":      "#00ff88",
#                 "NONE":     "#3a6080",
#             }
#             for evt in events:
#                 color = level_colors.get(evt["alert_level"], "#3a6080")
#                 icon  = "🔴" if evt["alert_level"] == "CRITICAL" else \
#                         "🟠" if evt["alert_level"] == "HIGH"     else \
#                         "🟡" if evt["alert_level"] == "MEDIUM"   else \
#                         "🟢" if evt["alert_level"] == "LOW"      else "⚪"
#                 st.markdown(f"""
#                 <div style="background:rgba(0,4,12,0.95);border:1px solid {color}40;
#                      border-left:3px solid {color};border-radius:6px;
#                      padding:0.75rem 1rem;margin-bottom:0.5rem;
#                      font-family:'IBM Plex Mono',monospace;">
#                     <div style="display:flex;align-items:center;gap:10px;">
#                         <span style="font-size:1.1rem;">{icon}</span>
#                         <span style="color:{color};font-size:0.65rem;font-weight:700;
#                              min-width:80px;">{evt['alert_level']}</span>
#                         <span style="color:#7ab3d4;font-size:0.72rem;">{evt['message']}</span>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)

#             if result.get("tms_signals"):
#                 st.markdown(f'<div class="terminal" style="color:#ff3355;margin-top:0.5rem;">⚡ TMS SIGNAL FIRED: proximity_violation → person(s) {list(result["tms_signals"].keys())} threat score boosted</div>', unsafe_allow_html=True)
#         else:
#             st.info("No events this frame — person is not inside any defined zone.")

#     # ── Zone Summaries ───────────────────────────────────
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("### Zone Summaries")
#     summaries = engine.get_all_summaries()
#     if summaries:
#         import pandas as pd
#         rows = []
#         for zid, s in summaries.items():
#             rows.append({
#                 "Zone": s["zone_name"],
#                 "Type": s["zone_type"],
#                 "Entries": s["total_entries"],
#                 "Exits": s["total_exits"],
#                 "Occupancy": s["current_occupancy"],
#                 "Avg Dwell (s)": s["avg_dwell_sec"],
#                 "Breaches": s["total_breaches"],
#                 "Alert": s["alert_level"],
#             })
#         st.dataframe(pd.DataFrame(rows), use_container_width=True)
#     else:
#         st.info("No zone data yet. Add zones and feed frames.")

#     # ── Event Log ────────────────────────────────────────
#     st.markdown("### Recent Event Log")
#     recent = engine.get_recent_events(15)
#     if recent:
#         import pandas as pd
#         df = pd.DataFrame([{
#             "Event":  e["event_type"].upper(),
#             "Zone":   e["zone_name"],
#             "Person": e["person_id"],
#             "Alert":  e["alert_level"],
#             "Message": e["message"],
#         } for e in recent])
#         st.dataframe(df, use_container_width=True)

#     # ── Session Summary ──────────────────────────────────
#     st.markdown("<hr>", unsafe_allow_html=True)
#     summary = engine.session_summary()
#     s1, s2, s3, s4 = st.columns(4)
#     s1.metric("ZONES DEFINED",    summary["total_zones"])
#     s2.metric("PERSONS TRACKED",  summary["total_persons_seen"])
#     s3.metric("TOTAL ENTRIES",    summary["total_entries"])
#     s4.metric("TOTAL BREACHES",   summary["total_breaches"])

#     if st.button("RESET ENGINE", key="zie_reset"):
#         engine.reset()
#         for key in ["zie_last_result"]:
#             if key in st.session_state:
#                 del st.session_state[key]
#         st.success("ZIE engine reset.")


# def guide_page():
#     """13th Module — Complete User Guide & Documentation."""
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">PhantomEye User Guide</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Complete documentation · module walkthroughs · novel algorithm deep dives · API reference · FAQ</div>', unsafe_allow_html=True)
#     st.markdown("""<div class="info-box"><strong>How to use this guide:</strong> Your complete reference for PhantomEye v3.4. Use the tabs below to navigate Quick Start, individual module walkthroughs, novel algorithm deep dives with math, API reference, and FAQ. You can also re-run the 5-step onboarding tour anytime from Quick Start.</div>""", unsafe_allow_html=True)

#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "⚡ Quick Start",
#         "📦 All Modules",
#         "🔬 Novel Algorithms",
#         "🔌 API Reference",
#         "❓ FAQ"
#     ])

#     # ── TAB 1 — QUICK START ──────────────────────────────
#     with tab1:
#         st.markdown('<div class="section-hdr">Quick Start — 5 Minutes to First Analysis</div>', unsafe_allow_html=True)
#         c1, c2, c3 = st.columns(3)
#         c1.metric("Modules", "13")
#         c2.metric("Novel Algorithms", "4")
#         c3.metric("GPU Required", "None")
#         st.markdown("<br>", unsafe_allow_html=True)
#         qs = [
#             ("🎯","01","Person Detection","Go to Detection → Upload any JPG/PNG → instant bounding boxes + confidence per person.","Annotated image · person count · bbox coordinates. < 2 seconds."),
#             ("🔥","02","Behavioral Analytics","Go to Analytics → Upload any MP4 video → click RUN BEHAVIORAL ANALYSIS → wait for processing.","Heatmap · dwell times · loitering alerts if any person stayed > 60s."),
#             ("⚠️","03","Weapon Detection","Go to Weapon → Upload any image → auto-scans 9 weapon classes.","THREAT DETECTED with class + confidence, OR scene clear confirmation."),
#             ("💬","04","NL Query — Roman Urdu","Go to NL Query → type: 'log jo loiter kar rahy thy aur angry thy' → press Enter.","Extracted filters + matched subjects from sample dataset."),
#             ("📊","05","Trigger TMS CRITICAL","Threat Score → ID=1 → Loitering ✓ · emotion=angry · Restricted Zone ✓ → click UPDATE 4 times.","Score climbs CLEAR→CRITICAL. Red alert fires at HIGH+."),
#         ]
#         for icon, num, title, how, expect in qs:
#             with st.expander(f"{icon} Step {num} — {title}"):
#                 st.markdown(f'<div class="terminal">{how}</div>', unsafe_allow_html=True)
#                 st.markdown(f'<div class="terminal" style="color:#00ff88;margin-top:0.5rem;">✓ Expected: {expect}</div>', unsafe_allow_html=True)
#         st.markdown("<br>", unsafe_allow_html=True)
#         if st.button("↺  RE-RUN ONBOARDING TOUR", key="rerun_tour"):
#             st.session_state.welcome_step = 1
#             st.session_state.first_visit_done = False
#             st.session_state.page = "welcome"
#             st.rerun()

#     # ── TAB 2 — ALL MODULES ─────────────────────────────
#     with tab2:
#         st.markdown('<div class="section-hdr">All 13 Modules — Step-by-Step Walkthroughs</div>', unsafe_allow_html=True)
#         mods = [
#             ("🎯","Person Detection","YOLOv8-nano · CPU · Class 0",False,
#              "Detects every person in an uploaded image. Returns bounding boxes and confidence scores per person. 100% CPU — no GPU.",
#              ["Click DETECTION in navigation","Upload any JPG or PNG image","Wait < 2 seconds for inference","View annotated image with bounding boxes","Expand Detection Log for raw coordinates"],
#              "JPG · PNG · Any size","Annotated image · person count · confidence · bbox coords",
#              "Works best with clearly visible people. Min recommended size: 50×50px.",
#              "yolov8n.pt · CPU · class 0 only · confidence threshold: 0.4"),
#             ("🔥","Behavioral Analytics","ByteTrack · Heatmap · Dwell",False,
#              "Multi-object tracking with persistent IDs, behavioral heatmap, dwell time per person, loitering alerts.",
#              ["Click ANALYTICS","Upload MP4/AVI/MOV video","Review video metadata","Click RUN BEHAVIORAL ANALYSIS","Wait (15s video ≈ 30s processing)","View heatmap — red = highest activity","Check metrics: persons · avg/max dwell · alerts"],
#              "MP4 · AVI · MOV · up to 15s analyzed","Heatmap · dwell times · loitering alerts with IDs",
#              "15s video cap. Longer videos auto-truncated.",
#              "ByteTrack IOU · NumPy heatmap · loitering threshold: 60s"),
#             ("🕵️","OSINT Audit","LBPH Face · Gallery",False,
#              "Upload face photo → Privacy Exposure Score 0–100. Matched against gallery. Risk: LOW/MEDIUM/HIGH.",
#              ["Click OSINT","Upload clear face photo (JPG/PNG)","Click EXECUTE AUDIT","View score 0–100 and risk level","Expand Match Log for gallery matches"],
#              "Clear face photo · JPG/PNG","Exposure score · risk level · match log · visualization",
#              "Use front-facing clear photo. Add gallery faces via API: POST /osint/add-to-gallery.",
#              "OpenCV LBPH · cosine distance · no data retained"),
#             ("🧠","Emotion Intelligence","DeepFace · TF · 7 Classes",False,
#              "Multi-face emotion analysis — dominant emotion, age estimate, gender per face.",
#              ["Click EMOTION","Upload image with faces","Wait 10–30s first run (TF model loading)","View original vs annotated side by side","Check per-subject: emotion · age · gender"],
#              "JPG · PNG · any image with faces","Dominant emotion · age estimate · gender · annotated image",
#              "First run is slow (TF loading). Subsequent runs faster. Works best with front-facing lit faces.",
#              "DeepFace + TF · OpenCV detector · 7 classes · 15% min face size filter"),
#             ("💬","NL Query Engine","Groq LLaMA 3 · Roman Urdu",False,
#              "Type surveillance queries in English or Roman Urdu. LLaMA 3 extracts structured filters.",
#              ["Click NL QUERY","Type any query (English or Roman Urdu)","Press Enter","View extracted filters","Check matched subjects"],
#              "English or Roman Urdu text","Extracted filters · matched subjects",
#              "Try: 'show me angry men who were loitering' OR 'log jo loiter kar rahy thy aur jinka emotion angry tha'",
#              "llama-3.1-8b-instant via Groq · JSON output · apply_filters() engine"),
#             ("⚠️","Weapon Detection","YOLOv8 Custom · 9 Classes",False,
#              "Custom YOLOv8 on 714 weapon images. 9 classes. Immediate threat alert.",
#              ["Click WEAPON","Upload any image","Wait < 1 second","View original vs annotated","THREAT DETECTED alert OR scene clear"],
#              "JPG · PNG · any image","THREAT alert OR clear · weapon class · confidence · annotated",
#              "9 classes: Handgun · Knife · Shotgun · Sniper · AR · SMG · Sword · Bazooka · GL",
#              "weapon_detector.pt · mAP50 53.2% · Handgun 89.5% · Shotgun 96.3% · SMG 98.6%"),
#             ("📊","Threat Momentum Score","NOVEL · TMS v1.0",True,
#              "Compound interest model for threat. Continuous score with 5 levels. No binary yes/no.",
#              ["Click THREAT SCORE in Row 2","Enter Person ID","Set signals: dwell · emotion · loitering · restricted zone","Click UPDATE THREAT SCORE","Repeat to see compound accumulation","Try: all boxes + angry → CRITICAL"],
#              "Person ID · position · emotion · dwell · boolean flags","TMS score · threat level · signal breakdown · momentum",
#              "Amplifier effect: when TMS is HIGH, new signals contribute MORE. Update same person 5–6 times to see exponential growth.",
#              "TMS(t) = TMS(t-1) × 0.5^(Δt/45s) + Σ(signal × weight × (1 + TMS/200))"),
#             ("🧬","Behavioral DNA","NOVEL · BDF v1.0",True,
#              "Re-identifies person across cameras using behavioral signature only. No face needed.",
#              ["Click BEHAVIORAL DNA","Set Person ID + position","Simulate Observations (30+)","Click REGISTER TO GALLERY","Change position slightly (simulate re-entry)","Simulate more observations","Click MATCH AGAINST GALLERY → MATCH FOUND"],
#              "Person ID · position · social distance · observations","BDF vector · similarity % · match result · explanation",
#              "Needs 15+ observations for reliable fingerprint. More observations = higher confidence.",
#              "5 components: gait(10) + velocity(10) + spatial(64) + social_dist(1) + dwell(64) · cosine · threshold 82%"),
#             ("🕸️","Social Graph","NOVEL · SGI v1.0",True,
#              "Detects who is together from movement alone. No prior information needed.",
#              ["Click SOCIAL GRAPH","Person 1: X=100, moveX=3 → simulate 50 frames","Person 2: X=120, moveX=3 → simulate 50 frames (same direction)","Person 3: X=500, moveX=-2 → simulate 50 frames (opposite)","Click DETECT GROUPS","Expected: P1+P2 grouped, P3 separate"],
#              "Person ID · start position · movement vector · frames","Social links · groups · cohesion · coordinated alerts",
#              "Same direction + speed = linked. Bank robbery: 3 enter separately but converge → SGI flags before action.",
#              "Link = proximity(0.40) + pearson_corr(0.35) + dwell_overlap(0.25) · BFS group detection"),
#             ("🚀","Predictive Exit Vector","NOVEL · PEV v1.0",True,
#              "Predicts frame boundary exit 3–5s ahead. Camera handoff use case.",
#              ["Click PREDICTIVE EXIT","Person ID=1, BBox X1=400 (near right edge)","Click AUTO-SIMULATE →RIGHT (20 frames)","View: EXIT SIDE=RIGHT · ETA · Confidence","Watch ALERT fire when ETA < 2s + conf > 0.4"],
#              "Person ID · bbox position · simulation frames","Exit side · ETA seconds · confidence · alert · trajectory",
#              "Needs 6+ frames before prediction is reliable. Confidence builds as more frames fed.",
#              "Velocity smoothing: 5 frames · horizon: 4s · confidence = stability × proximity × depth"),
#             ("📄","Intel Report","fpdf2 · Classified PDF",False,
#              "Generate classified PDF report. Dark theme, CLASSIFIED header, threat sections in red.",
#              ["Click REPORT","Fill session metadata","Set subjects with behavioral data","Check weapon detected if applicable","Click GENERATE PDF REPORT","Download immediately"],
#              "Session data · subjects · weapon detections","Downloadable classified PDF · dark theme · red threat sections",
#              "PDF generated in-memory. Nothing stored server-side. Each PDF unique to your session.",
#              "fpdf2 · dark bg #020408 · green text #00ff88 · red threat #ff3355"),
#             ("⚡","System Intel","Live Status",False,
#              "Dashboard: all modules, tech stack, benchmarks, API endpoints, deployment info.",
#              ["Click SYSTEM","View all 13 module statuses","Expand any module for tech specs","View full tech stack JSON"],
#              "No input required","Module registry · benchmarks · tech stack · deployment info",
#              "Use to verify system health and get API endpoint URLs for integration.",
#              "v3.4.0 · HuggingFace Spaces Docker · FastAPI OAS 3.1 · Python 3.10"),
#             ("📖","User Guide","This Module",False,
#              "Complete interactive documentation. Quick Start, walkthroughs, algo deep dives, API ref, FAQ.",
#              ["Click GUIDE in Row 2","Use tabs to navigate sections","Re-run onboarding tour from Quick Start tab","Bookmark for reference"],
#              "No input required","Full documentation · walkthroughs · API reference · FAQ",
#              "This is a living document. Check back after each upgrade for updated walkthroughs.",
#              "Interactive Streamlit tabs · 5-step tour · re-runnable onboarding"),
#         ]
#         for icon, name, tag, novel, what, steps_list, inp, out, tip, tech in mods:
#             border = "rgba(255,51,85,0.3)" if novel else "rgba(0,180,255,0.1)"
#             lc = "#ff3355" if novel else "#00b4ff"
#             with st.expander(f"{icon} {name}  ·  {tag}"):
#                 if novel:
#                     st.markdown('<div style="display:inline-block;background:rgba(255,51,85,0.1);border:1px solid rgba(255,51,85,0.3);border-radius:3px;font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;color:#ff3355;padding:2px 8px;letter-spacing:0.15em;margin-bottom:0.75rem;">⬡ NOVEL RESEARCH ALGORITHM</div>', unsafe_allow_html=True)
#                 st.markdown(f'<div class="terminal">{what}</div>', unsafe_allow_html=True)
#                 st.markdown("<br>", unsafe_allow_html=True)
#                 st.markdown('<div class="section-hdr">Step-by-Step</div>', unsafe_allow_html=True)
#                 for idx, s in enumerate(steps_list, 1):
#                     st.markdown(f'<div style="display:flex;gap:10px;padding:5px 0;border-bottom:1px solid rgba(0,180,255,0.05);"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:{lc};min-width:24px;font-weight:700;">{idx:02d}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#7ab3d4;">{s}</span></div>', unsafe_allow_html=True)
#                 st.markdown("<br>", unsafe_allow_html=True)
#                 ca, cb = st.columns(2)
#                 ca.markdown(f'<div class="terminal">INPUT: {inp}</div>', unsafe_allow_html=True)
#                 cb.markdown(f'<div class="terminal">OUTPUT: {out}</div>', unsafe_allow_html=True)
#                 st.markdown(f'<div class="info-box" style="margin-top:0.75rem;"><strong>Tip:</strong> {tip}</div>', unsafe_allow_html=True)
#                 st.markdown(f'<div class="terminal" style="margin-top:0.5rem;">TECH: {tech}</div>', unsafe_allow_html=True)

#     # ── TAB 3 — NOVEL ALGORITHMS ─────────────────────────
#     with tab3:
#         st.markdown('<div class="section-hdr red">4 Novel Algorithms — Deep Dive</div>', unsafe_allow_html=True)
#         st.markdown('<div class="section-sub">Original research · IUB AI Research Lab · No open-source equivalents · IEEE Access target</div>', unsafe_allow_html=True)
#         algo_data = [
#             ("📊","TMS v1.0","Threat Momentum Score",
#              "Unlike binary threat detection systems that output yes/no, TMS treats threat as continuous momentum — like compound interest. Every signal adds to a running score that decays over time, but amplifies when already elevated.",
#              "TMS(t) = TMS(t-1) × decay_factor + Σ(signal × weight × amplifier)\n\namplifier    = 1 + (TMS / 200)\ndecay_factor = 0.5^(Δt / 45s)   ← 45 second half-life",
#              "6 Behavioral Signals",
#              [("loitering","0.28","Person stays in one area beyond threshold"),
#               ("stress_emotion","0.22","Angry, fear, or disgust detected"),
#               ("rapid_movement","0.18","Sudden velocity spike"),
#               ("proximity_violation","0.15","Too close to restricted zone"),
#               ("gaze_anomaly","0.10","Unusual gaze pattern"),
#               ("group_formation","0.07","Part of flagged group (SGI output)")],
#              "CLEAR (0–20) → LOW (20–50) → MEDIUM (50–100) → HIGH (100–180) → CRITICAL (180+)",
#              "Person escalates CLEAR → CRITICAL across 5 frames with compound signals ✓"),
#             ("🧬","BDF v1.0","Behavioral DNA Fingerprint",
#              "Traditional Re-ID requires face recognition which fails with masks, hats, or distance. BDF identifies people using HOW they move — a behavioral fingerprint unique to each individual that persists across cameras.",
#              "BDF_vector = [\n  gait_signature(10),      ← stride rhythm histogram\n  velocity_profile(10),    ← speed distribution\n  spatial_preference(8×8), ← normalized grid heatmap\n  social_distance_avg(1),  ← preferred distance from others\n  dwell_zone_signature(64) ← stopping location pattern\n]\n\nMatch: cosine(BDF_a, BDF_b) > 0.82 → SAME PERSON",
#              "5 Behavioral Components",
#              [("Gait Signature","10 features","Stride rhythm and cadence pattern"),
#               ("Velocity Profile","10 features","Speed distribution across movement"),
#               ("Spatial Preference","64 features","Where person tends to stand on grid"),
#               ("Social Distance","1 feature","Average preferred distance from others"),
#               ("Dwell Zones","64 features","Which locations person stops at")],
#              "< 0.70 = NO MATCH · 0.70–0.82 = UNCERTAIN · > 0.82 = MATCH",
#              "Person 3 matched Person 1 at 99.99% after re-entry with new tracking ID ✓"),
#             ("🕸️","SGI v1.0","Social Graph Intelligence",
#              "Three bank robbers can enter a building separately and behave normally individually — but SGI detects their association before any overt action by analyzing movement correlation. No face recognition or prior info needed.",
#              "Link_strength = proximity(0.40) + pearson_velocity_corr(0.35) + dwell_overlap(0.25)\n\nGroup detection: BFS connected-component analysis\nAlert: coordinated link_type in group ≥ 2 persons",
#              "3 Association Signals",
#              [("Proximity Score","0.40 weight","How often persons are within 150px of each other"),
#               ("Velocity Correlation","0.35 weight","Pearson correlation — do they accelerate together?"),
#               ("Dwell Zone Overlap","0.25 weight","Do they stop at the same locations?")],
#              "0.0–0.3 = STRANGERS · 0.3–0.6 = ACQUAINTANCES · 0.6+ = ASSOCIATED",
#              "Person 1↔2 strength 0.570 (associated), Person 1↔3 strength 0.309 (strangers) ✓"),
#             ("🚀","PEV v1.0","Predictive Exit Vector",
#              "Existing systems only alert WHEN someone leaves. PEV predicts WHERE they will exit and WHEN — 3 to 5 seconds before it happens. Enables camera handoff: Camera B activates before person leaves Camera A.",
#              "1. Position history → deque(maxlen=15 frames)\n2. Smoothed velocity: sliding window avg (5 frames)\n3. Trajectory: step forward vx/vy × max_frames\n4. Boundary hit: first frame where x≤0, x≥W, y≤0, y≥H\n5. Confidence = stability × proximity × depth\n   - stability: 1 - CoV(velocity magnitudes)\n   - proximity: distance to nearest boundary\n   - depth:     history_frames / 15",
#              "Confidence Components",
#              [("Velocity Stability","50% weight","Is direction consistent? Low variance = high confidence"),
#               ("Boundary Proximity","30% weight","How close is person to frame edge?"),
#               ("History Depth","20% weight","How many frames of data available?")],
#              "conf < 0.4 = LOW · 0.4–0.7 = MEDIUM · > 0.7 = HIGH · ALERT: ETA < 2s AND conf > 0.4",
#              "All 4 directions correct · stationary = NONE · 3-person simultaneous ✓"),
#         ]
#         for icon, tag, name, concept, formula, sig_title, sigs, levels, result in algo_data:
#             st.markdown(f"""
#             <div style="background:rgba(0,4,12,0.95);border:1px solid rgba(255,51,85,0.15);border-top:2px solid #ff3355;
#                  border-radius:10px;padding:1.5rem;margin-bottom:1.5rem;">
#                 <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
#                     <span style="font-size:1.8rem;">{icon}</span>
#                     <span style="font-family:'Rajdhani',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;
#                          color:#ff3355;background:rgba(255,51,85,0.1);border:1px solid rgba(255,51,85,0.3);
#                          padding:2px 8px;border-radius:3px;">{tag}</span>
#                     <span style="font-family:'Exo 2',sans-serif;font-size:1.2rem;font-weight:900;color:#e8f4ff;">{name}</span>
#                 </div>
#                 <div style="font-family:'IBM Plex Mono',monospace;font-size:0.73rem;color:#7ab3d4;line-height:1.8;">{concept}</div>
#             </div>""", unsafe_allow_html=True)
#             with st.expander(f"📐 Formula + {sig_title}"):
#                 st.code(formula, language="python")
#                 st.markdown(f'<div class="section-hdr">{sig_title}</div>', unsafe_allow_html=True)
#                 for sn, sw, sd in sigs:
#                     st.markdown(f'<div style="display:flex;gap:12px;padding:5px 0;border-bottom:1px solid rgba(0,180,255,0.05);"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#ff3355;min-width:140px;font-weight:700;">{sn}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#00b4ff;min-width:80px;">{sw}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#5a8090;">{sd}</span></div>', unsafe_allow_html=True)
#                 st.markdown(f'<div class="terminal" style="margin-top:0.75rem;">LEVELS: {levels}</div>', unsafe_allow_html=True)
#                 st.markdown(f'<div class="terminal" style="color:#00ff88;">RESULT: {result}</div>', unsafe_allow_html=True)

#     # ── TAB 4 — API REFERENCE ───────────────────────────
#     with tab4:
#         st.markdown('<div class="section-hdr">API Reference — FastAPI OAS 3.1</div>', unsafe_allow_html=True)
#         st.markdown('<div class="terminal">Base URL: https://abu-sameer-66-phantomeye.hf.space · Docs: /docs · OpenAPI: /openapi.json</div>', unsafe_allow_html=True)
#         st.markdown("<br>", unsafe_allow_html=True)
#         endpoints = [
#             ("GET",  "/",                           "Root",               "System info · version · module list"),
#             ("GET",  "/health",                     "Health Check",       "Status · gallery size · timestamp"),
#             ("POST", "/detect",                     "Person Detection",   "Upload image → bbox + confidence per person"),
#             ("POST", "/osint/audit",                "OSINT Audit",        "Upload face → exposure score + matches"),
#             ("POST", "/osint/add-to-gallery",       "Add to Gallery",     "Upload face + person_id → add to gallery"),
#             ("GET",  "/osint/gallery",              "Gallery List",       "All person IDs in OSINT gallery"),
#             ("POST", "/track/video",                "Video Tracking",     "Upload video → tracking summary"),
#             ("GET",  "/outputs",                    "List Outputs",       "All generated output files"),
#             ("POST", "/api/predictive-exit/update", "PEV Update",         "Feed frame detections → exit predictions"),
#             ("GET",  "/api/predictive-exit/status", "PEV Status",         "Active track count + engine info"),
#             ("POST", "/api/predictive-exit/reset",  "PEV Reset",          "Clear all tracked persons"),
#         ]
#         mc = {"GET":"#00ff88","POST":"#00b4ff","DELETE":"#ff3355"}
#         for method, path, name, desc in endpoints:
#             color = mc.get(method, "#7ab3d4")
#             st.markdown(f'<div style="display:flex;align-items:center;gap:12px;padding:7px 0;border-bottom:1px solid rgba(0,180,255,0.06);"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;font-weight:700;color:{color};min-width:46px;">{method}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#00b4ff;min-width:260px;">{path}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.63rem;color:#7ab3d4;min-width:140px;">{name}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;color:#3a6080;">{desc}</span></div>', unsafe_allow_html=True)
#         st.markdown("<br>", unsafe_allow_html=True)
#         st.markdown('<div class="section-hdr">Quick Integration Example</div>', unsafe_allow_html=True)
#         st.code("""import requests

# BASE = "https://abu-sameer-66-phantomeye.hf.space"

# # Person Detection
# with open("image.jpg", "rb") as f:
#     resp = requests.post(f"{BASE}/detect", files={"file": f})
#     print(f"Found {resp.json()['total_persons']} persons")

# # Predictive Exit
# payload = {
#     "frame_id": 42, "frame_width": 640, "frame_height": 480, "fps": 25.0,
#     "detections": [{"person_id": 1, "bbox": {"x1": 550, "y1": 200, "x2": 610, "y2": 320}}]
# }
# resp = requests.post(f"{BASE}/api/predictive-exit/update", json=payload)
# for p in resp.json()["predictions"]:
#     print(f"Person {p['person_id']} → {p['exit_side']} in {p['seconds_to_exit']:.1f}s")
# """, language="python")

#     # ── TAB 5 — FAQ ─────────────────────────────────────
#     with tab5:
#         st.markdown('<div class="section-hdr">Frequently Asked Questions</div>', unsafe_allow_html=True)
#         faqs = [
#             ("Why is PhantomEye slow on first load?",
#              "DeepFace loads TensorFlow models on first use — 15–30 seconds on free HuggingFace tier. Subsequent requests are faster. If it times out, wait 30s and retry."),
#             ("Does PhantomEye require a GPU?",
#              "No. 100% CPU-optimized. YOLOv8-nano, OSNet Re-ID, ByteTrack, and all 13 modules run on CPU only. Designed for standard hardware including Dell Vostro AMD Ryzen."),
#             ("Can I use a live RTSP camera?",
#              "Not yet. Current version supports uploaded video files (MP4/AVI/MOV). RTSP live stream support is planned for Phase 5 (September 2026)."),
#             ("How do I add faces to the OSINT gallery?",
#              "Use API endpoint: POST /osint/add-to-gallery with face image + person_id parameter. View current gallery via GET /osint/gallery."),
#             ("Why does NL Query sometimes fail?",
#              "NL Query uses Groq API (LLaMA 3). Set GROQ_API_KEY in .env file. Free tier has rate limits — wait a few minutes if you hit them."),
#             ("What video formats are supported?",
#              "MP4, AVI, MOV. H.264 encoded MP4 works best. Max analyzed duration: 15 seconds. Recommended resolution: 720p or lower."),
#             ("How accurate is weapon detection?",
#              "mAP50 53.2% overall. Per-class: Handgun 89.5%, Shotgun 96.3%, SMG 98.6%, AR 94.2%. Trained on 714 real weapon images across 9 classes."),
#             ("Can I run PhantomEye locally?",
#              "Yes. Clone from github.com/Abu-Sameer-66/PhantomEye · conda env Python 3.10 · pip install -r requirements.txt · streamlit run app.py (port 7860) + uvicorn api.main:app (port 8000)."),
#             ("What is the difference between BDF and standard face Re-ID?",
#              "Standard Re-ID fails with masks, hats, distance, low resolution. BDF uses only behavioral patterns — walk style, speed, preferred locations. Works fully disguised."),
#             ("How do I report bugs or contribute?",
#              "Open an issue at github.com/Abu-Sameer-66/PhantomEye. For research collaboration contact via sameer-nadeem-portfolio.vercel.app."),
#         ]
#         for q, a in faqs:
#             with st.expander(f"Q: {q}"):
#                 st.markdown(f'<div class="terminal">{a}</div>', unsafe_allow_html=True)


# def main():
#     if "page"             not in st.session_state:
#         st.session_state.page = "landing"
#     if "session_id"       not in st.session_state:
#         st.session_state.session_id = "PE-" + str(uuid.uuid4())[:8].upper()
#     if "first_visit_done" not in st.session_state:
#         st.session_state.first_visit_done = False
#     if "welcome_step"     not in st.session_state:
#         st.session_state.welcome_step = 1

#     page = st.session_state.page

#     if   page == "landing":   landing()
#     elif page == "welcome":   welcome_flow()
#     elif page == "home":      home()
#     elif page == "DETECTION": detection_page()
#     elif page == "ANALYTICS": analytics_page()
#     elif page == "OSINT":     osint_page()
#     elif page == "EMOTION":   emotion_page()
#     elif page == "NL QUERY":  nlquery_page()
#     elif page == "WEAPON":    weapon_page()
#     elif page == "THREAT":    threat_page()
#     elif page == "BDF":       bdf_page()
#     elif page == "SGI":       sgi_page()
#     elif page == "PEV":       pev_page()
#     elif page == "REPORT":    report_page()
#     elif page == "INTEL":     intel_page()
#     elif page == "ZONE":      zone_page()
#     elif page == "GUIDE":     guide_page()


# if __name__ == "__main__":
#     main()




import cv2
import sys
import time
import uuid
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
    --bg-card: rgba(6, 18, 32, 0.88);
    --accent-blue: #00b4ff;
    --accent-cyan: #00fff0;
    --accent-red: #ff3355;
    --accent-green: #00ff88;
    --accent-gold: #f0b429;
    --border-glow: rgba(0, 180, 255, 0.4);
    --border-subtle: rgba(0, 180, 255, 0.1);
    --text-primary: #e8f4ff;
    --text-secondary: #7ab3d4;
    --text-dim: #3a6080;
    --grid-color: rgba(0, 180, 255, 0.03);
    --shadow-blue: 0 0 60px rgba(0, 180, 255, 0.2);
    --shadow-card: 0 12px 40px rgba(0, 0, 0, 0.8);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* TOP ACCENT LINE */
.stApp::after {
    content: ''; position: fixed; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,
        transparent 0%, var(--accent-blue) 15%,
        var(--accent-cyan) 50%, var(--accent-blue) 85%, transparent 100%);
    z-index: 9999; animation: topbar 4s ease-in-out infinite alternate;
}
@keyframes topbar { from { opacity: 0.5; } to { opacity: 1; filter: brightness(1.5); } }

/* BACKGROUND */
.stApp {
    background:
        radial-gradient(ellipse at 10% 30%, rgba(0,80,160,0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 10%, rgba(0,40,100,0.2) 0%, transparent 45%),
        radial-gradient(ellipse at 50% 90%, rgba(0,50,120,0.1) 0%, transparent 55%),
        linear-gradient(180deg, #020408 0%, #030b16 100%) !important;
    min-height: 100vh;
}

/* GRID */
.stApp::before {
    content: ''; position: fixed; inset: 0;
    background-image:
        linear-gradient(var(--grid-color) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-color) 1px, transparent 1px);
    background-size: 56px 56px;
    pointer-events: none; z-index: 0;
}

/* SESSION BAR */
.session-bar {
    display: flex; justify-content: space-between; align-items: center;
    background: rgba(0,8,18,0.75); border: 1px solid rgba(0,180,255,0.08);
    border-radius: 6px; padding: 0.5rem 1.4rem; margin-bottom: 2rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 1px 20px rgba(0,0,0,0.4);
}
.session-bar .sid { color: var(--text-dim); letter-spacing: 0.05em; }
.session-bar .sid span { color: var(--accent-blue); font-weight: 500; }
.session-bar .status { color: var(--accent-green); letter-spacing: 0.25em; font-size: 0.62rem; }
.session-bar .status::before { content: '● '; animation: blink 1.5s infinite; }
.session-bar .badge {
    font-family: 'Rajdhani', sans-serif; font-size: 0.58rem; font-weight: 700;
    letter-spacing: 0.3em; text-transform: uppercase; color: var(--accent-cyan);
    background: rgba(0,255,240,0.06); border: 1px solid rgba(0,255,240,0.25);
    border-radius: 3px; padding: 0.15rem 0.7rem;
}

/* HERO */
.hero-wrap {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 92vh; padding: 3rem 1rem; position: relative; text-align: center;
}
.hero-wrap::before {
    content: ''; position: absolute; width: 800px; height: 800px;
    background: radial-gradient(circle, rgba(0,180,255,0.06) 0%, transparent 68%);
    border-radius: 50%; top: 50%; left: 50%; transform: translate(-50%,-50%);
    animation: pulse-bg 6s ease-in-out infinite;
}
@keyframes pulse-bg {
    0%,100% { transform: translate(-50%,-50%) scale(1); opacity: 0.4; }
    50% { transform: translate(-50%,-50%) scale(1.15); opacity: 0.9; }
}
.hero-eye {
    font-size: 5.5rem; margin-bottom: 1.5rem;
    animation: float 6s ease-in-out infinite;
    filter: drop-shadow(0 0 50px rgba(0,180,255,1));
}
@keyframes float {
    0%,100% { transform: translateY(0) rotate(-2deg); }
    50% { transform: translateY(-24px) rotate(2deg); }
}
.hero-title {
    font-family: 'Exo 2', sans-serif;
    font-size: clamp(3.5rem, 8.5vw, 8rem); font-weight: 900; letter-spacing: 0.1em;
    background: linear-gradient(140deg, #ffffff 0%, #60c8ff 35%, var(--accent-cyan) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 0.6rem; line-height: 0.9;
    animation: reveal 0.8s ease-out both;
}
@keyframes reveal { from { opacity: 0; transform: translateY(32px); } to { opacity: 1; transform: translateY(0); } }
.hero-sub {
    font-family: 'Rajdhani', sans-serif; font-size: clamp(0.78rem, 1.8vw, 1rem);
    font-weight: 300; letter-spacing: 0.5em; color: var(--text-dim);
    margin-bottom: 0.6rem; text-transform: uppercase;
}
.hero-status {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; color: var(--accent-green);
    letter-spacing: 0.28em; margin-bottom: 2.5rem; opacity: 0.9;
}
.hero-status::before { content: '● '; animation: blink 1.5s infinite; }
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.1; } }

/* STATS */
.stats-row { display: flex; gap: 1.5rem; margin-bottom: 2.5rem; justify-content: center; flex-wrap: wrap; }
.stat-item {
    text-align: center; background: rgba(6,18,32,0.7);
    border: 1px solid rgba(0,180,255,0.12); border-radius: 10px;
    padding: 1rem 2rem; backdrop-filter: blur(20px); min-width: 105px;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.stat-item:hover { border-color: rgba(0,180,255,0.3); box-shadow: 0 0 20px rgba(0,180,255,0.1); }
.stat-value { font-family: 'Exo 2', sans-serif; font-size: 1.7rem; font-weight: 900; color: var(--accent-blue); display: block; }
.stat-label { font-size: 0.58rem; letter-spacing: 0.28em; color: var(--text-dim); text-transform: uppercase; margin-top: 0.3rem; display: block; }

/* MODULE GRID */
.module-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(255px, 1fr));
    gap: 1.25rem; width: 100%; max-width: 1240px; margin: 0 auto 3rem;
}
.mod-card {
    background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 14px;
    padding: 1.8rem 1.6rem; position: relative; overflow: hidden;
    transition: all 0.38s cubic-bezier(0.23,1,0.32,1); backdrop-filter: blur(24px);
}
.mod-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1.5px;
    background: linear-gradient(90deg, transparent 0%, var(--accent-blue) 30%, var(--accent-cyan) 70%, transparent 100%);
    opacity: 0; transition: opacity 0.35s;
}
.mod-card::after {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 0% 0%, rgba(0,180,255,0.08) 0%, transparent 60%);
    opacity: 0; transition: opacity 0.38s;
}
.mod-card:hover { border-color: rgba(0,180,255,0.32); transform: translateY(-6px) scale(1.005); box-shadow: var(--shadow-blue), var(--shadow-card); }
.mod-card:hover::before { opacity: 1; }
.mod-card:hover::after  { opacity: 1; }
.mod-card.research-card { border-color: rgba(255,51,85,0.15); }
.mod-card.research-card::before { background: linear-gradient(90deg, transparent, var(--accent-red), #ff8800, transparent); }
.mod-card.research-card::after  { background: radial-gradient(ellipse at 0% 0%, rgba(255,51,85,0.07) 0%, transparent 60%); }
.mod-card.research-card:hover { border-color: rgba(255,51,85,0.45); box-shadow: 0 0 50px rgba(255,51,85,0.1), var(--shadow-card); }

.mod-icon { font-size: 1.9rem; margin-bottom: 0.9rem; display: block; line-height: 1; }
.mod-name {
    font-family: 'Rajdhani', sans-serif; font-size: 0.9rem; font-weight: 700;
    letter-spacing: 0.22em; color: var(--accent-blue); text-transform: uppercase; margin-bottom: 0.45rem;
}
.mod-name.red { color: var(--accent-red); }
.mod-tag {
    display: inline-block; font-size: 0.55rem; letter-spacing: 0.15em;
    color: var(--accent-cyan); background: rgba(0,255,240,0.06);
    border: 1px solid rgba(0,255,240,0.18); border-radius: 3px;
    padding: 0.12rem 0.55rem; margin-bottom: 0.65rem; text-transform: uppercase;
}
.mod-tag.red { color: var(--accent-red); background: rgba(255,51,85,0.06); border-color: rgba(255,51,85,0.22); }
.mod-desc { font-size: 0.74rem; color: var(--text-secondary); line-height: 1.72; }
.mod-meta {
    font-size: 0.6rem; color: var(--text-dim); margin-top: 0.85rem;
    border-top: 1px solid rgba(0,180,255,0.07); padding-top: 0.65rem;
    letter-spacing: 0.03em; line-height: 1.5;
}

/* SCAN LINE */
.scan-line {
    width: 100%; max-width: 860px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,180,255,0.3), rgba(0,255,240,0.5), rgba(0,180,255,0.3), transparent);
    margin: 2rem auto; position: relative; overflow: hidden;
}
.scan-line::after {
    content: ''; position: absolute; width: 90px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,255,240,1), transparent);
    animation: scan 3.5s linear infinite;
}
@keyframes scan { from { left: -90px; } to { left: 100%; } }

/* APP HEADER */
.app-header {
    font-family: 'Exo 2', sans-serif; font-size: 1.6rem; font-weight: 800;
    letter-spacing: 0.35em;
    background: linear-gradient(135deg, #ffffff 0%, #70d4ff 60%, var(--accent-cyan) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    text-align: center; padding: 1.5rem 0 0.35rem;
}
.app-sub {
    font-family: 'Rajdhani', sans-serif; font-size: 0.68rem; color: var(--text-dim);
    letter-spacing: 0.5em; text-align: center; margin-bottom: 1.5rem; text-transform: uppercase;
}

/* NAV DIVIDER */
.nav-divider {
    display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;
}
.nav-divider-line { flex: 1; height: 1px; background: var(--border-subtle); }
.nav-divider-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.55rem; color: var(--text-dim);
    letter-spacing: 0.25em; text-transform: uppercase; white-space: nowrap;
}

/* BUTTONS */
.stButton > button {
    font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; font-size: 0.78rem !important;
    background: rgba(6,18,32,0.9) !important; color: var(--accent-blue) !important;
    border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 7px !important;
    padding: 0.65rem 0.8rem !important; transition: all 0.25s ease !important;
    text-transform: uppercase !important; width: 100% !important;
    white-space: nowrap !important;
}
.stButton > button:hover {
    background: rgba(0,180,255,0.1) !important; border-color: rgba(0,180,255,0.4) !important;
    color: var(--accent-cyan) !important;
    box-shadow: 0 0 20px rgba(0,180,255,0.2), inset 0 0 15px rgba(0,180,255,0.05) !important;
    transform: translateY(-2px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,90,180,0.5), rgba(0,180,255,0.25)) !important;
    border-color: var(--accent-blue) !important; color: #fff !important;
    box-shadow: 0 0 30px rgba(0,180,255,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 40px rgba(0,180,255,0.5) !important;
}

/* SECTION HEADERS */
.section-hdr {
    font-family: 'Exo 2', sans-serif; font-size: 1.2rem; font-weight: 700;
    letter-spacing: 0.28em; color: var(--accent-blue); text-transform: uppercase;
    padding: 0.5rem 0; border-bottom: 1px solid rgba(0,180,255,0.1);
    margin-bottom: 0.5rem; position: relative;
}
.section-hdr::after {
    content: ''; position: absolute; bottom: -1px; left: 0; width: 70px; height: 1.5px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
}
.section-hdr.red { color: var(--accent-red); }
.section-hdr.red::after { background: linear-gradient(90deg, var(--accent-red), #ff8800); }
.section-sub { font-size: 0.7rem; color: var(--text-secondary); letter-spacing: 0.18em; margin-bottom: 1.8rem; text-transform: uppercase; }

/* TERMINAL */
.terminal {
    background: rgba(0,6,16,0.95); border: 1px solid rgba(0,180,255,0.1);
    border-left: 2px solid var(--accent-blue); border-radius: 0 5px 5px 0;
    padding: 0.75rem 1.2rem; font-size: 0.7rem; color: var(--accent-green);
    letter-spacing: 0.12em; margin-top: 1.5rem; position: relative; overflow: hidden;
    box-shadow: inset 0 0 30px rgba(0,0,0,0.5);
}
.terminal::before {
    content: ''; position: absolute; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,136,0.008) 2px, rgba(0,255,136,0.008) 4px);
    pointer-events: none;
}

/* INFO BOX */
.info-box {
    background: rgba(0,8,20,0.8); border: 1px solid rgba(0,180,255,0.09);
    border-radius: 8px; padding: 1.1rem 1.4rem; margin-bottom: 1.5rem;
    font-size: 0.74rem; color: var(--text-secondary); line-height: 1.85;
    box-shadow: inset 0 1px 0 rgba(0,180,255,0.05);
}
.info-box strong { color: var(--accent-blue); font-weight: 500; }

/* STREAMLIT WIDGET OVERRIDES */
.stFileUploader { background: var(--bg-card) !important; border: 1px dashed rgba(0,180,255,0.25) !important; border-radius: 10px !important; padding: 1rem !important; }
.stTextInput > div > div { background: rgba(4,12,24,0.9) !important; border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 7px !important; color: var(--text-primary) !important; font-family: 'IBM Plex Mono', monospace !important; }
.stTextInput > div > div:focus-within { border-color: rgba(0,180,255,0.4) !important; box-shadow: 0 0 12px rgba(0,180,255,0.12) !important; }
.stSelectbox > div > div { background: rgba(4,12,24,0.9) !important; border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 7px !important; color: var(--text-primary) !important; }
.stNumberInput > div > div { background: rgba(4,12,24,0.9) !important; border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 7px !important; }
.stSlider > div > div > div { background: var(--accent-blue) !important; }

div[data-testid="metric-container"] {
    background: rgba(4,14,28,0.85) !important; border: 1px solid rgba(0,180,255,0.1) !important;
    border-radius: 10px !important; padding: 1rem !important; transition: all 0.25s;
}
div[data-testid="metric-container"]:hover { border-color: rgba(0,180,255,0.28) !important; box-shadow: 0 0 16px rgba(0,180,255,0.08) !important; }
div[data-testid="metric-container"] label { color: var(--text-dim) !important; font-size: 0.62rem !important; letter-spacing: 0.22em !important; font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important; }
div[data-testid="metric-container"] div[data-testid="metric-value"] { color: var(--accent-blue) !important; font-family: 'Exo 2', sans-serif !important; font-weight: 800 !important; }

div[data-testid="stDataFrame"] { background: rgba(4,14,28,0.85) !important; border: 1px solid rgba(0,180,255,0.1) !important; border-radius: 10px !important; overflow: hidden !important; }

.stSuccess { background: rgba(0,255,136,0.06) !important; border: 1px solid rgba(0,255,136,0.25) !important; border-radius: 7px !important; color: var(--accent-green) !important; }
.stError, .stWarning { background: rgba(255,51,85,0.06) !important; border: 1px solid rgba(255,51,85,0.25) !important; border-radius: 7px !important; }
.stInfo { background: rgba(0,180,255,0.06) !important; border: 1px solid rgba(0,180,255,0.18) !important; border-radius: 7px !important; color: var(--accent-blue) !important; }

hr { border-color: rgba(0,180,255,0.08) !important; margin: 1.5rem 0 !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(0,180,255,0.4); border-radius: 2px; }
.stSpinner > div { border-color: var(--accent-blue) transparent transparent transparent !important; }
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

@keyframes fadeInUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
.stMarkdown, .stButton, .stFileUploader { animation: fadeInUp 0.38s ease-out both; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    return PersonDetector()

@st.cache_resource
def load_osint():
    return OSINTAudit()

@st.cache_resource
def load_emotion_model():
    from core.emotion import process_frame_emotion
    return process_frame_emotion

@st.cache_resource
def load_weapon_model_cached():
    from core.weapon import load_weapon_model
    return load_weapon_model()


def render_session_bar():
    sid = st.session_state.get("session_id", "PE-XXXXXXXX")
    st.markdown(f"""
    <div class="session-bar">
        <div class="sid"><span>●</span>&nbsp;&nbsp;SESSION: <span>{sid}</span></div>
        <div class="status">ALL SYSTEMS ONLINE</div>
        <div class="badge">OPEN ACCESS</div>
    </div>
    """, unsafe_allow_html=True)


def back_button():
    if st.button("← BACK TO MODULES"):
        st.session_state.page = "home"
        st.rerun()


def landing():
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-eye">👁</div>
      <div class="hero-title">PHANTOMEYE</div>
      <div class="hero-sub">AI-Powered Surveillance Intelligence System</div>
      <div class="hero-status">[ SYSTEM ONLINE ] · OPEN ACCESS · BUILD v3.5</div>

      <div class="stats-row">
        <div class="stat-item"><span class="stat-value">15</span><span class="stat-label">Modules</span></div>
        <div class="stat-item"><span class="stat-value">5</span><span class="stat-label">Novel Algorithms</span></div>
        <div class="stat-item"><span class="stat-value">9</span><span class="stat-label">Weapon Classes</span></div>
        <div class="stat-item"><span class="stat-value">CPU</span><span class="stat-label">No GPU Required</span></div>
      </div>

      <div class="scan-line"></div>

      <div class="module-grid">
        <div class="mod-card">
          <div class="mod-icon">🎯</div>
          <div class="mod-name">Person Detection</div>
          <div class="mod-tag">YOLOv8-nano</div>
          <div class="mod-desc">Real-time person detection on any uploaded image. Returns bounding boxes and per-person confidence scores. Runs entirely on CPU — no GPU required.</div>
          <div class="mod-meta">Model: yolov8n.pt · Class 0 only · Confidence: 0.4 · CPU optimized</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">🔥</div>
          <div class="mod-name">Behavioral Analytics</div>
          <div class="mod-tag">ByteTrack · OpenCV</div>
          <div class="mod-desc">Persistent person IDs across frames, live behavioral heatmap showing movement density, per-person dwell times, and automated loitering alerts from any video.</div>
          <div class="mod-meta">Tracker: ByteTrack IOU · Heatmap: NumPy · Alert threshold: 60s · Max: 15s</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">🕵️</div>
          <div class="mod-name">OSINT Audit</div>
          <div class="mod-tag">LBPH Face Recognition</div>
          <div class="mod-desc">Upload a face and receive a Privacy Exposure Score from 0 to 100. LBPH embeddings matched against a reference gallery. Risk classified as LOW, MEDIUM, or HIGH.</div>
          <div class="mod-meta">Engine: OpenCV LBPH · Similarity: cosine · Score: 0–100 · No data stored</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">🧠</div>
          <div class="mod-name">Emotion Intelligence</div>
          <div class="mod-tag">DeepFace · TensorFlow</div>
          <div class="mod-desc">Multi-face emotion analysis. Returns dominant emotion, estimated age, and gender per face. False-positive filter rejects faces smaller than 15% of frame area.</div>
          <div class="mod-meta">Backend: DeepFace · Detector: OpenCV · Min face: 15% · 7 emotion classes</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">💬</div>
          <div class="mod-name">NL Query Engine</div>
          <div class="mod-tag">Groq LLaMA 3</div>
          <div class="mod-desc">Type a surveillance query in plain English or Roman Urdu. LLaMA 3 extracts structured filters — emotion, gender, age, dwell time, loitering — then matches against records.</div>
          <div class="mod-meta">Model: llama-3.1-8b-instant · English + Roman Urdu · Output: JSON filters</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">⚠️</div>
          <div class="mod-name">Weapon Detection</div>
          <div class="mod-tag">YOLOv8 Custom · 9 Classes</div>
          <div class="mod-desc">Custom YOLOv8 trained on 714 real weapon images. Handgun 89.5%, Shotgun 96.3%, SMG 98.6% average precision. Immediate threat alert fires on any detection.</div>
          <div class="mod-meta">Classes: Handgun · Knife · Shotgun · Sniper · AR · SMG · Sword · Bazooka · GL</div>
        </div>
        <div class="mod-card research-card">
          <div class="mod-icon">📊</div>
          <div class="mod-name red">Threat Momentum Score</div>
          <div class="mod-tag red">Novel Algorithm · TMS v1.0</div>
          <div class="mod-desc">Original research. Accumulates threat signals over time using a compound interest model — loitering, stress emotion, rapid movement, restricted zone, gaze anomaly, group formation.</div>
          <div class="mod-meta">6 signals · Decay: 45s half-life · Amplifier: score/200 · 5 threat levels</div>
        </div>
        <div class="mod-card research-card">
          <div class="mod-icon">🧬</div>
          <div class="mod-name red">Behavioral DNA</div>
          <div class="mod-tag red">Novel Algorithm · BDF v1.0</div>
          <div class="mod-desc">Camera-agnostic person re-identification using behavioral signature alone. Identifies the same person across cameras without face recognition — works through masks, hats, distance.</div>
          <div class="mod-meta">5 components: gait · velocity · spatial · social distance · dwell zones · Threshold: 82%</div>
        </div>
        <div class="mod-card research-card">
          <div class="mod-icon">🕸️</div>
          <div class="mod-name red">Social Graph</div>
          <div class="mod-tag red">Novel Algorithm · SGI v1.0</div>
          <div class="mod-desc">Detects who is associated with whom from movement correlation alone — no prior information needed. Three people entering separately but coordinating get flagged before any overt action.</div>
          <div class="mod-meta">Proximity · velocity sync · dwell overlap · BFS connected-component group detection</div>
        </div>
        <div class="mod-card research-card">
          <div class="mod-icon">🚀</div>
          <div class="mod-name red">Predictive Exit Vector</div>
          <div class="mod-tag red">Novel Algorithm · PEV v1.0</div>
          <div class="mod-desc">Predicts which frame boundary a person will cross and how many seconds remain — 3 to 5 seconds before actual exit. Velocity smoothing plus linear trajectory extrapolation. Designed for camera handoff in multi-camera surveillance grids.</div>
          <div class="mod-meta">Trajectory extrapolation · velocity smoothing · boundary proximity · confidence scoring · no open-source equivalent</div>
        </div>
        <div class="mod-card research-card">
          <div class="mod-icon">🛰️</div>
          <div class="mod-name red">Zone Intelligence</div>
          <div class="mod-tag red">Novel Algorithm · ZIE v1.0</div>
          <div class="mod-desc">Named surveillance zones with 4 types — RESTRICTED, MONITORED, CAPACITY LIMITED, SAFE. Every breach auto-fires TMS proximity_violation signal in real time. Suspicious traversal sequences detected automatically.</div>
          <div class="mod-meta">4 zone types · CRITICAL breach alerts · capacity enforcement · TMS integration · IUB AI Research Lab</div>
        </div>
        <div class="mod-card research-card">
          <div class="mod-icon">🕶️</div>
          <div class="mod-name red">Anonymization Mode</div>
          <div class="mod-tag red">ANE v1.0 · GDPR Compliant</div>
          <div class="mod-desc">Makes persons unidentifiable using 5 modes — Face Blur, Face Pixelate, Full Body Blur, Full Body Pixelate, Silhouette — while preserving all behavioral analytics completely intact.</div>
          <div class="mod-meta">5 modes · Gaussian blur · pixelation · silhouette · analytics intact · GDPR compliant</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">📄</div>
          <div class="mod-name">Intel Report</div>
          <div class="mod-tag">fpdf2 · PDF Export</div>
          <div class="mod-desc">Generate a classified PDF intelligence report from any session. Session overview, weapon threat log in red, per-subject behavioral records, and NL query history.</div>
          <div class="mod-meta">fpdf2 · Dark bg + green text · CLASSIFIED header · Threat sections in red</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">⚡</div>
          <div class="mod-name">System Intel</div>
          <div class="mod-tag">Live Status</div>
          <div class="mod-desc">Live system dashboard with all active modules, tech stack, benchmark results, API endpoint reference, and full deployment metadata for complete transparency.</div>
          <div class="mod-meta">v3.5.0 · HuggingFace Spaces · FastAPI OAS 3.1 · GitHub open source</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">📖</div>
          <div class="mod-name">User Guide</div>
          <div class="mod-tag">Complete Documentation</div>
          <div class="mod-desc">Complete interactive user guide — Quick Start in 5 minutes, per-module step-by-step walkthroughs, novel algorithm deep dives with math, API reference, and FAQ. Re-run the onboarding tour anytime.</div>
          <div class="mod-meta">5-step onboarding · 15 module walkthroughs · 5 algorithm deep dives · API reference · FAQ</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns([1, 2, 1])
    with cols[1]:
        if st.button("INITIALIZE SYSTEM  →", key="enter_btn"):
            if not st.session_state.get("first_visit_done", False):
                st.session_state.page = "welcome"
            else:
                st.session_state.page = "home"
            st.rerun()

def home():
    render_session_bar()
    st.markdown('<div class="app-header">👁 PHANTOMEYE</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">SELECT INTELLIGENCE MODULE · ALL SYSTEMS ACTIVE</div>', unsafe_allow_html=True)

    # Row 1 — Core modules
    st.markdown("""
    <div class="nav-divider">
        <div class="nav-divider-line"></div>
        <div class="nav-divider-label">Core Intelligence</div>
        <div class="nav-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    row1 = [
        ("DETECTION", "Detection"),
        ("ANALYTICS", "Analytics"),
        ("OSINT",     "OSINT"),
        ("EMOTION",   "Emotion"),
        ("NL QUERY",  "NL Query"),
        ("WEAPON",    "Weapon"),
    ]
    cols1 = st.columns(6)
    for i, (key, label) in enumerate(row1):
        with cols1[i]:
            if st.button(label, key=f"mod_{key}"):
                st.session_state.page = key
                st.rerun()

    # Row 2 — Research + utility
    st.markdown("""
    <div class="nav-divider" style="margin-top:0.75rem;">
        <div class="nav-divider-line"></div>
        <div class="nav-divider-label">Novel Research · Utility</div>
        <div class="nav-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    row2 = [
        ("THREAT", "Threat Score"),
        ("BDF",    "Behavioral DNA"),
        ("SGI",    "Social Graph"),
        ("PEV",    "Predictive Exit"),
        ("REPORT", "Report"),
        ("INTEL",  "System"),
        ("ZONE",   "Zone Intel"),
        ("ANON",   "Anonymize"),
        ("GUIDE",  "Guide"),
    ]
    cols2 = st.columns(9)
    for i, (key, label) in enumerate(row2):
        with cols2[i]:
            if st.button(label, key=f"mod2_{key}"):
                st.session_state.page = key
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="terminal">[ PHANTOMEYE v3.5 ] · YOLOv8 loaded · ByteTrack active · '
        'DeepFace online · Groq LLaMA connected · Weapon model ready · '
        'TMS v1.0 active · BDF v1.0 active · SGI v1.0 active · PEV v1.0 active · '
        'ZIE v1.0 active · ANE v1.0 active · All 15 modules ONLINE</div>',
        unsafe_allow_html=True
    )

def detection_page():
    render_session_bar()
    back_button()
    st.markdown('<div class="section-hdr">Person Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">YOLOv8-nano · CPU inference · class 0 persons only · confidence threshold 0.4</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>How it works:</strong> Upload any image and PhantomEye runs YOLOv8-nano inference entirely on CPU. Configured for class 0 (person) detection only at a confidence threshold of 0.4. Each detected person receives a bounding box and confidence score. Expand the detection log below the output image to inspect raw bbox coordinates and confidence per subject. No GPU required at any point.</div>""", unsafe_allow_html=True)
    st.markdown('<div class="terminal">yolov8n.pt · device: cpu · class 0 only · confidence threshold: 0.4</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], key="det_up")
    if uploaded:
        data  = np.frombuffer(uploaded.read(), np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Cannot decode image.")
            return
        with st.spinner("Running inference..."):
            detector   = load_detector()
            t0         = time.time()
            detections = detector.detect(image)
            elapsed    = round(time.time() - t0, 3)
            annotated  = detector.draw(image, detections)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PERSONS DETECTED", len(detections))
        c2.metric("INFERENCE TIME",   f"{elapsed}s")
        c3.metric("MODEL",            "YOLOv8n")
        c4.metric("DEVICE",           "CPU")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection output", use_container_width=True)
        if detections:
            st.markdown('<div class="section-hdr">Detection Log</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Expand each entry to inspect bounding box coordinates and confidence score</div>', unsafe_allow_html=True)
            for i, d in enumerate(detections):
                with st.expander(f"PERSON_{i+1:03d}  ·  CONF: {d['confidence']}"):
                    st.json({"id": i+1, "bbox": list(d["bbox"]), "confidence": d["confidence"]})


def analytics_page():
    render_session_bar()
    back_button()
    st.markdown('<div class="section-hdr">Behavioral Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">ByteTrack · behavioral heatmap · dwell time · loitering alerts</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>How it works:</strong> Upload a video and PhantomEye processes up to 15 seconds of footage. ByteTrack assigns a persistent ID to each person and maintains it across frames, including through brief occlusion. A NumPy heatmap accumulates every pixel position each person visits — high-activity zones appear red. Dwell time is tracked per ID in seconds. If any person remains in one area beyond the loitering threshold, an alert fires listing their tracked ID.</div>""", unsafe_allow_html=True)
    st.markdown('<div class="terminal">ByteTrack IOU matching · heatmap: NumPy accumulation · loitering threshold: 60s · max analysis window: 15s</div>', unsafe_allow_html=True)

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
        st.markdown(f'<div class="terminal">{w}x{h} @ {fps}fps · {total} total frames · analysis cap: {min(total, fps*15)} frames</div>', unsafe_allow_html=True)
        if st.button("RUN BEHAVIORAL ANALYSIS"):
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
                    stat.markdown(f'<div class="terminal">Processing frame {i}/{limit} · active persons: {len(active)}</div>', unsafe_allow_html=True)
            cap.release()
            tmp.unlink(missing_ok=True)
            prog.progress(100)
            stat.empty()
            s = analyzer.summary()
            st.success("Analysis complete")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("TOTAL PERSONS", s.get("total_persons", 0))
            c2.metric("AVG DWELL",     f"{s.get('avg_dwell_sec', 0)}s")
            c3.metric("MAX DWELL",     f"{s.get('max_dwell_sec', 0)}s")
            c4.metric("LOITER ALERTS", s.get("total_alerts", 0))
            if s.get("total_alerts", 0) > 0:
                st.warning(f"Loitering detected — Subject IDs: {s.get('loiterers', [])}")
            heat = analyzer.get_heatmap_overlay(np.zeros((h, w, 3), dtype=np.uint8))
            st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), caption="Behavioral heatmap — red zones indicate highest activity density", use_container_width=True)


def osint_page():
    render_session_bar()
    back_button()
    st.markdown('<div class="section-hdr">OSINT Privacy Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">LBPH face embedding · gallery match · exposure score 0–100 · risk classification</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>How it works:</strong> Upload a face photo and PhantomEye extracts an LBPH (Local Binary Pattern Histogram) embedding from the detected face region. This is compared against every person in the reference gallery using cosine similarity. The Privacy Exposure Score (0–100) reflects recognition confidence — higher score means stronger match. Risk level: LOW (score &lt; 40), MEDIUM (40–70), HIGH (&gt; 70). All processing in-session only — nothing stored server-side at any point.</div>""", unsafe_allow_html=True)
    st.markdown('<div class="terminal">Engine: OpenCV LBPH · Similarity: cosine distance · Score: 0–100 · Risk: LOW / MEDIUM / HIGH · No data retention</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        query_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="osint_up")
    with c2:
        osint = load_osint()
        st.metric("GALLERY SIZE", f"{len(osint.gallery)} persons")
        st.metric("ENGINE",       "LBPH Face Recognition")
    if query_file and st.button("EXECUTE AUDIT"):
        data  = np.frombuffer(query_file.read(), np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Cannot decode image.")
            return
        with st.spinner("Running audit..."):
            result = osint.audit(image, query_id=Path(query_file.name).stem)
        c1, c2, c3 = st.columns(3)
        c1.metric("RISK LEVEL",     result["risk_level"])
        c2.metric("EXPOSURE SCORE", f"{result['exposure_score']}/100")
        c3.metric("MATCHES FOUND",  len(result["matches"]))
        st.markdown(f'<div class="terminal">{result["message"]}</div>', unsafe_allow_html=True)
        if result["matches"]:
            st.markdown('<div class="section-hdr">Match Log</div>', unsafe_allow_html=True)
            for m in result["matches"]:
                st.markdown(f'<div class="terminal">MATCH: {m["matched_id"]} · CONF: {m["confidence"]}% · SOURCE: {m["source"]}</div>', unsafe_allow_html=True)
        vis = osint.visualize(image, result)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="OSINT visualization output", use_container_width=True)


def emotion_page():
    render_session_bar()
    process_frame_emotion = load_emotion_model()
    back_button()
    st.markdown('<div class="section-hdr">Emotion Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">DeepFace · TensorFlow · dominant emotion · age · gender per face</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>How it works:</strong> PhantomEye runs DeepFace analysis on every detected face in the uploaded image. Returns dominant emotion from 7 classes (angry, fear, sad, happy, surprise, neutral, disgust), an estimated age, and gender classification. A false-positive filter discards any face region smaller than 15% of the frame area — prevents noise from distant or partially visible faces. Multiple faces in a single image are processed independently.</div>""", unsafe_allow_html=True)
    st.markdown('<div class="terminal">DeepFace + TensorFlow · OpenCV face detector · min face size: 15% of frame · 7 emotion classes · multi-subject</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded:
        from PIL import Image
        img       = Image.open(uploaded).convert("RGB")
        frame     = np.array(img)
        frame_bgr = frame[:, :, ::-1].copy()
        with st.spinner("Analyzing faces..."):
            annotated, results = process_frame_emotion(frame_bgr)
        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, caption="Original", use_container_width=True)
        with col2:
            st.image(annotated[:, :, ::-1], caption="Emotion analysis output", use_container_width=True)
        if results:
            st.markdown("<hr>")
            st.markdown('<div class="section-hdr">Detected Subjects</div>', unsafe_allow_html=True)
            for i, r in enumerate(results):
                emotion = r.get("dominant_emotion", "N/A").upper()
                age     = int(r.get("age", 0))
                gender  = r.get("dominant_gender", r.get("gender", "N/A"))
                if isinstance(gender, dict):
                    gender = max(gender, key=gender.get)
                c1, c2, c3 = st.columns(3)
                c1.metric(f"SUBJECT {i+1} EMOTION", emotion)
                c2.metric("AGE ESTIMATE",            f"{age} yrs")
                c3.metric("GENDER",                  gender.upper())
        else:
            st.warning("No faces detected in this image.")
    else:
        st.info("Upload a face image to begin analysis.")


def nlquery_page():
    render_session_bar()
    from core.nlquery import parse_nl_query, apply_filters
    back_button()
    st.markdown('<div class="section-hdr">NL Query Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Groq LLaMA 3 · English + Roman Urdu · structured filter extraction</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>How it works:</strong> Type any surveillance query in natural language — English or Roman Urdu both work. Groq's LLaMA 3 (llama-3.1-8b-instant) parses the intent and extracts structured filters: emotion type, gender, age range, minimum dwell time, and loitering status. Filters are applied against person records and matching subjects are returned in a filterable table. This is the first open-source surveillance system with multilingual NL query support including Roman Urdu.</div>""", unsafe_allow_html=True)
    st.markdown('<div class="terminal">llama-3.1-8b-instant via Groq · JSON structured filter extraction · Roman Urdu supported · apply_filters() on person records</div>', unsafe_allow_html=True)

    query = st.text_input("Enter your query", placeholder="show me angry men who were loitering  |  log jo loiter kar rahy thy")
    if query:
        with st.spinner("Parsing query..."):
            result = parse_nl_query(query)
        if result['success']:
            filters = result['filters']
            st.success(f"Understood: {filters['summary']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("EMOTION",   filters['emotion']  or "ANY")
            col2.metric("GENDER",    filters['gender']   or "ANY")
            col3.metric("MAX AGE",   filters['max_age']  or "ANY")
            col4, col5 = st.columns(2)
            col4.metric("LOITERING", "YES" if filters['loitering'] else "ANY")
            col5.metric("MIN DWELL", f"{filters['min_dwell_seconds']}s" if filters['min_dwell_seconds'] else "ANY")
            st.markdown("<hr>")
            st.markdown('<div class="section-hdr">Filter Results — Sample Dataset</div>', unsafe_allow_html=True)
            sample_records = [
                {"id": 1, "emotion": "angry",   "gender": "Man",   "age": 28, "dwell_seconds": 45,  "loitering": False},
                {"id": 2, "emotion": "neutral",  "gender": "Woman", "age": 22, "dwell_seconds": 180, "loitering": True},
                {"id": 3, "emotion": "happy",    "gender": "Man",   "age": 35, "dwell_seconds": 20,  "loitering": False},
                {"id": 4, "emotion": "angry",    "gender": "Man",   "age": 41, "dwell_seconds": 200, "loitering": True},
                {"id": 5, "emotion": "sad",      "gender": "Woman", "age": 19, "dwell_seconds": 90,  "loitering": False},
                {"id": 6, "emotion": "fear",     "gender": "Man",   "age": 26, "dwell_seconds": 310, "loitering": True},
            ]
            matched = apply_filters(sample_records, filters)
            if matched:
                st.success(f"{len(matched)} subject(s) matched from {len(sample_records)} records")
                import pandas as pd
                st.dataframe(pd.DataFrame(matched), use_container_width=True)
            else:
                st.warning("No subjects matched this query.")
        else:
            st.error(f"Parse failed: {result['error']}")
    else:
        st.info("Type a query above — English or Roman Urdu both work.")


def weapon_page():
    render_session_bar()
    back_button()
    st.markdown('<div class="section-hdr">Weapon Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">YOLOv8 custom trained · 9 weapon classes · real-time threat alert</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>How it works:</strong> A custom YOLOv8 model trained from scratch on 714 real-world weapon images across 9 classes — trained on Kaggle T4 GPU. Achieves Handgun 89.5%, Shotgun 96.3%, SMG 98.6% average precision at mAP50 53.2%. Upload any image — detected weapons are highlighted with red bounding boxes and an immediate threat alert fires with the weapon class and confidence score. A clean result confirms the scene is clear.</div>""", unsafe_allow_html=True)
    st.markdown('<div class="terminal">weapon_detector.pt · mAP50: 53.2% · Handgun: 89.5% · Shotgun: 96.3% · SMG: 98.6% · 714 training images · 9 classes</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded:
        from PIL import Image
        from core.weapon import detect_weapons
        img       = Image.open(uploaded).convert("RGB")
        frame     = np.array(img)
        frame_bgr = frame[:, :, ::-1].copy()
        model     = load_weapon_model_cached()
        with st.spinner("Scanning for threats..."):
            annotated, detections = detect_weapons(frame_bgr, model)
        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, caption="Original", use_container_width=True)
        with col2:
            st.image(annotated[:, :, ::-1], caption="Threat analysis output", use_container_width=True)
        st.markdown("<hr>")
        if detections:
            st.error(f"THREAT DETECTED — {len(detections)} weapon(s) identified")
            st.markdown('<div class="section-hdr red">Detected Threats</div>', unsafe_allow_html=True)
            for d in detections:
                c1, c2 = st.columns(2)
                c1.metric("WEAPON CLASS", d['class_name'])
                c2.metric("CONFIDENCE",   f"{d['confidence']:.0%}")
        else:
            st.success("No weapons detected — scene clear")
    else:
        st.info("Upload an image to begin weapon scan.")


def threat_page():
    render_session_bar()
    back_button()
    from core.threat_momentum import ThreatMomentumEngine

    st.markdown('<div class="section-hdr red">Threat Momentum Score</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Novel temporal threat accumulation · compound behavioral signal model · TMS v1.0</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>Research contribution:</strong> Unlike binary threat detection systems that output a single yes/no result, TMS accumulates behavioral signals over time using a compound interest model. Each new signal contributes to the score weighted by importance. When the score is already elevated, new signals contribute proportionally more — the amplifier effect. The score decays with a 45-second half-life when no signals arrive, modeling how real threat situations escalate gradually, not instantaneously.<br><br><strong>6 signals and weights:</strong> loitering (0.28) · stress emotion (0.22) · rapid movement (0.18) · proximity violation (0.15) · gaze anomaly (0.10) · group formation (0.07)</div>""", unsafe_allow_html=True)
    st.markdown('<div class="terminal">TMS v1.0 · decay half-life: 45s · amplifier: 1 + score/200 · 5 levels: CLEAR / LOW / MEDIUM / HIGH / CRITICAL</div>', unsafe_allow_html=True)

    if "tms_engine" not in st.session_state:
        st.session_state.tms_engine = ThreatMomentumEngine()
    engine = st.session_state.tms_engine

    st.markdown("### Subject Input")
    c1, c2, c3 = st.columns(3)
    with c1:
        person_id     = st.number_input("Person ID", min_value=1, value=1)
        dwell_seconds = st.number_input("Dwell Time (seconds)", min_value=0.0, value=0.0, step=5.0)
        is_loitering  = st.checkbox("Loitering detected")
    with c2:
        emotion       = st.selectbox("Detected Emotion", ["none", "neutral", "angry", "fear", "disgust", "sad", "happy", "surprise"])
        in_restricted = st.checkbox("In restricted zone")
        group_anomaly = st.checkbox("Group anomaly detected")
    with c3:
        px = st.number_input("Position X (px)", min_value=0, value=320)
        py = st.number_input("Position Y (px)", min_value=0, value=240)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("UPDATE THREAT SCORE", type="primary"):
            result = engine.update_person(
                person_id=person_id, position=(px, py),
                emotion=None if emotion == "none" else emotion,
                dwell_seconds=dwell_seconds, is_loitering=is_loitering,
                in_restricted_zone=in_restricted, group_anomaly=group_anomaly,
            )
            st.session_state.last_tms = result
    with col_b:
        if st.button("RESET THIS PERSON"):
            engine.reset_person(person_id)
            if "last_tms" in st.session_state:
                del st.session_state.last_tms
            st.success(f"Person {person_id} profile cleared.")

    if "last_tms" in st.session_state:
        r = st.session_state.last_tms
        level_colors = {"CLEAR": "#10b981", "LOW": "#3b82f6", "MEDIUM": "#f59e0b", "HIGH": "#ef4444", "CRITICAL": "#ff0033"}
        color = level_colors.get(r.threat_level, "#ffffff")
        st.markdown(f"""
        <div style="text-align:center; padding:2.5rem; margin:1.5rem 0;
            background:rgba(0,4,12,0.97); border:2px solid {color};
            border-radius:12px; box-shadow: 0 0 60px {color}18;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.58rem; color:#2a4060; letter-spacing:0.4em; margin-bottom:0.8rem; text-transform:uppercase;">
                Threat Momentum Score · Person {r.person_id}
            </div>
            <div style="font-size:5.5rem; font-weight:900; color:{color}; font-family:'Exo 2',sans-serif; line-height:0.9; letter-spacing:-0.02em;">{r.tms_score:.1f}</div>
            <div style="font-size:1rem; font-weight:700; color:{color}; letter-spacing:0.5em; margin-top:0.7rem; font-family:'Rajdhani',sans-serif;">{r.threat_level}</div>
            <div style="font-size:0.62rem; color:#2a4060; margin-top:0.8rem; font-family:'IBM Plex Mono',monospace; letter-spacing:0.08em;">
                Momentum: {r.momentum:+.2f}/frame &nbsp;&nbsp;|&nbsp;&nbsp; Time in system: {r.time_in_system}s
            </div>
        </div>
        """, unsafe_allow_html=True)
        if r.alert:
            st.error(r.alert_message)
        c1, c2, c3 = st.columns(3)
        c1.metric("ACTIVE SIGNALS", len(r.active_signals))
        c2.metric("MOMENTUM",       f"{r.momentum:+.3f}")
        c3.metric("TIME IN SYSTEM", f"{r.time_in_system}s")
        if r.signal_breakdown:
            st.markdown('<div class="section-hdr">Signal Breakdown</div>', unsafe_allow_html=True)
            import pandas as pd
            df = pd.DataFrame([{"Signal": k.replace("_", " ").upper(), "Score Contribution": round(v, 3)} for k, v in r.signal_breakdown.items()])
            st.dataframe(df, use_container_width=True)

    st.markdown("<hr>")
    st.markdown('<div class="section-hdr">Session Summary</div>', unsafe_allow_html=True)
    summary = engine.summary()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("PERSONS TRACKED", summary["total_persons_tracked"])
    s2.metric("TOTAL ALERTS",    summary["total_alerts"])
    s3.metric("HIGHEST TMS",     summary["highest_tms"])
    s4.metric("AVG TMS",         summary["avg_tms"])
    if summary["level_distribution"]:
        st.markdown('<div class="terminal">Distribution: ' + ' · '.join(f"{k}: {v}" for k, v in summary["level_distribution"].items()) + '</div>', unsafe_allow_html=True)
    if st.button("RESET ALL PROFILES"):
        engine.reset_all()
        if "last_tms" in st.session_state:
            del st.session_state.last_tms
        st.success("All threat profiles cleared.")


def bdf_page():
    render_session_bar()
    back_button()
    from core.behavioral_dna import BehavioralDNAEngine

    st.markdown('<div class="section-hdr red">Behavioral DNA Fingerprint</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Camera-agnostic re-identification · no face required · pure movement signature · BDF v1.0</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>Research contribution:</strong> Identifies the same person across cameras using behavioral signature alone — gait rhythm, velocity profile, spatial preference zones, social distance pattern, and dwell locations. Works with masks, hats, and at distances where face recognition completely fails. When a person re-enters the scene with a new tracking ID, BDF matches them to their previous identity using cosine similarity on a 5-component behavioral feature vector. Match threshold: 82%.<br><br><strong>5 behavioral components:</strong> gait signature (stride rhythm histogram) · velocity profile (speed distribution) · spatial preference (normalized grid heatmap) · social distance average · dwell zone signature (stopping locations)</div>""", unsafe_allow_html=True)
    st.markdown('<div class="terminal">BDF v1.0 · 5 behavioral signals · cosine similarity · match threshold: 82% · min observations: 15 frames</div>', unsafe_allow_html=True)

    if "bdf_engine" not in st.session_state:
        st.session_state.bdf_engine = BehavioralDNAEngine(640, 480)
    engine = st.session_state.bdf_engine

    st.markdown("### Add Observations")
    c1, c2, c3 = st.columns(3)
    with c1:
        obs_id   = st.number_input("Person ID", min_value=1, value=1)
        pos_x    = st.number_input("Position X", min_value=0, max_value=640, value=320)
    with c2:
        pos_y    = st.number_input("Position Y", min_value=0, max_value=480, value=240)
        soc_dist = st.number_input("Nearest person distance (px)", min_value=0.0, value=100.0)
    with c3:
        n_obs = st.number_input("Observations to simulate", min_value=1, max_value=100, value=30)

    if st.button("SIMULATE OBSERVATIONS"):
        for i in range(int(n_obs)):
            x = int(pos_x + i * 2 + np.random.randn() * 3)
            y = int(pos_y + np.sin(i * 0.3) * 15 + np.random.randn() * 2)
            engine.observe(obs_id, (max(0, x), max(0, y)), soc_dist)
        st.success(f"Added {n_obs} observations for Person {obs_id}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("REGISTER TO GALLERY", type="primary"):
            bdf = engine.extract_and_register(obs_id)
            if bdf:
                st.success(f"Person {obs_id} registered — confidence: {bdf.confidence:.2f} | observations: {bdf.observation_count}")
            else:
                st.warning(f"Insufficient data. Need at least 15 observations for Person {obs_id}.")
    with col_b:
        if st.button("MATCH AGAINST GALLERY"):
            result = engine.match_against_gallery(obs_id)
            st.session_state.last_bdf = result

    if "last_bdf" in st.session_state:
        r = st.session_state.last_bdf
        color = "#00b4ff" if r.is_match else "#10b981"
        st.markdown(f"""
        <div style="padding:2rem; margin:1rem 0; background:rgba(0,4,12,0.97);
            border:2px solid {color}; border-radius:10px; box-shadow: 0 0 40px {color}18;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.58rem; color:#2a4060; letter-spacing:0.35em; margin-bottom:0.6rem; text-transform:uppercase;">
                Behavioral DNA Match · Person {r.query_id}
            </div>
            <div style="font-size:2.2rem; font-weight:900; color:{color}; font-family:'Exo 2',sans-serif; letter-spacing:0.05em;">{"MATCH FOUND" if r.is_match else "NO MATCH"}</div>
            <div style="font-size:0.72rem; color:#5a8090; margin-top:0.8rem; font-family:'IBM Plex Mono',monospace; line-height:1.65;">{r.explanation}</div>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("SIMILARITY",  f"{r.similarity:.1%}")
        c2.metric("MATCHED ID",  str(r.matched_id) if r.matched_id else "None")
        c3.metric("CONFIDENCE",  f"{r.confidence:.2f}")

    st.markdown("<hr>")
    st.markdown('<div class="section-hdr">Gallery & Session</div>', unsafe_allow_html=True)
    summary = engine.summary()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("TRACKED",   summary["persons_tracked"])
    s2.metric("BDF READY", summary["bdf_ready"])
    s3.metric("GALLERY",   summary["gallery_size"])
    s4.metric("MATCHES",   summary["matches_detected"])
    if st.button("RESET ALL"):
        engine.reset_all()
        if "last_bdf" in st.session_state:
            del st.session_state.last_bdf
        st.success("BDF engine reset.")


def sgi_page():
    render_session_bar()
    back_button()
    from core.social_graph import SocialGraphEngine

    st.markdown('<div class="section-hdr red">Social Graph Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Real-time group detection · no prior information · pure behavioral correlation · SGI v1.0</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>Research contribution:</strong> Detects "who is with whom" from surveillance footage without any prior information — no face recognition, no name lists, no pre-registration. Three bank robbers entering a building separately — SGI detects their association before any overt action occurs, purely from movement correlation. Uses three behavioral signals: spatial proximity, velocity synchronization (do they accelerate and decelerate together?), and shared dwell zones. Connected-component BFS then extracts groups from the link graph.<br><br><strong>Link strength formula:</strong> proximity score (0.40) + Pearson velocity correlation (0.35) + dwell zone overlap (0.25)</div>""", unsafe_allow_html=True)
    st.markdown('<div class="terminal">SGI v1.0 · proximity threshold: 150px · Pearson velocity correlation · group detection: BFS connected-component analysis</div>', unsafe_allow_html=True)

    if "sgi_engine" not in st.session_state:
        st.session_state.sgi_engine = SocialGraphEngine(proximity_px=150)
    engine = st.session_state.sgi_engine

    st.markdown("### Simulate Person Movement")
    c1, c2, c3 = st.columns(3)
    with c1:
        obs_id = st.number_input("Person ID", min_value=1, value=1)
        pos_x  = st.number_input("Start Position X", min_value=0, max_value=1920, value=320)
    with c2:
        pos_y  = st.number_input("Start Position Y", min_value=0, max_value=1080, value=240)
        n_obs  = st.number_input("Frames to simulate", min_value=1, max_value=200, value=50)
    with c3:
        move_x = st.number_input("Movement X per frame", min_value=-10, max_value=10, value=2)
        move_y = st.number_input("Movement Y per frame", min_value=-10, max_value=10, value=0)

    if st.button("SIMULATE MOVEMENT"):
        for i in range(int(n_obs)):
            x = int(pos_x + i * move_x + np.random.randn() * 2)
            y = int(pos_y + i * move_y + np.random.randn() * 2)
            engine.observe(obs_id, (max(0, x), max(0, y)))
        engine._update_links()
        st.success(f"Simulated {n_obs} frames for Person {obs_id}")

    if st.button("DETECT GROUPS", type="primary"):
        st.session_state.sgi_result = {
            "groups":  engine.detect_groups(),
            "links":   engine.get_all_links(),
            "summary": engine.summary(),
        }

    if "sgi_result" in st.session_state:
        res     = st.session_state.sgi_result
        groups  = res["groups"]
        links   = res["links"]
        summary = res["summary"]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("PERSONS TRACKED",  summary["persons_tracked"])
        s2.metric("ACTIVE LINKS",     summary["active_links"])
        s3.metric("GROUPS DETECTED",  summary["groups_detected"])
        s4.metric("TOTAL ALERTS",     summary["total_alerts"])

        if groups:
            st.markdown('<div class="section-hdr">Detected Groups</div>', unsafe_allow_html=True)
            for g in groups:
                color = "#ef4444" if g.alert else "#00b4ff"
                st.markdown(f"""
                <div style="padding:1rem 1.5rem; margin:0.5rem 0; background:rgba(0,4,12,0.92); border:1px solid {color}; border-radius:8px;">
                    <div style="font-family:'Rajdhani',sans-serif; font-size:0.82rem; font-weight:700; color:{color}; letter-spacing:0.18em; margin-bottom:0.4rem;">
                        GROUP {g.group_id} · {g.formation.upper()} · Cohesion: {g.cohesion:.3f}
                    </div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.66rem; color:#5a8090; line-height:1.5;">
                        Members: {g.members}{"  ·  ALERT: " + g.alert_reason if g.alert else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No groups detected yet. Simulate matching movement patterns for multiple persons, then detect groups.")

        if links:
            st.markdown('<div class="section-hdr">Social Link Graph</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">All pairwise behavioral associations detected between tracked persons</div>', unsafe_allow_html=True)
            import pandas as pd
            df = pd.DataFrame([{
                "Persons":         f"{l.person_a} -- {l.person_b}",
                "Strength":        l.strength,
                "Type":            l.link_type,
                "Frames Observed": l.frame_count,
                "Proximity (px)":  l.evidence.get("proximity_px", 0),
                "Velocity Corr":   l.evidence.get("velocity_corr", 0),
            } for l in links])
            st.dataframe(df, use_container_width=True)

    st.markdown("<hr>")
    if st.button("RESET ENGINE"):
        engine.reset_all()
        if "sgi_result" in st.session_state:
            del st.session_state.sgi_result
        st.success("Social graph engine reset.")


def pev_page():
    render_session_bar()
    back_button()
    from core.predictive_exit import PredictiveExitEngine

    st.markdown('<div class="section-hdr red">Predictive Exit Vector</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Frame boundary exit prediction · 3–5 seconds ahead · camera handoff intelligence · PEV v1.0</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <strong>Research contribution:</strong> PEV v1.0 tracks each person's position history and
        computes a smoothed velocity vector using a sliding window over recent frames.
        Linear trajectory extrapolation then determines which frame boundary — LEFT, RIGHT, TOP, or BOTTOM —
        the person will cross and how many seconds remain before exit.
        Prediction fires <strong>3 to 5 seconds before actual exit</strong>, enabling downstream camera handoff
        in multi-camera surveillance grids. Confidence is composed from three factors: velocity stability
        (is the direction consistent?), boundary proximity (how close are they?), and history depth
        (how many frames observed?). No equivalent open-source implementation exists for
        real-time multi-person exit prediction in surveillance systems.
        <br><br>
        <strong>Algorithm:</strong> position history → sliding-window velocity smoothing →
        linear trajectory extrapolation → boundary intersection detection →
        confidence scoring → ExitPrediction output
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="terminal">PEV v1.0 · velocity smoothing window: 5 frames · prediction horizon: 4s · confidence: stability × proximity × depth · IUB AI Research Lab</div>', unsafe_allow_html=True)

    st.markdown("### Simulate Exit Prediction")
    st.markdown('<div class="section-sub">Manually feed person positions to test the prediction engine in real time</div>', unsafe_allow_html=True)

    if "pev_engine" not in st.session_state:
        st.session_state.pev_engine = PredictiveExitEngine(frame_width=640, frame_height=480, fps=25)
    engine = st.session_state.pev_engine

    c1, c2, c3 = st.columns(3)
    with c1:
        sim_person_id = st.number_input("Person ID", min_value=1, value=1, key="pev_pid")
        bbox_x1       = st.number_input("BBox X1", min_value=0, max_value=620, value=400, key="pev_x1")
    with c2:
        bbox_y1       = st.number_input("BBox Y1", min_value=0, max_value=460, value=200, key="pev_y1")
        bbox_w        = st.number_input("BBox Width", min_value=20, max_value=200, value=50, key="pev_w")
    with c3:
        bbox_h        = st.number_input("BBox Height", min_value=20, max_value=300, value=100, key="pev_h")
        n_auto_frames = st.number_input("Auto-simulate frames", min_value=1, max_value=50, value=20, key="pev_nf")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("FEED SINGLE FRAME"):
            dets = [{"person_id": sim_person_id, "bbox": [bbox_x1, bbox_y1, bbox_x1 + bbox_w, bbox_y1 + bbox_h]}]
            preds = engine.update(dets)
            st.session_state.pev_result = preds

    with col_b:
        if st.button("AUTO-SIMULATE →RIGHT", type="primary"):
            preds_last = []
            for i in range(int(n_auto_frames)):
                x1 = bbox_x1 + i * 12
                dets = [{"person_id": sim_person_id, "bbox": [x1, bbox_y1, x1 + bbox_w, bbox_y1 + bbox_h]}]
                preds_last = engine.update(dets)
            st.session_state.pev_result = preds_last
            st.success(f"Simulated {n_auto_frames} frames — person moving RIGHT")

    with col_c:
        if st.button("RESET ENGINE"):
            engine.reset()
            if "pev_result" in st.session_state:
                del st.session_state.pev_result
            st.success("PEV engine reset.")

    if "pev_result" in st.session_state:
        preds = st.session_state.pev_result
        st.markdown("<hr>")
        st.markdown('<div class="section-hdr">Live Predictions</div>', unsafe_allow_html=True)

        if not preds or all(p.exit_side == "NONE" for p in preds):
            st.info("No exit predicted yet — feed more frames or simulate movement toward a boundary.")
        else:
            for pred in preds:
                if pred.exit_side == "NONE":
                    continue

                side_colors = {
                    "LEFT":   "#00b4ff",
                    "RIGHT":  "#00ff88",
                    "TOP":    "#f0b429",
                    "BOTTOM": "#ff3355",
                }
                color  = side_colors.get(pred.exit_side, "#ffffff")
                alert_html = (
                    '<div style="color:#ff3355;font-weight:700;font-size:0.75rem;'
                    'letter-spacing:0.2em;margin-top:0.5rem;">⚠ EXIT IMMINENT</div>'
                    if pred.alert else ""
                )

                st.markdown(f"""
                <div style="padding:1.8rem 2rem; margin:0.75rem 0;
                    background:rgba(0,4,12,0.97); border:2px solid {color};
                    border-radius:10px; box-shadow: 0 0 40px {color}18;
                    font-family:'IBM Plex Mono',monospace;">
                    <div style="font-size:0.55rem; color:#2a4060; letter-spacing:0.4em;
                        margin-bottom:0.7rem; text-transform:uppercase;">
                        Predictive Exit Vector · Person {pred.person_id}
                    </div>
                    <div style="display:flex; align-items:center; gap:2rem; flex-wrap:wrap;">
                        <div>
                            <div style="font-size:2.5rem; font-weight:900; color:{color};
                                font-family:'Exo 2',sans-serif; line-height:1;">
                                {pred.exit_side}
                            </div>
                            <div style="font-size:0.6rem; color:#3a6080; margin-top:0.2rem;
                                letter-spacing:0.15em;">EXIT SIDE</div>
                        </div>
                        <div>
                            <div style="font-size:2.5rem; font-weight:900; color:#fff;
                                font-family:'Exo 2',sans-serif; line-height:1;">
                                {pred.seconds_to_exit:.2f}s
                            </div>
                            <div style="font-size:0.6rem; color:#3a6080; margin-top:0.2rem;
                                letter-spacing:0.15em;">TIME TO EXIT</div>
                        </div>
                        <div>
                            <div style="font-size:2.5rem; font-weight:900; color:#00fff0;
                                font-family:'Exo 2',sans-serif; line-height:1;">
                                {pred.confidence:.3f}
                            </div>
                            <div style="font-size:0.6rem; color:#3a6080; margin-top:0.2rem;
                                letter-spacing:0.15em;">CONFIDENCE</div>
                        </div>
                        <div>
                            <div style="font-size:1.1rem; font-weight:700; color:#7ab3d4;
                                font-family:'Rajdhani',sans-serif; line-height:1.3;">
                                ({pred.predicted_exit_point[0]}, {pred.predicted_exit_point[1]})
                            </div>
                            <div style="font-size:0.6rem; color:#3a6080; margin-top:0.2rem;
                                letter-spacing:0.15em;">EXIT POINT (px)</div>
                        </div>
                    </div>
                    <div style="margin-top:0.85rem; font-size:0.62rem; color:#2a4060;">
                        Velocity: vx={pred.current_velocity[0]:+.2f} · vy={pred.current_velocity[1]:+.2f} px/frame
                    </div>
                    {alert_html}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr>")
        st.markdown('<div class="section-hdr">Engine Status</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        s1.metric("ACTIVE TRACKS", len(engine.tracks))
        s2.metric("FRAMES PROCESSED", engine.frame_counter)

        if engine.tracks:
            st.markdown('<div class="section-sub">Tracked persons — history depth per ID</div>', unsafe_allow_html=True)
            import pandas as pd
            rows = [{"Person ID": pid, "History Frames": len(t.positions)} for pid, t in engine.tracks.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


def report_page():
    render_session_bar()
    from core.reporter import generate_report
    back_button()
    st.markdown('<div class="section-hdr">Intelligence Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Classified PDF · session data · threat log · subject records · one-click download</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>How it works:</strong> Fill in session data — total persons detected, loitering alerts, per-subject behavioral records with emotion and dwell time, and any weapon detections from the session. Click Generate and PhantomEye produces a classified PDF using fpdf2. Dark background with green terminal-style text. Weapon threat sections highlighted in red. CLASSIFIED header on the first page. The file is immediately available for download — nothing is stored server-side at any point.</div>""", unsafe_allow_html=True)
    st.markdown('<div class="terminal">fpdf2 · dark theme · CLASSIFIED header · weapon threat sections in red · immediate download · zero server-side storage</div>', unsafe_allow_html=True)

    st.markdown("### Session Data")
    col1, col2 = st.columns(2)
    with col1:
        session_id       = st.text_input("Session ID",           value=st.session_state.get("session_id", "PE-SESSION-001"))
        total_persons    = st.number_input("Total Persons",      min_value=0, value=5)
        duration         = st.number_input("Duration (seconds)", min_value=0, value=300)
    with col2:
        loitering_alerts = st.number_input("Loitering Alerts",   min_value=0, value=1)
        nl_query         = st.text_input("NL Query (optional)",  value="")
        nl_result        = st.text_input("NL Result (optional)", value="")

    st.markdown("### Detected Subjects")
    num_subjects = st.slider("Number of subjects", 1, 10, 3)
    detections = []
    for i in range(num_subjects):
        c1, c2, c3, c4, c5, _ = st.columns(6)
        detections.append({
            "id":            i + 1,
            "emotion":       c1.selectbox(f"Emotion {i+1}", ["neutral","angry","happy","sad","fear","surprise"], key=f"em_{i}"),
            "gender":        c2.selectbox(f"Gender {i+1}",  ["Man","Woman"], key=f"gen_{i}"),
            "age":           c3.number_input(f"Age {i+1}",  10, 80, 25, key=f"age_{i}"),
            "dwell_seconds": c4.number_input(f"Dwell {i+1}", 0, 600, 60, key=f"dw_{i}"),
            "loitering":     c5.checkbox(f"Loiter {i+1}",   key=f"lo_{i}"),
        })

    st.markdown("### Weapon Detections")
    has_weapon = st.checkbox("Weapon detected in session?")
    weapon_detections = []
    if has_weapon:
        wc1, wc2 = st.columns(2)
        weapon_class = wc1.selectbox("Weapon Class", ["Handgun","Knife","Shotgun","SMG","Automatic Rifle","Sniper","Sword"])
        weapon_conf  = wc2.slider("Confidence", 0.3, 1.0, 0.85)
        weapon_detections.append({"class_name": weapon_class, "confidence": weapon_conf})

    st.markdown("<hr>")
    if st.button("GENERATE PDF REPORT", type="primary"):
        data = {
            "session_id":        session_id,
            "total_persons":     total_persons,
            "duration_seconds":  duration,
            "loitering_alerts":  loitering_alerts,
            "weapon_detections": weapon_detections,
            "detections":        detections,
            "heatmap_img":       None,
            "frame_sample":      None,
            "nl_query":          nl_query,
            "nl_result":         nl_result,
        }
        with st.spinner("Generating classified report..."):
            path = generate_report(data)
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        st.success("Report generated.")
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"phantomeye_report_{session_id}.pdf",
            mime="application/pdf"
        )


def intel_page():
    render_session_bar()
    back_button()
    st.markdown('<div class="section-hdr">System Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Module registry · model benchmarks · deployment info · novel contributions</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SYSTEM",  "PhantomEye")
    c2.metric("VERSION", "v3.5.0")
    c3.metric("STATUS",  "ONLINE")
    c4.metric("MODULES", "15 ACTIVE")

    st.markdown("<br>", unsafe_allow_html=True)
    modules_info = [
        ("DETECTION",         "YOLOv8-nano",   "yolov8n.pt · class 0 · confidence 0.4+ · CPU only"),
        ("ANALYTICS",         "ByteTrack",     "IOU matching · NumPy heatmap · dwell time tracking · loitering threshold: 60s"),
        ("OSINT",             "LBPH Face",     "LBPH embedding · cosine gallery search · score 0–100 · LOW/MEDIUM/HIGH risk"),
        ("EMOTION",           "DeepFace + TF", "7 emotion classes · age + gender · OpenCV detector · 15% min face size filter"),
        ("NL QUERY",          "Groq LLaMA 3",  "llama-3.1-8b-instant · English + Roman Urdu · JSON structured filter extraction"),
        ("WEAPON",            "YOLOv8 Custom", "9 classes · mAP50 53.2% · Handgun 89.5% · Shotgun 96.3% · SMG 98.6%"),
        ("THREAT MOMENTUM",   "TMS v1.0",      "Novel · 6 behavioral signals · compound amplifier · 45s decay · 5 threat levels"),
        ("BEHAVIORAL DNA",    "BDF v1.0",      "Novel · 5 behavioral components · cosine similarity · 82% match threshold"),
        ("SOCIAL GRAPH",      "SGI v1.0",      "Novel · proximity + velocity sync + dwell overlap · BFS group detection"),
        ("PREDICTIVE EXIT",   "PEV v1.0",      "Novel · velocity smoothing · linear trajectory extrapolation · boundary prediction 3–5s ahead · camera handoff"),
        ("ZONE INTELLIGENCE", "ZIE v1.0",      "Novel · 4 zone types · RESTRICTED/MONITORED/CAPACITY/SAFE · TMS integration · breach detection · suspicious sequence"),
        ("ANONYMIZATION",     "ANE v1.0",      "GDPR compliant · 5 modes · face blur/pixelate · body blur/pixelate · silhouette · analytics intact"),
        ("REPORT",            "fpdf2",         "Classified PDF · dark theme · CLASSIFIED header · threat sections in red"),
        ("API",               "FastAPI",       "OAS 3.1 · CORS enabled · uvicorn · modular route handlers"),
        ("USER GUIDE",        "Interactive",   "5-step onboarding · 15 module walkthroughs · 5 novel algo deep dives · API reference · FAQ · re-runnable tour"),
    ]
    for name, tech, desc in modules_info:
        with st.expander(f"{name}  ·  {tech}  ·  ACTIVE"):
            st.markdown(f'<div class="terminal">{desc}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.json({
        "author":              "Abu-Sameer-66",
        "github":              "https://github.com/Abu-Sameer-66/PhantomEye",
        "huggingface":         "https://abu-sameer-66-phantomeye.hf.space",
        "stack":               ["Python 3.10", "YOLOv8", "DeepFace", "ByteTrack", "FastAPI", "Streamlit", "Groq", "fpdf2"],
        "novel_contributions": [
            "Threat Momentum Score (TMS v1.0) — temporal compound threat accumulation",
            "Behavioral DNA Fingerprint (BDF v1.0) — camera-agnostic behavioral re-ID",
            "Social Graph Intelligence (SGI v1.0) — implicit group detection from movement",
            "Predictive Exit Vector (PEV v1.0) — 3-5s ahead frame boundary exit prediction",
            "Zone Intelligence Engine (ZIE v1.0) — named zone breach detection with TMS integration",
        ],
        "paper_status":        "in progress",
        "version":             "v3.5.0",
        "status":              "online",
        "access":              "open",
        "user_guide":          "active — 5-step onboarding + 15 module walkthroughs",
    })


def welcome_flow():
    """First-visit onboarding — 5 steps."""
    if "welcome_step" not in st.session_state:
        st.session_state.welcome_step = 1
    step = st.session_state.welcome_step
    progress = (step - 1) / 4
    st.markdown(f"""
    <div style="position:fixed;top:0;left:0;right:0;height:2px;z-index:9999;
         background:linear-gradient(90deg,#00b4ff {int(progress*100)}%,rgba(0,180,255,0.1) {int(progress*100)}%);"></div>
    """, unsafe_allow_html=True)
    render_session_bar()
    steps_html = ""
    for i in range(1, 6):
        if i < step:   color, border, txt = "#00ff88","#00ff88","#020408"
        elif i == step: color, border, txt = "transparent","#00b4ff","#fff"
        else:           color, border, txt = "transparent","#1a3a5c","#1a3a5c"
        steps_html += f'<div style="width:28px;height:28px;border-radius:50%;border:2px solid {border};background:{color};display:flex;align-items:center;justify-content:center;font-family:\'IBM Plex Mono\',monospace;font-size:11px;font-weight:700;color:{txt};">{"✓" if i < step else i}</div>'
        if i < 5: steps_html += f'<div style="flex:1;height:1px;background:{"#00ff88" if i < step else "#1a3a5c"};margin:0 4px;"></div>'
    st.markdown(f'<div style="display:flex;align-items:center;gap:0;max-width:360px;margin:2rem auto 0;">{steps_html}</div>', unsafe_allow_html=True)

    if step == 1:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem 2rem;">
            <div style="font-size:5rem;margin-bottom:1.5rem;filter:drop-shadow(0 0 40px rgba(0,180,255,0.8));">👁</div>
            <div style="font-family:'Exo 2',sans-serif;font-size:clamp(2.5rem,6vw,5rem);font-weight:900;letter-spacing:0.1em;
                 background:linear-gradient(135deg,#fff 0%,#60c8ff 50%,#00fff0 100%);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:0.75rem;">
                PHANTOMEYE</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:0.9rem;letter-spacing:0.4em;color:#3a6080;text-transform:uppercase;margin-bottom:0.5rem;">
                AI-Powered Surveillance Intelligence System</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#00ff88;letter-spacing:0.25em;margin-bottom:3rem;">
                ● BUILD v3.5 · 15 MODULES · 5 NOVEL ALGORITHMS · OPEN ACCESS</div>
            <div style="max-width:600px;margin:0 auto 3rem;background:rgba(0,8,20,0.8);border:1px solid rgba(0,180,255,0.1);
                 border-radius:12px;padding:1.5rem 2rem;font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:#7ab3d4;line-height:1.8;text-align:left;">
                PhantomEye is a production-grade AI surveillance system built entirely on CPU — no GPU required.
                Upload any image or video and get instant intelligent analysis across 15 specialized modules.
                Five original research algorithms <span style="color:#00b4ff;font-weight:700;">TMS · BDF · SGI · PEV · ZIE</span>
                have no open-source equivalents anywhere.</div>
        </div>""", unsafe_allow_html=True)

    elif step == 2:
        st.markdown("""
        <div style="text-align:center;padding:2rem 1rem 1rem;">
            <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;letter-spacing:0.2em;color:#00b4ff;margin-bottom:0.5rem;">15 INTELLIGENCE MODULES</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#3a6080;letter-spacing:0.2em;margin-bottom:2rem;">CORE INTELLIGENCE · NOVEL RESEARCH · UTILITY</div>
        </div>""", unsafe_allow_html=True)
        modules_overview = [
            ("🎯","Person Detection","YOLOv8-nano · CPU · Conf 0.4",False),
            ("🔥","Behavioral Analytics","ByteTrack · Heatmap · Dwell",False),
            ("🕵️","OSINT Audit","LBPH Face · Score 0–100",False),
            ("🧠","Emotion Intelligence","DeepFace · 7 Classes",False),
            ("💬","NL Query Engine","Groq LLaMA 3 · Roman Urdu",False),
            ("⚠️","Weapon Detection","YOLOv8 Custom · 9 Classes",False),
            ("📊","Threat Momentum","Novel · TMS v1.0",True),
            ("🧬","Behavioral DNA","Novel · BDF v1.0",True),
            ("🕸️","Social Graph","Novel · SGI v1.0",True),
            ("🚀","Predictive Exit","Novel · PEV v1.0",True),
            ("🛰️","Zone Intelligence","Novel · ZIE v1.0",True),
            ("🕶️","Anonymization","Novel · ANE v1.0",True),
            ("📄","Intel Report","fpdf2 · Classified PDF",False),
            ("⚡","System Intel","Live Status · API Ref",False),
            ("📖","User Guide","Walkthroughs · FAQ",False),
        ]
        rows = [modules_overview[i:i+4] for i in range(0, len(modules_overview), 4)]
        for row in rows:
            cols = st.columns(len(row))
            for idx, (icon, name, desc, novel) in enumerate(row):
                bc = "rgba(255,51,85,0.3)" if novel else "rgba(0,180,255,0.15)"
                nc = "#ff3355" if novel else "#00b4ff"
                with cols[idx]:
                    st.markdown(f"""
                    <div style="background:rgba(0,8,20,0.8);border:1px solid {bc};border-radius:10px;
                         padding:1rem;text-align:center;margin-bottom:0.75rem;">
                        <div style="font-size:1.4rem;margin-bottom:0.3rem;">{icon}</div>
                        <div style="font-family:'Rajdhani',sans-serif;font-size:0.7rem;font-weight:700;
                             letter-spacing:0.08em;color:{nc};margin-bottom:0.25rem;">{name}</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;color:#3a6080;line-height:1.4;">{desc}</div>
                    </div>""", unsafe_allow_html=True)

    elif step == 3:
        st.markdown("""
        <div style="text-align:center;padding:2rem 1rem 1rem;">
            <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;letter-spacing:0.2em;color:#ff3355;margin-bottom:0.5rem;">5 NOVEL ALGORITHMS</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#3a6080;letter-spacing:0.2em;margin-bottom:2rem;">NO OPEN-SOURCE EQUIVALENTS · IUB AI RESEARCH LAB · 2025</div>
        </div>""", unsafe_allow_html=True)
        algos = [
            ("📊","TMS v1.0","Threat Momentum Score",
             "Like compound interest for threat — each bad signal adds more than the last.",
             "TMS(t) = TMS(t-1) × decay + Σ(signal × weight × amplifier)","CLEAR → LOW → MEDIUM → HIGH → CRITICAL"),
            ("🧬","BDF v1.0","Behavioral DNA Fingerprint",
             "Identifies same person across cameras using HOW they move — no face needed.",
             "BDF = [gait(10) + velocity(10) + spatial(64) + social_dist(1) + dwell(64)]","cosine similarity > 0.82 → SAME PERSON"),
            ("🕸️","SGI v1.0","Social Graph Intelligence",
             "Detects who is with whom purely from movement — no prior info needed.",
             "Link = proximity(0.40) + velocity_corr(0.35) + dwell_overlap(0.25)","BFS group detection → coordinated movement alert"),
            ("🚀","PEV v1.0","Predictive Exit Vector",
             "Predicts which exit and how many seconds — 3 to 5 seconds before it happens.",
             "trajectory → boundary_intersect() → confidence = stability × proximity × depth","LEFT/RIGHT/TOP/BOTTOM · ETA seconds · camera handoff"),
            ("🛰️","ZIE v1.0","Zone Intelligence Engine",
             "Defines named zones on a frame — restricted, monitored, capacity-limited, safe — with live breach alerts.",
             "breach → proximity_violation signal → TMS escalation","RESTRICTED/MONITORED/CAPACITY/SAFE · CRITICAL breach alerts"),
        ]
        for icon, tag, name, simple, formula, output in algos:
            st.markdown(f"""
            <div style="background:rgba(0,4,12,0.95);border:1px solid rgba(255,51,85,0.2);border-left:3px solid #ff3355;
                 border-radius:8px;padding:1.25rem 1.5rem;margin-bottom:1rem;">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.6rem;">
                    <span style="font-size:1.4rem;">{icon}</span>
                    <span style="font-family:'Rajdhani',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;
                         color:#ff3355;background:rgba(255,51,85,0.1);border:1px solid rgba(255,51,85,0.3);
                         padding:2px 8px;border-radius:3px;">{tag}</span>
                    <span style="font-family:'Exo 2',sans-serif;font-size:1rem;font-weight:700;color:#e8f4ff;">{name}</span>
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#00b4ff;margin-bottom:0.4rem;">"{simple}"</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#3a6080;background:rgba(0,0,0,0.3);
                     border-radius:4px;padding:5px 10px;margin-bottom:0.35rem;">{formula}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#00ff88;">→ {output}</div>
            </div>""", unsafe_allow_html=True)

    elif step == 4:
        st.markdown("""
        <div style="text-align:center;padding:2rem 1rem 1rem;">
            <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;letter-spacing:0.2em;color:#00ff88;margin-bottom:0.5rem;">QUICK START — 5 MINUTES</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#3a6080;letter-spacing:0.2em;margin-bottom:2rem;">FOLLOW THESE STEPS TO RUN YOUR FIRST ANALYSIS</div>
        </div>""", unsafe_allow_html=True)
        steps_qs = [
            ("01","Person Detection","Upload any JPG/PNG image → instant bounding boxes + confidence per person. < 2 seconds on CPU."),
            ("02","Behavioral Analytics","Upload any MP4 video → click RUN → heatmap + dwell times + loitering alerts. 15s video ≈ 30s processing."),
            ("03","Weapon Detection","Upload any image → scans 9 weapon classes → THREAT DETECTED or scene clear. < 1 second."),
            ("04","NL Query — Roman Urdu","Type: 'log jo loiter kar rahy thy aur angry thy' → extracted filters + matched subjects."),
            ("05","Try TMS → Trigger CRITICAL","Threat Score → Person 1 → check Loitering + emotion angry + Restricted Zone → click UPDATE 4 times → CRITICAL fires."),
            ("06","Try Anonymization","Anonymize → pick a mode → upload an image → see persons blurred/pixelated while analytics stay intact."),
        ]
        for num, title, desc in steps_qs:
            st.markdown(f"""
            <div style="background:rgba(0,8,20,0.8);border:1px solid rgba(0,180,255,0.1);border-radius:10px;
                 padding:1rem 1.4rem;margin-bottom:0.75rem;display:flex;gap:1rem;">
                <div style="font-family:'Exo 2',sans-serif;font-size:2rem;font-weight:900;color:rgba(0,180,255,0.2);
                     line-height:1;flex-shrink:0;min-width:44px;">{num}</div>
                <div>
                    <div style="font-family:'Rajdhani',sans-serif;font-size:0.82rem;font-weight:700;
                         color:#00b4ff;letter-spacing:0.1em;margin-bottom:0.3rem;">{title}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#7ab3d4;line-height:1.6;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    elif step == 5:
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem 2rem;">
            <div style="font-size:4rem;margin-bottom:1.5rem;filter:drop-shadow(0 0 30px rgba(0,255,136,0.8));">✓</div>
            <div style="font-family:'Exo 2',sans-serif;font-size:2rem;font-weight:900;letter-spacing:0.2em;color:#00ff88;margin-bottom:0.75rem;">SYSTEM READY</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#7ab3d4;letter-spacing:0.15em;margin-bottom:3rem;">
                All 15 modules loaded · 5 novel algorithms active · CPU optimized · Open Access</div>
            <div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;max-width:500px;margin:0 auto 2rem;">
                <div style="flex:1;min-width:110px;background:rgba(0,8,20,0.8);border:1px solid rgba(0,255,136,0.2);border-radius:8px;padding:1rem;text-align:center;">
                    <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;color:#00ff88;">15</div>
                    <div style="font-size:0.55rem;color:#3a6080;font-family:'IBM Plex Mono',monospace;letter-spacing:0.1em;">MODULES</div>
                </div>
                <div style="flex:1;min-width:110px;background:rgba(0,8,20,0.8);border:1px solid rgba(255,51,85,0.2);border-radius:8px;padding:1rem;text-align:center;">
                    <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;color:#ff3355;">5</div>
                    <div style="font-size:0.55rem;color:#3a6080;font-family:'IBM Plex Mono',monospace;letter-spacing:0.1em;">NOVEL ALGOS</div>
                </div>
                <div style="flex:1;min-width:110px;background:rgba(0,8,20,0.8);border:1px solid rgba(0,180,255,0.2);border-radius:8px;padding:1rem;text-align:center;">
                    <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;color:#00b4ff;">CPU</div>
                    <div style="font-size:0.55rem;color:#3a6080;font-family:'IBM Plex Mono',monospace;letter-spacing:0.1em;">NO GPU</div>
                </div>
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#3a6080;">
                Access the <span style="color:#00b4ff;">GUIDE</span> module anytime from the navigation for walkthroughs, deep dives, and API reference.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_l:
        if step > 1:
            if st.button("← BACK", key="welcome_back"):
                st.session_state.welcome_step -= 1
                st.rerun()
    with col_m:
        labels = ["","WHAT IS PHANTOMEYE","15 MODULES","5 NOVEL ALGORITHMS","QUICK START","SYSTEM READY"]
        st.markdown(f'<div style="text-align:center;font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;color:#3a6080;letter-spacing:0.15em;padding-top:0.65rem;">STEP {step} OF 5 · {labels[step]}</div>', unsafe_allow_html=True)
    with col_r:
        if step < 5:
            if st.button("NEXT →", key="welcome_next"):
                st.session_state.welcome_step += 1
                st.rerun()
        else:
            if st.button("ENTER SYSTEM →", key="welcome_enter", type="primary"):
                st.session_state.page = "home"
                st.session_state.first_visit_done = True
                st.rerun()
    if step < 5:
        st.markdown("<br>", unsafe_allow_html=True)
        cs = st.columns([3, 1])
        with cs[1]:
            if st.button("Skip Guide", key="welcome_skip"):
                st.session_state.page = "home"
                st.session_state.first_visit_done = True
                st.rerun()


def zone_page():
    render_session_bar()
    back_button()
    from core.zone_intelligence import ZoneIntelligenceEngine, ZoneType, AlertLevel

    st.markdown('<div class="section-hdr red">Zone Intelligence Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Novel · ZIE v1.0 · Restricted zones · Capacity limits · Breach alerts · TMS integration</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <strong>Research contribution:</strong> ZIE v1.0 defines named surveillance zones
        directly on the video frame. Four zone types:
        <span style="color:#ff3355;font-weight:600;">RESTRICTED</span> (no entry — CRITICAL alert),
        <span style="color:#f0b429;font-weight:600;">MONITORED</span> (entry logged),
        <span style="color:#00fff0;font-weight:600;">CAPACITY LIMITED</span> (max occupancy enforced),
        <span style="color:#00ff88;font-weight:600;">SAFE</span> (normal zone).
        Every breach automatically feeds the <strong>proximity_violation</strong> signal into
        TMS v1.0 — escalating the threat score of the offending person in real time.
        Suspicious zone traversal sequences are also detected automatically.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="terminal">ZIE v1.0 · 4 zone types · CRITICAL breach alerts · capacity enforcement · TMS integration · suspicious sequence detection</div>', unsafe_allow_html=True)

    if "zie_engine" not in st.session_state:
        st.session_state.zie_engine = ZoneIntelligenceEngine()
    engine = st.session_state.zie_engine

    # ── Zone Definition ──────────────────────────────────
    st.markdown("### Define Zones")
    st.markdown('<div class="section-sub">Add zones by specifying name, type, and pixel coordinates on a 640×480 frame</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        zone_name = st.text_input("Zone Name", value="Server Room", key="zie_name")
        zone_type = st.selectbox("Zone Type", ["RESTRICTED", "MONITORED", "CAPACITY_LIMITED", "SAFE"], key="zie_type")
    with c2:
        zx1 = st.number_input("X1 (left)", min_value=0, max_value=639, value=400, key="zie_x1")
        zy1 = st.number_input("Y1 (top)", min_value=0, max_value=479, value=100, key="zie_y1")
    with c3:
        zx2 = st.number_input("X2 (right)", min_value=1, max_value=640, value=600, key="zie_x2")
        zy2 = st.number_input("Y2 (bottom)", min_value=1, max_value=480, value=350, key="zie_y2")
        max_cap = st.number_input("Max Capacity", min_value=1, max_value=50, value=5, key="zie_cap")

    col_add, col_clear = st.columns(2)
    with col_add:
        if st.button("ADD ZONE", type="primary", key="zie_add"):
            zt = ZoneType(zone_type)
            engine.add_zone(zone_name, zt, zx1, zy1, zx2, zy2, max_capacity=max_cap)
            st.success(f"Zone '{zone_name}' added. Total zones: {len(engine.zones)}")
    with col_clear:
        if st.button("CLEAR ALL ZONES", key="zie_clear"):
            engine.clear_zones()
            st.success("All zones cleared.")

    # ── Preset Scenarios ─────────────────────────────────
    st.markdown("### Quick Presets")
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("🏦 Bank Scenario", key="preset_bank"):
            engine.clear_zones()
            engine.add_zone("Vault",        ZoneType.RESTRICTED,       420, 50,  620, 280)
            engine.add_zone("Counter",      ZoneType.MONITORED,        50,  50,  400, 250)
            engine.add_zone("Lobby",        ZoneType.CAPACITY_LIMITED, 50,  280, 640, 480, max_capacity=10)
            engine.add_zone("Exit",         ZoneType.SAFE,             600, 280, 640, 480)
            st.success("Bank scenario loaded — 4 zones defined.")
    with p2:
        if st.button("🏥 Hospital Scenario", key="preset_hospital"):
            engine.clear_zones()
            engine.add_zone("ICU",          ZoneType.RESTRICTED,       400, 50,  640, 300)
            engine.add_zone("Ward",         ZoneType.MONITORED,        50,  50,  380, 300)
            engine.add_zone("Waiting Area", ZoneType.CAPACITY_LIMITED, 50,  300, 640, 480, max_capacity=8)
            engine.add_zone("Reception",    ZoneType.SAFE,             50,  300, 200, 480)
            st.success("Hospital scenario loaded — 4 zones defined.")
    with p3:
        if st.button("🏢 Office Scenario", key="preset_office"):
            engine.clear_zones()
            engine.add_zone("Server Room",  ZoneType.RESTRICTED,       450, 50,  640, 250)
            engine.add_zone("CEO Office",   ZoneType.MONITORED,        200, 50,  430, 250)
            engine.add_zone("Open Floor",   ZoneType.CAPACITY_LIMITED, 50,  250, 640, 480, max_capacity=15)
            engine.add_zone("Corridor",     ZoneType.SAFE,             50,  50,  180, 480)
            st.success("Office scenario loaded — 4 zones defined.")

    # ── Active Zones Display ─────────────────────────────
    if engine.zones:
        st.markdown("### Active Zones")
        type_colors = {
            "RESTRICTED":       "#ff3355",
            "MONITORED":        "#f0b429",
            "CAPACITY_LIMITED": "#00fff0",
            "SAFE":             "#00ff88",
        }
        for zid, zone in engine.zones.items():
            color = type_colors.get(zone.zone_type.value, "#00b4ff")
            occ   = sum(1 for s in engine.person_states.values() if zid in s.current_zones)
            st.markdown(f"""
            <div style="background:rgba(0,8,20,0.8);border:1px solid {color}40;border-left:3px solid {color};
                 border-radius:6px;padding:0.6rem 1rem;margin-bottom:0.5rem;
                 font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                 display:flex;align-items:center;gap:1.5rem;">
                <span style="color:{color};font-weight:700;min-width:20px;">[{zid}]</span>
                <span style="color:#e8f4ff;font-weight:600;min-width:140px;">{zone.name}</span>
                <span style="color:{color};font-size:0.62rem;min-width:140px;">{zone.zone_type.value}</span>
                <span style="color:#3a6080;font-size:0.62rem;">BBox: ({zone.x1},{zone.y1}) → ({zone.x2},{zone.y2})</span>
                <span style="color:#00ff88;font-size:0.62rem;margin-left:auto;">Occupancy: {occ}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Person Simulation ────────────────────────────────
    st.markdown("### Simulate Person Movement")
    st.markdown('<div class="section-sub">Feed person positions frame by frame to test zone detection</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        sim_pid = st.number_input("Person ID", min_value=1, value=1, key="zie_pid")
        sim_x1  = st.number_input("BBox X1", min_value=0, max_value=639, value=410, key="zie_sx1")
    with s2:
        sim_y1  = st.number_input("BBox Y1", min_value=0, max_value=479, value=110, key="zie_sy1")
        sim_x2  = st.number_input("BBox X2", min_value=1, max_value=640, value=460, key="zie_sx2")
    with s3:
        sim_y2  = st.number_input("BBox Y2", min_value=1, max_value=480, value=210, key="zie_sy2")

    if st.button("FEED FRAME", type="primary", key="zie_feed"):
        if not engine.zones:
            st.warning("Define at least one zone first — or use a preset above.")
        else:
            dets   = [{"person_id": sim_pid, "bbox": [sim_x1, sim_y1, sim_x2, sim_y2]}]
            result = engine.update(dets)
            st.session_state.zie_last_result = result

    if "zie_last_result" in st.session_state:
        result = st.session_state.zie_last_result
        events = result.get("events", [])

        if events:
            st.markdown("### Events This Frame")
            level_colors = {
                "CRITICAL": "#ff3355",
                "HIGH":     "#f0b429",
                "MEDIUM":   "#00b4ff",
                "LOW":      "#00ff88",
                "NONE":     "#3a6080",
            }
            for evt in events:
                color = level_colors.get(evt["alert_level"], "#3a6080")
                icon  = "🔴" if evt["alert_level"] == "CRITICAL" else \
                        "🟠" if evt["alert_level"] == "HIGH"     else \
                        "🟡" if evt["alert_level"] == "MEDIUM"   else \
                        "🟢" if evt["alert_level"] == "LOW"      else "⚪"
                st.markdown(f"""
                <div style="background:rgba(0,4,12,0.95);border:1px solid {color}40;
                     border-left:3px solid {color};border-radius:6px;
                     padding:0.75rem 1rem;margin-bottom:0.5rem;
                     font-family:'IBM Plex Mono',monospace;">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="font-size:1.1rem;">{icon}</span>
                        <span style="color:{color};font-size:0.65rem;font-weight:700;
                             min-width:80px;">{evt['alert_level']}</span>
                        <span style="color:#7ab3d4;font-size:0.72rem;">{evt['message']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if result.get("tms_signals"):
                st.markdown(f'<div class="terminal" style="color:#ff3355;margin-top:0.5rem;">⚡ TMS SIGNAL FIRED: proximity_violation → person(s) {list(result["tms_signals"].keys())} threat score boosted</div>', unsafe_allow_html=True)
        else:
            st.info("No events this frame — person is not inside any defined zone.")

    # ── Zone Summaries ───────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Zone Summaries")
    summaries = engine.get_all_summaries()
    if summaries:
        import pandas as pd
        rows = []
        for zid, s in summaries.items():
            rows.append({
                "Zone": s["zone_name"],
                "Type": s["zone_type"],
                "Entries": s["total_entries"],
                "Exits": s["total_exits"],
                "Occupancy": s["current_occupancy"],
                "Avg Dwell (s)": s["avg_dwell_sec"],
                "Breaches": s["total_breaches"],
                "Alert": s["alert_level"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No zone data yet. Add zones and feed frames.")

    # ── Event Log ────────────────────────────────────────
    st.markdown("### Recent Event Log")
    recent = engine.get_recent_events(15)
    if recent:
        import pandas as pd
        df = pd.DataFrame([{
            "Event":  e["event_type"].upper(),
            "Zone":   e["zone_name"],
            "Person": e["person_id"],
            "Alert":  e["alert_level"],
            "Message": e["message"],
        } for e in recent])
        st.dataframe(df, use_container_width=True)

    # ── Session Summary ──────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    summary = engine.session_summary()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("ZONES DEFINED",    summary["total_zones"])
    s2.metric("PERSONS TRACKED",  summary["total_persons_seen"])
    s3.metric("TOTAL ENTRIES",    summary["total_entries"])
    s4.metric("TOTAL BREACHES",   summary["total_breaches"])

    if st.button("RESET ENGINE", key="zie_reset"):
        engine.reset()
        for key in ["zie_last_result"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("ZIE engine reset.")


def anon_page():
    render_session_bar()
    back_button()
    from core.anonymizer import get_engine, AnonMode

    st.markdown('<div class="section-hdr">Anonymization Mode</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">ANE v1.0 · GDPR compliant · 5 modes · analytics preserved</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <strong>How it works:</strong> ANE v1.0 makes persons unidentifiable while preserving
        all behavioral analytics — tracking, heatmap, dwell time, and zone intelligence
        continue working normally. Five anonymization modes:
        <span style="color:#00b4ff;">Face Blur</span> ·
        <span style="color:#00b4ff;">Face Pixelate</span> ·
        <span style="color:#00b4ff;">Full Body Blur</span> ·
        <span style="color:#00b4ff;">Full Body Pixelate</span> ·
        <span style="color:#00b4ff;">Silhouette</span>.
        Designed for enterprise deployments requiring GDPR compliance.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="terminal">ANE v1.0 · 5 modes · Gaussian blur · pixelation · silhouette · GDPR compliant · analytics intact</div>', unsafe_allow_html=True)

    engine = get_engine()

    c1, c2 = st.columns(2)
    with c1:
        mode = st.selectbox("Anonymization Mode", [
            AnonMode.FACE_BLUR,
            AnonMode.FACE_PIXELATE,
            AnonMode.FULL_BLUR,
            AnonMode.FULL_PIXELATE,
            AnonMode.SILHOUETTE,
        ], key="anon_mode")
    with c2:
        intensity = st.slider("Blur Intensity", min_value=5, max_value=99, value=25, step=2, key="anon_intensity")

    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="anon_upload")

    if uploaded:
        from PIL import Image
        import numpy as np

        img       = Image.open(uploaded).convert("RGB")
        frame     = np.array(img)
        frame_bgr = frame[:, :, ::-1].copy()

        detector   = load_detector()
        detections = detector.detect(frame_bgr)
        dets       = [{"bbox": d["bbox"]} for d in detections]

        result     = engine.anonymize_image(frame_bgr, dets, mode=mode, intensity=intensity)
        anon_frame = engine.draw_anon_overlay(result.frame, result)

        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, caption="Original", use_container_width=True)
        with col2:
            st.image(anon_frame[:, :, ::-1], caption=f"Anonymized — {mode}", use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PERSONS FOUND",   result.persons_found)
        c2.metric("FACES BLURRED",   result.faces_blurred)
        c3.metric("BODIES BLURRED",  result.bodies_blurred)
        c4.metric("MODE",            result.mode.upper())
        st.markdown(f'<div class="terminal">ANONYMIZED · Mode: {mode} · Intensity: {intensity} · Persons: {result.persons_found} · Analytics: INTACT</div>', unsafe_allow_html=True)
    else:
        st.info("Upload an image to begin anonymization.")


def guide_page():
    """User Guide & Documentation."""
    render_session_bar()
    back_button()
    st.markdown('<div class="section-hdr">PhantomEye User Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Complete documentation · module walkthroughs · novel algorithm deep dives · API reference · FAQ</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>How to use this guide:</strong> Your complete reference for PhantomEye v3.5. Use the tabs below to navigate Quick Start, individual module walkthroughs, novel algorithm deep dives with math, API reference, and FAQ. You can also re-run the 5-step onboarding tour anytime from Quick Start.</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚡ Quick Start",
        "📦 All Modules",
        "🔬 Novel Algorithms",
        "🔌 API Reference",
        "❓ FAQ"
    ])

    # ── TAB 1 — QUICK START ──────────────────────────────
    with tab1:
        st.markdown('<div class="section-hdr">Quick Start — 5 Minutes to First Analysis</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Modules", "15")
        c2.metric("Novel Algorithms", "5")
        c3.metric("GPU Required", "None")
        st.markdown("<br>", unsafe_allow_html=True)
        qs = [
            ("🎯","01","Person Detection","Go to Detection → Upload any JPG/PNG → instant bounding boxes + confidence per person.","Annotated image · person count · bbox coordinates. < 2 seconds."),
            ("🔥","02","Behavioral Analytics","Go to Analytics → Upload any MP4 video → click RUN BEHAVIORAL ANALYSIS → wait for processing.","Heatmap · dwell times · loitering alerts if any person stayed > 60s."),
            ("⚠️","03","Weapon Detection","Go to Weapon → Upload any image → auto-scans 9 weapon classes.","THREAT DETECTED with class + confidence, OR scene clear confirmation."),
            ("💬","04","NL Query — Roman Urdu","Go to NL Query → type: 'log jo loiter kar rahy thy aur angry thy' → press Enter.","Extracted filters + matched subjects from sample dataset."),
            ("📊","05","Trigger TMS CRITICAL","Threat Score → ID=1 → Loitering ✓ · emotion=angry · Restricted Zone ✓ → click UPDATE 4 times.","Score climbs CLEAR→CRITICAL. Red alert fires at HIGH+."),
            ("🕶️","06","Try Anonymization","Anonymize → pick a mode (Face Blur, Pixelate, Silhouette) → upload an image.","Persons anonymized in output image while detection counts stay intact."),
        ]
        for icon, num, title, how, expect in qs:
            with st.expander(f"{icon} Step {num} — {title}"):
                st.markdown(f'<div class="terminal">{how}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="terminal" style="color:#00ff88;margin-top:0.5rem;">✓ Expected: {expect}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺  RE-RUN ONBOARDING TOUR", key="rerun_tour"):
            st.session_state.welcome_step = 1
            st.session_state.first_visit_done = False
            st.session_state.page = "welcome"
            st.rerun()

    # ── TAB 2 — ALL MODULES ─────────────────────────────
    with tab2:
        st.markdown('<div class="section-hdr">All 15 Modules — Step-by-Step Walkthroughs</div>', unsafe_allow_html=True)
        mods = [
            ("🎯","Person Detection","YOLOv8-nano · CPU · Class 0",False,
             "Detects every person in an uploaded image. Returns bounding boxes and confidence scores per person. 100% CPU — no GPU.",
             ["Click DETECTION in navigation","Upload any JPG or PNG image","Wait < 2 seconds for inference","View annotated image with bounding boxes","Expand Detection Log for raw coordinates"],
             "JPG · PNG · Any size","Annotated image · person count · confidence · bbox coords",
             "Works best with clearly visible people. Min recommended size: 50×50px.",
             "yolov8n.pt · CPU · class 0 only · confidence threshold: 0.4"),
            ("🔥","Behavioral Analytics","ByteTrack · Heatmap · Dwell",False,
             "Multi-object tracking with persistent IDs, behavioral heatmap, dwell time per person, loitering alerts.",
             ["Click ANALYTICS","Upload MP4/AVI/MOV video","Review video metadata","Click RUN BEHAVIORAL ANALYSIS","Wait (15s video ≈ 30s processing)","View heatmap — red = highest activity","Check metrics: persons · avg/max dwell · alerts"],
             "MP4 · AVI · MOV · up to 15s analyzed","Heatmap · dwell times · loitering alerts with IDs",
             "15s video cap. Longer videos auto-truncated.",
             "ByteTrack IOU · NumPy heatmap · loitering threshold: 60s"),
            ("🕵️","OSINT Audit","LBPH Face · Gallery",False,
             "Upload face photo → Privacy Exposure Score 0–100. Matched against gallery. Risk: LOW/MEDIUM/HIGH.",
             ["Click OSINT","Upload clear face photo (JPG/PNG)","Click EXECUTE AUDIT","View score 0–100 and risk level","Expand Match Log for gallery matches"],
             "Clear face photo · JPG/PNG","Exposure score · risk level · match log · visualization",
             "Use front-facing clear photo. Add gallery faces via API: POST /osint/add-to-gallery.",
             "OpenCV LBPH · cosine distance · no data retained"),
            ("🧠","Emotion Intelligence","DeepFace · TF · 7 Classes",False,
             "Multi-face emotion analysis — dominant emotion, age estimate, gender per face.",
             ["Click EMOTION","Upload image with faces","Wait 10–30s first run (TF model loading)","View original vs annotated side by side","Check per-subject: emotion · age · gender"],
             "JPG · PNG · any image with faces","Dominant emotion · age estimate · gender · annotated image",
             "First run is slow (TF loading). Subsequent runs faster. Works best with front-facing lit faces.",
             "DeepFace + TF · OpenCV detector · 7 classes · 15% min face size filter"),
            ("💬","NL Query Engine","Groq LLaMA 3 · Roman Urdu",False,
             "Type surveillance queries in English or Roman Urdu. LLaMA 3 extracts structured filters.",
             ["Click NL QUERY","Type any query (English or Roman Urdu)","Press Enter","View extracted filters","Check matched subjects"],
             "English or Roman Urdu text","Extracted filters · matched subjects",
             "Try: 'show me angry men who were loitering' OR 'log jo loiter kar rahy thy aur jinka emotion angry tha'",
             "llama-3.1-8b-instant via Groq · JSON output · apply_filters() engine"),
            ("⚠️","Weapon Detection","YOLOv8 Custom · 9 Classes",False,
             "Custom YOLOv8 on 714 weapon images. 9 classes. Immediate threat alert.",
             ["Click WEAPON","Upload any image","Wait < 1 second","View original vs annotated","THREAT DETECTED alert OR scene clear"],
             "JPG · PNG · any image","THREAT alert OR clear · weapon class · confidence · annotated",
             "9 classes: Handgun · Knife · Shotgun · Sniper · AR · SMG · Sword · Bazooka · GL",
             "weapon_detector.pt · mAP50 53.2% · Handgun 89.5% · Shotgun 96.3% · SMG 98.6%"),
            ("📊","Threat Momentum Score","NOVEL · TMS v1.0",True,
             "Compound interest model for threat. Continuous score with 5 levels. No binary yes/no.",
             ["Click THREAT SCORE in Row 2","Enter Person ID","Set signals: dwell · emotion · loitering · restricted zone","Click UPDATE THREAT SCORE","Repeat to see compound accumulation","Try: all boxes + angry → CRITICAL"],
             "Person ID · position · emotion · dwell · boolean flags","TMS score · threat level · signal breakdown · momentum",
             "Amplifier effect: when TMS is HIGH, new signals contribute MORE. Update same person 5–6 times to see exponential growth.",
             "TMS(t) = TMS(t-1) × 0.5^(Δt/45s) + Σ(signal × weight × (1 + TMS/200))"),
            ("🧬","Behavioral DNA","NOVEL · BDF v1.0",True,
             "Re-identifies person across cameras using behavioral signature only. No face needed.",
             ["Click BEHAVIORAL DNA","Set Person ID + position","Simulate Observations (30+)","Click REGISTER TO GALLERY","Change position slightly (simulate re-entry)","Simulate more observations","Click MATCH AGAINST GALLERY → MATCH FOUND"],
             "Person ID · position · social distance · observations","BDF vector · similarity % · match result · explanation",
             "Needs 15+ observations for reliable fingerprint. More observations = higher confidence.",
             "5 components: gait(10) + velocity(10) + spatial(64) + social_dist(1) + dwell(64) · cosine · threshold 82%"),
            ("🕸️","Social Graph","NOVEL · SGI v1.0",True,
             "Detects who is together from movement alone. No prior information needed.",
             ["Click SOCIAL GRAPH","Person 1: X=100, moveX=3 → simulate 50 frames","Person 2: X=120, moveX=3 → simulate 50 frames (same direction)","Person 3: X=500, moveX=-2 → simulate 50 frames (opposite)","Click DETECT GROUPS","Expected: P1+P2 grouped, P3 separate"],
             "Person ID · start position · movement vector · frames","Social links · groups · cohesion · coordinated alerts",
             "Same direction + speed = linked. Bank robbery: 3 enter separately but converge → SGI flags before action.",
             "Link = proximity(0.40) + pearson_corr(0.35) + dwell_overlap(0.25) · BFS group detection"),
            ("🚀","Predictive Exit Vector","NOVEL · PEV v1.0",True,
             "Predicts frame boundary exit 3–5s ahead. Camera handoff use case.",
             ["Click PREDICTIVE EXIT","Person ID=1, BBox X1=400 (near right edge)","Click AUTO-SIMULATE →RIGHT (20 frames)","View: EXIT SIDE=RIGHT · ETA · Confidence","Watch ALERT fire when ETA < 2s + conf > 0.4"],
             "Person ID · bbox position · simulation frames","Exit side · ETA seconds · confidence · alert · trajectory",
             "Needs 6+ frames before prediction is reliable. Confidence builds as more frames fed.",
             "Velocity smoothing: 5 frames · horizon: 4s · confidence = stability × proximity × depth"),
            ("🛰️","Zone Intelligence","NOVEL · ZIE v1.0",True,
             "Defines named surveillance zones with breach detection and TMS integration.",
             ["Click ZONE INTEL","Use a preset (Bank/Hospital/Office) or define your own zones","Simulate a person walking into a RESTRICTED zone","Click FEED FRAME","Watch CRITICAL breach event + TMS signal fire"],
             "Zone coordinates · zone type · person bbox per frame","Breach events · zone summaries · occupancy · TMS signal boosts",
             "RESTRICTED zone entry always fires CRITICAL. Try the Bank preset for a fast demo.",
             "4 zone types · capacity enforcement · proximity_violation → TMS · suspicious sequence detection"),
            ("🕶️","Anonymization","NOVEL · ANE v1.0",True,
             "GDPR-compliant anonymization — blurs or pixelates persons while preserving analytics.",
             ["Click ANONYMIZE","Pick a mode: Face Blur, Face Pixelate, Full Blur, Full Pixelate, or Silhouette","Adjust Blur Intensity slider","Upload any image with people","View side-by-side original vs anonymized"],
             "Image with visible persons · mode · intensity","Anonymized image · persons found · faces/bodies blurred count",
             "Use Silhouette mode for the strongest privacy guarantee while keeping pose data usable.",
             "ANE v1.0 · Haar cascade face detection · 5 modes · analytics fully preserved · GDPR compliant"),
            ("📄","Intel Report","fpdf2 · Classified PDF",False,
             "Generate classified PDF report. Dark theme, CLASSIFIED header, threat sections in red.",
             ["Click REPORT","Fill session metadata","Set subjects with behavioral data","Check weapon detected if applicable","Click GENERATE PDF REPORT","Download immediately"],
             "Session data · subjects · weapon detections","Downloadable classified PDF · dark theme · red threat sections",
             "PDF generated in-memory. Nothing stored server-side. Each PDF unique to your session.",
             "fpdf2 · dark bg #020408 · green text #00ff88 · red threat #ff3355"),
            ("⚡","System Intel","Live Status",False,
             "Dashboard: all modules, tech stack, benchmarks, API endpoints, deployment info.",
             ["Click SYSTEM","View all 15 module statuses","Expand any module for tech specs","View full tech stack JSON"],
             "No input required","Module registry · benchmarks · tech stack · deployment info",
             "Use to verify system health and get API endpoint URLs for integration.",
             "v3.5.0 · HuggingFace Spaces Docker · FastAPI OAS 3.1 · Python 3.10"),
            ("📖","User Guide","This Module",False,
             "Complete interactive documentation. Quick Start, walkthroughs, algo deep dives, API ref, FAQ.",
             ["Click GUIDE in Row 2","Use tabs to navigate sections","Re-run onboarding tour from Quick Start tab","Bookmark for reference"],
             "No input required","Full documentation · walkthroughs · API reference · FAQ",
             "This is a living document. Check back after each upgrade for updated walkthroughs.",
             "Interactive Streamlit tabs · 5-step tour · re-runnable onboarding"),
        ]
        for icon, name, tag, novel, what, steps_list, inp, out, tip, tech in mods:
            border = "rgba(255,51,85,0.3)" if novel else "rgba(0,180,255,0.1)"
            lc = "#ff3355" if novel else "#00b4ff"
            with st.expander(f"{icon} {name}  ·  {tag}"):
                if novel:
                    st.markdown('<div style="display:inline-block;background:rgba(255,51,85,0.1);border:1px solid rgba(255,51,85,0.3);border-radius:3px;font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;color:#ff3355;padding:2px 8px;letter-spacing:0.15em;margin-bottom:0.75rem;">⬡ NOVEL RESEARCH ALGORITHM</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="terminal">{what}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-hdr">Step-by-Step</div>', unsafe_allow_html=True)
                for idx, s in enumerate(steps_list, 1):
                    st.markdown(f'<div style="display:flex;gap:10px;padding:5px 0;border-bottom:1px solid rgba(0,180,255,0.05);"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:{lc};min-width:24px;font-weight:700;">{idx:02d}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#7ab3d4;">{s}</span></div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                ca, cb = st.columns(2)
                ca.markdown(f'<div class="terminal">INPUT: {inp}</div>', unsafe_allow_html=True)
                cb.markdown(f'<div class="terminal">OUTPUT: {out}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-box" style="margin-top:0.75rem;"><strong>Tip:</strong> {tip}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="terminal" style="margin-top:0.5rem;">TECH: {tech}</div>', unsafe_allow_html=True)

    # ── TAB 3 — NOVEL ALGORITHMS ─────────────────────────
    with tab3:
        st.markdown('<div class="section-hdr red">5 Novel Algorithms — Deep Dive</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Original research · IUB AI Research Lab · No open-source equivalents · IEEE Access target</div>', unsafe_allow_html=True)
        algo_data = [
            ("📊","TMS v1.0","Threat Momentum Score",
             "Unlike binary threat detection systems that output yes/no, TMS treats threat as continuous momentum — like compound interest. Every signal adds to a running score that decays over time, but amplifies when already elevated.",
             "TMS(t) = TMS(t-1) × decay_factor + Σ(signal × weight × amplifier)\n\namplifier    = 1 + (TMS / 200)\ndecay_factor = 0.5^(Δt / 45s)   ← 45 second half-life",
             "6 Behavioral Signals",
             [("loitering","0.28","Person stays in one area beyond threshold"),
              ("stress_emotion","0.22","Angry, fear, or disgust detected"),
              ("rapid_movement","0.18","Sudden velocity spike"),
              ("proximity_violation","0.15","Too close to restricted zone"),
              ("gaze_anomaly","0.10","Unusual gaze pattern"),
              ("group_formation","0.07","Part of flagged group (SGI output)")],
             "CLEAR (0–20) → LOW (20–50) → MEDIUM (50–100) → HIGH (100–180) → CRITICAL (180+)",
             "Person escalates CLEAR → CRITICAL across 5 frames with compound signals ✓"),
            ("🧬","BDF v1.0","Behavioral DNA Fingerprint",
             "Traditional Re-ID requires face recognition which fails with masks, hats, or distance. BDF identifies people using HOW they move — a behavioral fingerprint unique to each individual that persists across cameras.",
             "BDF_vector = [\n  gait_signature(10),      ← stride rhythm histogram\n  velocity_profile(10),    ← speed distribution\n  spatial_preference(8×8), ← normalized grid heatmap\n  social_distance_avg(1),  ← preferred distance from others\n  dwell_zone_signature(64) ← stopping location pattern\n]\n\nMatch: cosine(BDF_a, BDF_b) > 0.82 → SAME PERSON",
             "5 Behavioral Components",
             [("Gait Signature","10 features","Stride rhythm and cadence pattern"),
              ("Velocity Profile","10 features","Speed distribution across movement"),
              ("Spatial Preference","64 features","Where person tends to stand on grid"),
              ("Social Distance","1 feature","Average preferred distance from others"),
              ("Dwell Zones","64 features","Which locations person stops at")],
             "< 0.70 = NO MATCH · 0.70–0.82 = UNCERTAIN · > 0.82 = MATCH",
             "Person 3 matched Person 1 at 99.99% after re-entry with new tracking ID ✓"),
            ("🕸️","SGI v1.0","Social Graph Intelligence",
             "Three bank robbers can enter a building separately and behave normally individually — but SGI detects their association before any overt action by analyzing movement correlation. No face recognition or prior info needed.",
             "Link_strength = proximity(0.40) + pearson_velocity_corr(0.35) + dwell_overlap(0.25)\n\nGroup detection: BFS connected-component analysis\nAlert: coordinated link_type in group ≥ 2 persons",
             "3 Association Signals",
             [("Proximity Score","0.40 weight","How often persons are within 150px of each other"),
              ("Velocity Correlation","0.35 weight","Pearson correlation — do they accelerate together?"),
              ("Dwell Zone Overlap","0.25 weight","Do they stop at the same locations?")],
             "0.0–0.3 = STRANGERS · 0.3–0.6 = ACQUAINTANCES · 0.6+ = ASSOCIATED",
             "Person 1↔2 strength 0.570 (associated), Person 1↔3 strength 0.309 (strangers) ✓"),
            ("🚀","PEV v1.0","Predictive Exit Vector",
             "Existing systems only alert WHEN someone leaves. PEV predicts WHERE they will exit and WHEN — 3 to 5 seconds before it happens. Enables camera handoff: Camera B activates before person leaves Camera A.",
             "1. Position history → deque(maxlen=15 frames)\n2. Smoothed velocity: sliding window avg (5 frames)\n3. Trajectory: step forward vx/vy × max_frames\n4. Boundary hit: first frame where x≤0, x≥W, y≤0, y≥H\n5. Confidence = stability × proximity × depth\n   - stability: 1 - CoV(velocity magnitudes)\n   - proximity: distance to nearest boundary\n   - depth:     history_frames / 15",
             "Confidence Components",
             [("Velocity Stability","50% weight","Is direction consistent? Low variance = high confidence"),
              ("Boundary Proximity","30% weight","How close is person to frame edge?"),
              ("History Depth","20% weight","How many frames of data available?")],
             "conf < 0.4 = LOW · 0.4–0.7 = MEDIUM · > 0.7 = HIGH · ALERT: ETA < 2s AND conf > 0.4",
             "All 4 directions correct · stationary = NONE · 3-person simultaneous ✓"),
            ("🛰️","ZIE v1.0","Zone Intelligence Engine",
             "Surveillance systems traditionally treat the whole frame as one undifferentiated space. ZIE lets an operator define named zones with distinct rules — a vault is not a lobby — and ties zone breaches directly into the TMS threat pipeline so a restricted-zone entry immediately escalates that person's threat score.",
             "Zone membership: bbox-center inside [x1,y1,x2,y2]\nOn RESTRICTED entry → AlertLevel.CRITICAL + TMS signal: proximity_violation\nOn CAPACITY_LIMITED breach → occupancy > max_capacity → AlertLevel.HIGH\nSuspicious sequence: rapid traversal across ≥3 zones in <N frames",
             "4 Zone Types",
             [("RESTRICTED","CRITICAL alert","No entry permitted — immediate TMS escalation"),
              ("MONITORED","logged only","Entry/exit recorded, no automatic alert"),
              ("CAPACITY_LIMITED","HIGH if exceeded","Occupancy enforced against max_capacity"),
              ("SAFE","informational","Normal zone, baseline dwell tracking")],
             "NONE → LOW → MEDIUM → HIGH → CRITICAL (zone-specific alert ladder)",
             "Bank preset: Vault breach fires CRITICAL + boosts TMS proximity_violation signal ✓"),
        ]
        for icon, tag, name, concept, formula, sig_title, sigs, levels, result in algo_data:
            st.markdown(f"""
            <div style="background:rgba(0,4,12,0.95);border:1px solid rgba(255,51,85,0.15);border-top:2px solid #ff3355;
                 border-radius:10px;padding:1.5rem;margin-bottom:1.5rem;">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
                    <span style="font-size:1.8rem;">{icon}</span>
                    <span style="font-family:'Rajdhani',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;
                         color:#ff3355;background:rgba(255,51,85,0.1);border:1px solid rgba(255,51,85,0.3);
                         padding:2px 8px;border-radius:3px;">{tag}</span>
                    <span style="font-family:'Exo 2',sans-serif;font-size:1.2rem;font-weight:900;color:#e8f4ff;">{name}</span>
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.73rem;color:#7ab3d4;line-height:1.8;">{concept}</div>
            </div>""", unsafe_allow_html=True)
            with st.expander(f"📐 Formula + {sig_title}"):
                st.code(formula, language="python")
                st.markdown(f'<div class="section-hdr">{sig_title}</div>', unsafe_allow_html=True)
                for sn, sw, sd in sigs:
                    st.markdown(f'<div style="display:flex;gap:12px;padding:5px 0;border-bottom:1px solid rgba(0,180,255,0.05);"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#ff3355;min-width:140px;font-weight:700;">{sn}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#00b4ff;min-width:80px;">{sw}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#5a8090;">{sd}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="terminal" style="margin-top:0.75rem;">LEVELS: {levels}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="terminal" style="color:#00ff88;">RESULT: {result}</div>', unsafe_allow_html=True)

    # ── TAB 4 — API REFERENCE ───────────────────────────
    with tab4:
        st.markdown('<div class="section-hdr">API Reference — FastAPI OAS 3.1</div>', unsafe_allow_html=True)
        st.markdown('<div class="terminal">Base URL: https://abu-sameer-66-phantomeye.hf.space · Docs: /docs · OpenAPI: /openapi.json</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        endpoints = [
            ("GET",  "/",                           "Root",               "System info · version · module list"),
            ("GET",  "/health",                     "Health Check",       "Status · gallery size · timestamp"),
            ("POST", "/detect",                     "Person Detection",   "Upload image → bbox + confidence per person"),
            ("POST", "/osint/audit",                "OSINT Audit",        "Upload face → exposure score + matches"),
            ("POST", "/osint/add-to-gallery",       "Add to Gallery",     "Upload face + person_id → add to gallery"),
            ("GET",  "/osint/gallery",              "Gallery List",       "All person IDs in OSINT gallery"),
            ("POST", "/track/video",                "Video Tracking",     "Upload video → tracking summary"),
            ("GET",  "/outputs",                    "List Outputs",       "All generated output files"),
            ("POST", "/api/predictive-exit/update", "PEV Update",         "Feed frame detections → exit predictions"),
            ("GET",  "/api/predictive-exit/status", "PEV Status",         "Active track count + engine info"),
            ("POST", "/api/predictive-exit/reset",  "PEV Reset",          "Clear all tracked persons"),
            ("POST", "/api/zone-intelligence/zone", "ZIE Add Zone",       "Define a new surveillance zone"),
            ("POST", "/api/zone-intelligence/update","ZIE Update",        "Feed frame detections → zone events"),
            ("GET",  "/api/zone-intelligence/summary","ZIE Summary",      "Per-zone occupancy and breach summary"),
        ]
        mc = {"GET":"#00ff88","POST":"#00b4ff","DELETE":"#ff3355"}
        for method, path, name, desc in endpoints:
            color = mc.get(method, "#7ab3d4")
            st.markdown(f'<div style="display:flex;align-items:center;gap:12px;padding:7px 0;border-bottom:1px solid rgba(0,180,255,0.06);"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;font-weight:700;color:{color};min-width:46px;">{method}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#00b4ff;min-width:260px;">{path}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.63rem;color:#7ab3d4;min-width:140px;">{name}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;color:#3a6080;">{desc}</span></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-hdr">Quick Integration Example</div>', unsafe_allow_html=True)
        st.code("""import requests

BASE = "https://abu-sameer-66-phantomeye.hf.space"

# Person Detection
with open("image.jpg", "rb") as f:
    resp = requests.post(f"{BASE}/detect", files={"file": f})
    print(f"Found {resp.json()['total_persons']} persons")

# Predictive Exit
payload = {
    "frame_id": 42, "frame_width": 640, "frame_height": 480, "fps": 25.0,
    "detections": [{"person_id": 1, "bbox": {"x1": 550, "y1": 200, "x2": 610, "y2": 320}}]
}
resp = requests.post(f"{BASE}/api/predictive-exit/update", json=payload)
for p in resp.json()["predictions"]:
    print(f"Person {p['person_id']} → {p['exit_side']} in {p['seconds_to_exit']:.1f}s")
""", language="python")

    # ── TAB 5 — FAQ ─────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-hdr">Frequently Asked Questions</div>', unsafe_allow_html=True)
        faqs = [
            ("Why is PhantomEye slow on first load?",
             "DeepFace loads TensorFlow models on first use — 15–30 seconds on free HuggingFace tier. Subsequent requests are faster. If it times out, wait 30s and retry."),
            ("Does PhantomEye require a GPU?",
             "No. 100% CPU-optimized. YOLOv8-nano, OSNet Re-ID, ByteTrack, and all 15 modules run on CPU only. Designed for standard hardware including Dell Vostro AMD Ryzen."),
            ("Can I use a live RTSP camera?",
             "Not yet. Current version supports uploaded video files (MP4/AVI/MOV). RTSP live stream support is planned for a future phase."),
            ("How do I add faces to the OSINT gallery?",
             "Use API endpoint: POST /osint/add-to-gallery with face image + person_id parameter. View current gallery via GET /osint/gallery."),
            ("Why does NL Query sometimes fail?",
             "NL Query uses Groq API (LLaMA 3). Set GROQ_API_KEY in .env file. Free tier has rate limits — wait a few minutes if you hit them."),
            ("What video formats are supported?",
             "MP4, AVI, MOV. H.264 encoded MP4 works best. Max analyzed duration: 15 seconds. Recommended resolution: 720p or lower."),
            ("How accurate is weapon detection?",
             "mAP50 53.2% overall. Per-class: Handgun 89.5%, Shotgun 96.3%, SMG 98.6%, AR 94.2%. Trained on 714 real weapon images across 9 classes."),
            ("Can I run PhantomEye locally?",
             "Yes. Clone from github.com/Abu-Sameer-66/PhantomEye · conda env Python 3.10 · pip install -r requirements.txt · streamlit run app.py (port 7860) + uvicorn api.main:app (port 8000)."),
            ("What is the difference between BDF and standard face Re-ID?",
             "Standard Re-ID fails with masks, hats, distance, low resolution. BDF uses only behavioral patterns — walk style, speed, preferred locations. Works fully disguised."),
            ("Does anonymization affect detection accuracy?",
             "No. ANE v1.0 runs after detection — bounding boxes, tracking IDs, dwell times, and zone events are computed first, then the visual output is anonymized. All numeric analytics remain fully intact."),
            ("How do I report bugs or contribute?",
             "Open an issue at github.com/Abu-Sameer-66/PhantomEye. For research collaboration contact via sameer-nadeem-portfolio.vercel.app."),
        ]
        for q, a in faqs:
            with st.expander(f"Q: {q}"):
                st.markdown(f'<div class="terminal">{a}</div>', unsafe_allow_html=True)


def main():
    if "page"             not in st.session_state:
        st.session_state.page = "landing"
    if "session_id"       not in st.session_state:
        st.session_state.session_id = "PE-" + str(uuid.uuid4())[:8].upper()
    if "first_visit_done" not in st.session_state:
        st.session_state.first_visit_done = False
    if "welcome_step"     not in st.session_state:
        st.session_state.welcome_step = 1

    page = st.session_state.page

    if   page == "landing":   landing()
    elif page == "welcome":   welcome_flow()
    elif page == "home":      home()
    elif page == "DETECTION": detection_page()
    elif page == "ANALYTICS": analytics_page()
    elif page == "OSINT":     osint_page()
    elif page == "EMOTION":   emotion_page()
    elif page == "NL QUERY":  nlquery_page()
    elif page == "WEAPON":    weapon_page()
    elif page == "THREAT":    threat_page()
    elif page == "BDF":       bdf_page()
    elif page == "SGI":       sgi_page()
    elif page == "PEV":       pev_page()
    elif page == "REPORT":    report_page()
    elif page == "INTEL":     intel_page()
    elif page == "ZONE":      zone_page()
    elif page == "ANON":      anon_page()
    elif page == "GUIDE":     guide_page()


if __name__ == "__main__":
    main()
    
    
    
    
