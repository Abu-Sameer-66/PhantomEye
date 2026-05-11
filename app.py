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
#     --bg-secondary: #050d15;
#     --bg-card: rgba(6, 18, 32, 0.85);
#     --bg-glass: rgba(0, 180, 255, 0.04);
#     --accent-blue: #00b4ff;
#     --accent-cyan: #00fff0;
#     --accent-amber: #ffb300;
#     --accent-red: #ff3355;
#     --accent-green: #00ff88;
#     --border-glow: rgba(0, 180, 255, 0.35);
#     --border-subtle: rgba(0, 180, 255, 0.12);
#     --text-primary: #e8f4ff;
#     --text-secondary: #7ab3d4;
#     --text-dim: #3a6080;
#     --grid-color: rgba(0, 180, 255, 0.035);
#     --shadow-blue: 0 0 50px rgba(0, 180, 255, 0.18);
#     --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.7);
# }

# *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

# html, body, [class*="css"] {
#     font-family: 'IBM Plex Mono', monospace;
#     background: var(--bg-primary) !important;
#     color: var(--text-primary) !important;
# }

# /* ── TOP ACCENT BAR ──────────────────────────────── */
# .stApp::after {
#     content: '';
#     position: fixed;
#     top: 0; left: 0; right: 0;
#     height: 2px;
#     background: linear-gradient(90deg,
#         transparent 0%,
#         var(--accent-blue) 20%,
#         var(--accent-cyan) 50%,
#         var(--accent-blue) 80%,
#         transparent 100%);
#     z-index: 9999;
#     animation: top-bar-glow 4s ease-in-out infinite alternate;
# }

# @keyframes top-bar-glow {
#     from { opacity: 0.6; }
#     to   { opacity: 1; filter: brightness(1.4); }
# }

# .stApp {
#     background:
#         radial-gradient(ellipse at 15% 40%, rgba(0, 60, 130, 0.18) 0%, transparent 55%),
#         radial-gradient(ellipse at 85% 15%, rgba(0, 30, 90, 0.22) 0%, transparent 50%),
#         radial-gradient(ellipse at 50% 85%, rgba(0, 40, 110, 0.12) 0%, transparent 60%),
#         linear-gradient(180deg, #020408 0%, #030a14 100%) !important;
#     min-height: 100vh;
# }

# .stApp::before {
#     content: '';
#     position: fixed;
#     top: 0; left: 0; right: 0; bottom: 0;
#     background-image:
#         linear-gradient(var(--grid-color) 1px, transparent 1px),
#         linear-gradient(90deg, var(--grid-color) 1px, transparent 1px);
#     background-size: 60px 60px;
#     pointer-events: none;
#     z-index: 0;
# }

# /* ── SESSION BAR ─────────────────────────────────── */
# .session-bar {
#     display: flex;
#     justify-content: space-between;
#     align-items: center;
#     background: rgba(0, 10, 20, 0.7);
#     border: 1px solid var(--border-subtle);
#     border-radius: 6px;
#     padding: 0.45rem 1.2rem;
#     margin-bottom: 1.8rem;
#     font-family: 'IBM Plex Mono', monospace;
#     font-size: 0.68rem;
#     backdrop-filter: blur(12px);
# }

# .session-bar .sid   { color: var(--text-secondary); }
# .session-bar .sid span { color: var(--accent-blue); }
# .session-bar .status { color: var(--accent-green); letter-spacing: 0.2em; }
# .session-bar .status::before { content: '● '; animation: blink 1.5s infinite; }
# .session-bar .badge {
#     font-family: 'Rajdhani', sans-serif;
#     font-size: 0.6rem; font-weight: 700;
#     letter-spacing: 0.25em; text-transform: uppercase;
#     color: var(--accent-cyan);
#     background: rgba(0,255,240,0.08);
#     border: 1px solid rgba(0,255,240,0.3);
#     border-radius: 4px;
#     padding: 0.15rem 0.6rem;
# }

# /* ── HERO / LANDING ─────────────────────────────── */
# .hero-wrap {
#     display: flex; flex-direction: column;
#     align-items: center; justify-content: center;
#     min-height: 92vh; padding: 3rem 1rem;
#     position: relative;
# }

# .hero-wrap::before {
#     content: '';
#     position: absolute;
#     width: 700px; height: 700px;
#     background: radial-gradient(circle, rgba(0, 180, 255, 0.07) 0%, transparent 70%);
#     border-radius: 50%;
#     top: 50%; left: 50%;
#     transform: translate(-50%, -50%);
#     animation: pulse-glow 5s ease-in-out infinite;
# }

# @keyframes pulse-glow {
#     0%, 100% { transform: translate(-50%, -50%) scale(1);   opacity: 0.5; }
#     50%       { transform: translate(-50%, -50%) scale(1.12); opacity: 1; }
# }

# .hero-eye {
#     font-size: 5.5rem; margin-bottom: 1.5rem;
#     animation: float 5s ease-in-out infinite;
#     filter: drop-shadow(0 0 40px rgba(0, 180, 255, 0.9));
# }

# @keyframes float {
#     0%, 100% { transform: translateY(0px) rotate(-2deg); }
#     50%       { transform: translateY(-22px) rotate(2deg); }
# }

# .hero-title {
#     font-family: 'Exo 2', sans-serif;
#     font-size: clamp(3.5rem, 8vw, 7.5rem);
#     font-weight: 900;
#     letter-spacing: 0.15em;
#     background: linear-gradient(135deg, #ffffff 0%, var(--accent-blue) 40%, var(--accent-cyan) 100%);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     background-clip: text;
#     margin-bottom: 0.5rem;
#     animation: title-reveal 1s ease-out forwards;
# }

# @keyframes title-reveal {
#     from { opacity: 0; transform: translateY(30px); }
#     to   { opacity: 1; transform: translateY(0); }
# }

# .hero-sub {
#     font-family: 'Rajdhani', sans-serif;
#     font-size: clamp(0.85rem, 2vw, 1.05rem);
#     font-weight: 300;
#     letter-spacing: 0.45em;
#     color: var(--text-secondary);
#     margin-bottom: 0.5rem;
#     text-transform: uppercase;
# }

# .hero-status {
#     font-size: 0.7rem;
#     color: var(--accent-green);
#     letter-spacing: 0.3em;
#     margin-bottom: 3rem;
# }

# .hero-status::before {
#     content: '● ';
#     animation: blink 1.5s infinite;
# }

# @keyframes blink {
#     0%, 100% { opacity: 1; }
#     50%       { opacity: 0.15; }
# }

# /* ── STATS ROW ───────────────────────────────────── */
# .stats-row {
#     display: flex; gap: 2rem; margin-bottom: 2.5rem;
#     justify-content: center; flex-wrap: wrap;
# }

# .stat-item {
#     text-align: center;
#     background: var(--bg-card);
#     border: 1px solid var(--border-subtle);
#     border-radius: 10px;
#     padding: 1rem 1.8rem;
#     backdrop-filter: blur(16px);
#     min-width: 110px;
# }

# .stat-value {
#     font-family: 'Exo 2', sans-serif;
#     font-size: 1.6rem; font-weight: 900;
#     color: var(--accent-blue);
#     display: block;
# }

# .stat-label {
#     font-size: 0.62rem; letter-spacing: 0.25em;
#     color: var(--text-dim); text-transform: uppercase;
#     margin-top: 0.2rem; display: block;
# }

# /* ── MODULE GRID ─────────────────────────────────── */
# .module-grid {
#     display: grid;
#     grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
#     gap: 1.4rem;
#     width: 100%;
#     max-width: 1100px;
#     margin: 0 auto 3rem;
# }

# .mod-card {
#     background: var(--bg-card);
#     border: 1px solid var(--border-subtle);
#     border-radius: 14px;
#     padding: 2rem 1.8rem;
#     position: relative;
#     overflow: hidden;
#     transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
#     backdrop-filter: blur(20px);
# }

# .mod-card::before {
#     content: '';
#     position: absolute;
#     top: 0; left: 0; right: 0;
#     height: 2px;
#     background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), transparent);
#     opacity: 0;
#     transition: opacity 0.3s;
# }

# .mod-card::after {
#     content: '';
#     position: absolute;
#     inset: 0;
#     background: radial-gradient(ellipse at top left, rgba(0,180,255,0.1) 0%, transparent 65%);
#     opacity: 0;
#     transition: opacity 0.4s;
# }

# .mod-card:hover {
#     border-color: var(--border-glow);
#     transform: translateY(-7px);
#     box-shadow: var(--shadow-blue), var(--shadow-card);
# }

# .mod-card:hover::before { opacity: 1; }
# .mod-card:hover::after  { opacity: 1; }

# .mod-icon { font-size: 2rem; margin-bottom: 1rem; display: block; }
# .mod-name {
#     font-family: 'Rajdhani', sans-serif;
#     font-size: 0.95rem; font-weight: 600;
#     letter-spacing: 0.2em; color: var(--accent-blue);
#     text-transform: uppercase; margin-bottom: 0.8rem;
# }

# .mod-tag {
#     display: inline-block;
#     font-size: 0.58rem; letter-spacing: 0.18em;
#     color: var(--accent-cyan);
#     background: rgba(0,255,240,0.07);
#     border: 1px solid rgba(0,255,240,0.2);
#     border-radius: 3px; padding: 0.1rem 0.5rem;
#     margin-bottom: 0.7rem; text-transform: uppercase;
# }

# .mod-desc {
#     font-size: 0.76rem;
#     color: var(--text-secondary);
#     line-height: 1.75;
#     letter-spacing: 0.02em;
# }

# /* ── SCAN LINE ───────────────────────────────────── */
# .scan-line {
#     width: 100%; max-width: 900px;
#     height: 1px;
#     background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), var(--accent-blue), transparent);
#     margin: 2rem auto;
#     position: relative; overflow: hidden;
# }

# .scan-line::after {
#     content: '';
#     position: absolute;
#     width: 80px; height: 100%;
#     background: linear-gradient(90deg, transparent, rgba(0,255,240,0.9), transparent);
#     animation: scan 3s linear infinite;
# }

# @keyframes scan {
#     from { left: -80px; }
#     to   { left: 100%; }
# }

# /* ── APP HEADER ──────────────────────────────────── */
# .app-header {
#     font-family: 'Exo 2', sans-serif;
#     font-size: 1.7rem; font-weight: 700;
#     letter-spacing: 0.3em;
#     background: linear-gradient(135deg, #fff, var(--accent-blue));
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     background-clip: text;
#     text-align: center;
#     padding: 1.5rem 0 0.4rem;
# }

# .app-sub {
#     font-family: 'Rajdhani', sans-serif;
#     font-size: 0.72rem;
#     color: var(--text-dim);
#     letter-spacing: 0.45em;
#     text-align: center;
#     margin-bottom: 2rem;
#     text-transform: uppercase;
# }

# /* ── MODULE NAV BUTTONS ──────────────────────────── */
# .stButton > button {
#     font-family: 'Rajdhani', sans-serif !important;
#     font-weight: 600 !important;
#     letter-spacing: 0.12em !important;
#     font-size: 0.8rem !important;
#     background: var(--bg-card) !important;
#     color: var(--accent-blue) !important;
#     border: 1px solid var(--border-subtle) !important;
#     border-radius: 8px !important;
#     padding: 0.7rem 1rem !important;
#     transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1) !important;
#     position: relative !important;
#     overflow: hidden !important;
#     text-transform: uppercase !important;
#     width: 100% !important;
# }

# .stButton > button:hover {
#     background: rgba(0, 180, 255, 0.1) !important;
#     border-color: var(--accent-blue) !important;
#     color: var(--accent-cyan) !important;
#     box-shadow: 0 0 25px rgba(0, 180, 255, 0.25),
#                 inset 0 0 20px rgba(0, 180, 255, 0.05) !important;
#     transform: translateY(-2px) !important;
# }

# .stButton > button[kind="primary"] {
#     background: linear-gradient(135deg, rgba(0,100,200,0.4), rgba(0,180,255,0.2)) !important;
#     border-color: var(--accent-blue) !important;
#     color: #fff !important;
#     box-shadow: 0 0 25px rgba(0,180,255,0.25) !important;
# }

# /* ── SECTION HEADERS ─────────────────────────────── */
# .section-hdr {
#     font-family: 'Exo 2', sans-serif;
#     font-size: 1.25rem; font-weight: 700;
#     letter-spacing: 0.25em; color: var(--accent-blue);
#     text-transform: uppercase;
#     padding: 0.5rem 0;
#     border-bottom: 1px solid var(--border-subtle);
#     margin-bottom: 0.5rem;
#     position: relative;
# }

# .section-hdr::after {
#     content: '';
#     position: absolute;
#     bottom: -1px; left: 0;
#     width: 90px; height: 2px;
#     background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
# }

# .section-sub {
#     font-size: 0.73rem; color: var(--text-secondary);
#     letter-spacing: 0.15em; margin-bottom: 2rem;
#     text-transform: uppercase;
# }

# /* ── TERMINAL STATUS BAR ─────────────────────────── */
# .terminal {
#     background: rgba(0,10,20,0.92);
#     border: 1px solid var(--border-subtle);
#     border-left: 3px solid var(--accent-blue);
#     border-radius: 6px;
#     padding: 0.8rem 1.2rem;
#     font-size: 0.72rem; color: var(--accent-green);
#     letter-spacing: 0.15em; margin-top: 1.5rem;
#     position: relative; overflow: hidden;
# }

# .terminal::before {
#     content: '';
#     position: absolute; inset: 0;
#     background: repeating-linear-gradient(
#         0deg,
#         transparent, transparent 2px,
#         rgba(0,255,136,0.012) 2px, rgba(0,255,136,0.012) 4px
#     );
#     pointer-events: none;
# }

# /* ── STREAMLIT OVERRIDES ─────────────────────────── */
# .stFileUploader {
#     background: var(--bg-card) !important;
#     border: 1px dashed var(--border-glow) !important;
#     border-radius: 10px !important;
#     padding: 1rem !important;
# }

# .stTextInput > div > div {
#     background: var(--bg-card) !important;
#     border: 1px solid var(--border-subtle) !important;
#     border-radius: 8px !important;
#     color: var(--text-primary) !important;
#     font-family: 'IBM Plex Mono', monospace !important;
# }

# .stTextInput > div > div:focus-within {
#     border-color: var(--accent-blue) !important;
#     box-shadow: 0 0 15px rgba(0,180,255,0.15) !important;
# }

# .stSelectbox > div > div {
#     background: var(--bg-card) !important;
#     border: 1px solid var(--border-subtle) !important;
#     border-radius: 8px !important;
#     color: var(--text-primary) !important;
# }

# .stNumberInput > div > div {
#     background: var(--bg-card) !important;
#     border: 1px solid var(--border-subtle) !important;
#     border-radius: 8px !important;
# }

# .stSlider > div > div > div { background: var(--accent-blue) !important; }

# div[data-testid="metric-container"] {
#     background: var(--bg-card) !important;
#     border: 1px solid var(--border-subtle) !important;
#     border-radius: 10px !important;
#     padding: 1rem !important;
#     transition: border-color 0.3s;
# }

# div[data-testid="metric-container"]:hover {
#     border-color: var(--border-glow) !important;
# }

# div[data-testid="metric-container"] label {
#     color: var(--text-secondary) !important;
#     font-size: 0.68rem !important;
#     letter-spacing: 0.2em !important;
#     font-family: 'Rajdhani', sans-serif !important;
#     font-weight: 600 !important;
# }

# div[data-testid="metric-container"] div[data-testid="metric-value"] {
#     color: var(--accent-blue) !important;
#     font-family: 'Exo 2', sans-serif !important;
#     font-weight: 700 !important;
# }

# div[data-testid="stDataFrame"] {
#     background: var(--bg-card) !important;
#     border: 1px solid var(--border-subtle) !important;
#     border-radius: 10px !important;
#     overflow: hidden !important;
# }

# .stSuccess {
#     background: rgba(0,255,136,0.07) !important;
#     border: 1px solid rgba(0,255,136,0.28) !important;
#     border-radius: 8px !important;
#     color: var(--accent-green) !important;
# }

# .stError, .stWarning {
#     background: rgba(255,51,85,0.07) !important;
#     border: 1px solid rgba(255,51,85,0.28) !important;
#     border-radius: 8px !important;
# }

# .stInfo {
#     background: rgba(0,180,255,0.07) !important;
#     border: 1px solid rgba(0,180,255,0.2) !important;
#     border-radius: 8px !important;
#     color: var(--accent-blue) !important;
# }

# hr { border-color: var(--border-subtle) !important; margin: 1.5rem 0 !important; }

# ::-webkit-scrollbar { width: 4px; }
# ::-webkit-scrollbar-track { background: var(--bg-primary); }
# ::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 2px; opacity: 0.5; }

# .stSpinner > div {
#     border-color: var(--accent-blue) transparent transparent transparent !important;
# }

# section[data-testid="stSidebar"] { display: none !important; }
# #MainMenu { visibility: hidden; }
# footer    { visibility: hidden; }
# header    { visibility: hidden; }

# @keyframes fadeInUp {
#     from { opacity: 0; transform: translateY(20px); }
#     to   { opacity: 1; transform: translateY(0); }
# }

# .stMarkdown, .stButton, .stFileUploader {
#     animation: fadeInUp 0.4s ease-out forwards;
# }
# </style>
# """, unsafe_allow_html=True)


# # ─────────────────────────────────────────────────────
# #  CACHED LOADERS
# # ─────────────────────────────────────────────────────
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


# # ─────────────────────────────────────────────────────
# #  SESSION BAR  (replaces trust bar — no limits)
# # ─────────────────────────────────────────────────────
# def render_session_bar():
#     sid = st.session_state.get("session_id", "PE-XXXXXXXX")
#     st.markdown(f"""
#     <div class="session-bar">
#         <div class="sid"><span>●</span> &nbsp;SESSION: <span>{sid}</span></div>
#         <div class="status">ALL SYSTEMS ONLINE</div>
#         <div class="badge">OPEN ACCESS</div>
#     </div>
#     """, unsafe_allow_html=True)


# # ─────────────────────────────────────────────────────
# #  LANDING PAGE
# # ─────────────────────────────────────────────────────
# def landing():
#     st.markdown("""
#     <div class="hero-wrap">
#       <div class="hero-eye">👁</div>
#       <div class="hero-title">PHANTOMEYE</div>
#       <div class="hero-sub">AI-POWERED SURVEILLANCE INTELLIGENCE SYSTEM</div>
#       <div class="hero-status">[ SYSTEM ONLINE ] · OPEN ACCESS · BUILD v3.0</div>

#       <div class="stats-row">
#         <div class="stat-item">
#           <span class="stat-value">8</span>
#           <span class="stat-label">Modules</span>
#         </div>
#         <div class="stat-item">
#           <span class="stat-value">97%</span>
#           <span class="stat-label">Accuracy</span>
#         </div>
#         <div class="stat-item">
#           <span class="stat-value">9</span>
#           <span class="stat-label">Weapon Classes</span>
#         </div>
#         <div class="stat-item">
#           <span class="stat-value">CPU</span>
#           <span class="stat-label">No GPU Needed</span>
#         </div>
#       </div>

#       <div class="scan-line"></div>

#       <div class="module-grid">
#         <div class="mod-card">
#           <div class="mod-icon">🎯</div>
#           <div class="mod-name">Person Detection</div>
#           <div class="mod-tag">YOLOv8-nano</div>
#           <div class="mod-desc">Detects every person in any image with confidence scores and bounding boxes in real-time.</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">🔥</div>
#           <div class="mod-name">Behavioral Analytics</div>
#           <div class="mod-tag">ByteTrack · OpenCV</div>
#           <div class="mod-desc">Live heatmap, persistent ID tracking, dwell time and automated loitering alerts from video.</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">🕵️</div>
#           <div class="mod-name">OSINT Audit</div>
#           <div class="mod-tag">LBPH Face Recognition</div>
#           <div class="mod-desc">Upload a face — get a privacy exposure score 0–100 with gallery-matched identities.</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">🧠</div>
#           <div class="mod-name">Emotion Intelligence</div>
#           <div class="mod-tag">DeepFace · TensorFlow</div>
#           <div class="mod-desc">Age, gender and dominant emotion recognition on any face image with confidence scores.</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">💬</div>
#           <div class="mod-name">NL Query Engine</div>
#           <div class="mod-tag">Groq LLaMA 3</div>
#           <div class="mod-desc">Ask in plain English or Roman Urdu — AI extracts filters and matches subjects instantly.</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">⚠️</div>
#           <div class="mod-name">Weapon Detection</div>
#           <div class="mod-tag">YOLOv8 Custom · 9 Classes</div>
#           <div class="mod-desc">Handgun 89.5% · Shotgun 96.3% · SMG 98.6% — trained on 714 real-world images.</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">📄</div>
#           <div class="mod-name">Intel Report</div>
#           <div class="mod-tag">fpdf2 · Cyberpunk PDF</div>
#           <div class="mod-desc">One-click classified PDF: session overview, threat alerts, subject log, query history.</div>
#         </div>
#         <div class="mod-card">
#           <div class="mod-icon">⚡</div>
#           <div class="mod-name">System Intel</div>
#           <div class="mod-tag">Live Status</div>
#           <div class="mod-desc">Module health, API endpoints, model benchmarks and full deployment information.</div>
#         </div>
#       </div>
#     </div>
#     """, unsafe_allow_html=True)

#     cols = st.columns([1, 2, 1])
#     with cols[1]:
#         if st.button("INITIALIZE SYSTEM  →", key="enter_btn"):
#             st.session_state.page = "home"
#             st.rerun()


# # ─────────────────────────────────────────────────────
# #  HOME — MODULE SELECTOR
# # ─────────────────────────────────────────────────────
# def home():
#     render_session_bar()
#     st.markdown('<div class="app-header">👁 PHANTOMEYE</div>', unsafe_allow_html=True)
#     st.markdown('<div class="app-sub">SELECT INTELLIGENCE MODULE · ALL SYSTEMS ACTIVE</div>', unsafe_allow_html=True)

#     modules = [
#         ("DETECTION", "🎯 Detection"),
#         ("ANALYTICS", "🔥 Analytics"),
#         ("OSINT",     "🕵️ OSINT"),
#         ("EMOTION",   "🧠 Emotion"),
#         ("NL QUERY",  "💬 NL Query"),
#         ("WEAPON",    "⚠️ Weapon"),
#         ("REPORT",    "📄 Report"),
#         ("INTEL",     "⚡ System"),
#     ]
#     cols = st.columns(len(modules))
#     for i, (key, label) in enumerate(modules):
#         with cols[i]:
#             if st.button(label, key=f"mod_{key}"):
#                 st.session_state.page = key
#                 st.rerun()

#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown(
#         '<div class="terminal">[ PHANTOMEYE v3.0 ] · YOLOv8 loaded · ByteTrack active · '
#         'DeepFace online · Groq LLaMA connected · Weapon model ready · All 8 modules ACTIVE</div>',
#         unsafe_allow_html=True
#     )


# # ─────────────────────────────────────────────────────
# #  SHARED BACK BUTTON
# # ─────────────────────────────────────────────────────
# def back_button():
#     if st.button("← BACK TO MODULES"):
#         st.session_state.page = "home"
#         st.rerun()


# # ─────────────────────────────────────────────────────
# #  MODULE PAGES
# # ─────────────────────────────────────────────────────
# def detection_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">🎯 Person Detection</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">YOLOv8-nano · CPU inference · upload any image</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="terminal">YOLOv8-nano · class 0 person only · '
#         'confidence threshold 0.4 · CPU optimized</div>',
#         unsafe_allow_html=True
#     )

#     uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], key="det_up")

#     if uploaded:
#         data  = np.frombuffer(uploaded.read(), np.uint8)
#         image = cv2.imdecode(data, cv2.IMREAD_COLOR)
#         if image is None:
#             st.error("Cannot decode image.")
#             return

#         with st.spinner("SCANNING..."):
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

#         st.image(
#             cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
#             caption="DETECTION OUTPUT", use_container_width=True
#         )

#         if detections:
#             st.markdown('<div class="section-hdr">Detection Log</div>', unsafe_allow_html=True)
#             for i, d in enumerate(detections):
#                 with st.expander(f"PERSON_{i+1:03d}  ·  CONF: {d['confidence']}"):
#                     st.json({"id": i+1, "bbox": list(d["bbox"]), "confidence": d["confidence"]})


# def analytics_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">🔥 Behavioral Analytics</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">ByteTrack · heatmap · dwell time · loitering alerts</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="terminal">Upload video · persistent ID tracking · '
#         'real-time behavioral heatmap generation</div>',
#         unsafe_allow_html=True
#     )

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

#         st.markdown(
#             f'<div class="terminal">{w}×{h} @ {fps}fps · {total} frames loaded · '
#             f'analysis limit: {min(total, fps*15)} frames</div>',
#             unsafe_allow_html=True
#         )

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
#                     stat.markdown(
#                         f'<div class="terminal">PROCESSING FRAME {i}/{limit} · '
#                         f'ACTIVE PERSONS: {len(active)}</div>',
#                         unsafe_allow_html=True
#                     )

#             cap.release()
#             tmp.unlink(missing_ok=True)
#             prog.progress(100)
#             stat.empty()

#             s = analyzer.summary()
#             st.success("✓ ANALYSIS COMPLETE")

#             c1, c2, c3, c4 = st.columns(4)
#             c1.metric("TOTAL PERSONS",   s.get("total_persons", 0))
#             c2.metric("AVG DWELL TIME",  f"{s.get('avg_dwell_sec', 0)}s")
#             c3.metric("MAX DWELL TIME",  f"{s.get('max_dwell_sec', 0)}s")
#             c4.metric("LOITER ALERTS",   s.get("total_alerts", 0))

#             if s.get("total_alerts", 0) > 0:
#                 st.warning(f"⚠ LOITERING ALERT — Subject IDs: {s.get('loiterers', [])}")

#             heat = analyzer.get_heatmap_overlay(np.zeros((h, w, 3), dtype=np.uint8))
#             st.image(
#                 cv2.cvtColor(heat, cv2.COLOR_BGR2RGB),
#                 caption="BEHAVIORAL HEATMAP — RED = HIGH ACTIVITY ZONE",
#                 use_container_width=True
#             )


# def osint_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">🕵️ OSINT Privacy Audit</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Face upload · privacy exposure score · gallery match · risk report</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="terminal">LBPH embeddings · score 0–100 · '
#         'LOW / MEDIUM / HIGH risk classification</div>',
#         unsafe_allow_html=True
#     )

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

#         with st.spinner("RUNNING OSINT AUDIT..."):
#             result = osint.audit(image, query_id=Path(query_file.name).stem)

#         risk    = result["risk_level"]
#         score   = result["exposure_score"]
#         matches = result["matches"]

#         c1, c2, c3 = st.columns(3)
#         c1.metric("RISK LEVEL",     risk)
#         c2.metric("EXPOSURE SCORE", f"{score}/100")
#         c3.metric("MATCHES FOUND",  len(matches))

#         st.markdown(
#             f'<div class="terminal">{result["message"]}</div>',
#             unsafe_allow_html=True
#         )

#         if matches:
#             st.markdown('<div class="section-hdr">Match Log</div>', unsafe_allow_html=True)
#             for m in matches:
#                 st.markdown(
#                     f'<div class="terminal">MATCH: {m["matched_id"]} · '
#                     f'CONF: {m["confidence"]}% · SOURCE: {m["source"]}</div>',
#                     unsafe_allow_html=True
#                 )

#         vis = osint.visualize(image, result)
#         st.image(
#             cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
#             caption="OSINT VISUALIZATION",
#             use_container_width=True
#         )


# def emotion_page():
#     render_session_bar()
#     process_frame_emotion = load_emotion_model()
#     back_button()
#     st.markdown('<div class="section-hdr">🧠 Emotion Intelligence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">DeepFace · Age · Gender · Dominant Emotion per face</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="terminal">DeepFace + TensorFlow · 15% min face size filter · '
#         'multi-face support</div>',
#         unsafe_allow_html=True
#     )

#     uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

#     if uploaded:
#         from PIL import Image
#         img       = Image.open(uploaded).convert("RGB")
#         frame     = np.array(img)
#         frame_bgr = frame[:, :, ::-1].copy()

#         with st.spinner("ANALYZING FACES..."):
#             annotated, results = process_frame_emotion(frame_bgr)

#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(frame, caption="ORIGINAL", use_container_width=True)
#         with col2:
#             st.image(annotated[:, :, ::-1], caption="EMOTION ANALYSIS", use_container_width=True)

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
#         st.info("Upload a face image to begin emotion analysis.")


# def nlquery_page():
#     render_session_bar()
#     from core.nlquery import parse_nl_query, apply_filters
#     back_button()
#     st.markdown('<div class="section-hdr">💬 NL Query Engine</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Groq LLaMA 3 · English + Roman Urdu · structured filter extraction</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="terminal">llama-3.1-8b-instant · structured JSON extraction · '
#         'apply_filters() on person records</div>',
#         unsafe_allow_html=True
#     )

#     query = st.text_input(
#         "Enter your query",
#         placeholder="e.g.  show me angry men  |  log jo loiter kar rahy thy"
#     )

#     if query:
#         with st.spinner("PARSING QUERY..."):
#             result = parse_nl_query(query)

#         if result['success']:
#             filters = result['filters']
#             st.success(f"✓ Understood: {filters['summary']}")

#             col1, col2, col3 = st.columns(3)
#             col1.metric("EMOTION",   filters['emotion']  or "ANY")
#             col2.metric("GENDER",    filters['gender']   or "ANY")
#             col3.metric("MAX AGE",   filters['max_age']  or "ANY")

#             col4, col5 = st.columns(2)
#             col4.metric("LOITERING", "YES" if filters['loitering'] else "ANY")
#             col5.metric("MIN DWELL", f"{filters['min_dwell_seconds']}s" if filters['min_dwell_seconds'] else "ANY")

#             st.markdown("<hr>")
#             st.markdown('<div class="section-hdr">Simulate Against Sample Data</div>', unsafe_allow_html=True)

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
#                 st.success(f"{len(matched)} subject(s) matched out of {len(sample_records)}")
#                 import pandas as pd
#                 st.dataframe(pd.DataFrame(matched), use_container_width=True)
#             else:
#                 st.warning("No subjects matched this query in sample data.")
#         else:
#             st.error(f"Query parse failed: {result['error']}")
#     else:
#         st.info("Type a query above — English or Roman Urdu both work.")


# def weapon_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">⚠️ Weapon Detection</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">YOLOv8 Custom · 9 classes · Handgun · Knife · Shotgun · SMG · Rifle · Sword</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="terminal">mAP50: 53.2% · Handgun: 89.5% · Shotgun: 96.3% · '
#         'SMG: 98.6% · trained on 714 real-world images</div>',
#         unsafe_allow_html=True
#     )

#     uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

#     if uploaded:
#         from PIL import Image
#         from core.weapon import detect_weapons

#         img       = Image.open(uploaded).convert("RGB")
#         frame     = np.array(img)
#         frame_bgr = frame[:, :, ::-1].copy()
#         model     = load_weapon_model_cached()

#         with st.spinner("SCANNING FOR THREATS..."):
#             annotated, detections = detect_weapons(frame_bgr, model)

#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(frame, caption="ORIGINAL", use_container_width=True)
#         with col2:
#             st.image(annotated[:, :, ::-1], caption="THREAT ANALYSIS", use_container_width=True)

#         st.markdown("<hr>")
#         if detections:
#             st.error(f"⚠ THREAT DETECTED — {len(detections)} weapon(s) found!")
#             st.markdown('<div class="section-hdr">Detected Threats</div>', unsafe_allow_html=True)
#             for d in detections:
#                 c1, c2 = st.columns(2)
#                 c1.metric("WEAPON CLASS", d['class_name'])
#                 c2.metric("CONFIDENCE",   f"{d['confidence']:.0%}")
#         else:
#             st.success("✓ NO WEAPONS DETECTED — Scene clear")
#     else:
#         st.info("Upload an image to scan for weapons.")


# def report_page():
#     render_session_bar()
#     from core.reporter import generate_report
#     back_button()
#     st.markdown('<div class="section-hdr">📄 Intelligence Report</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Branded cyberpunk PDF · CLASSIFIED header · one-click download</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="terminal">fpdf2 · dark bg · green text · threat alerts in red · '
#         'session + subject + query log</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown("### SESSION DATA")
#     col1, col2 = st.columns(2)
#     with col1:
#         session_id     = st.text_input("Session ID",         value=st.session_state.get("session_id", "PE-SESSION-001"))
#         total_persons  = st.number_input("Total Persons",    min_value=0, value=5)
#         duration       = st.number_input("Duration (sec)",   min_value=0, value=300)
#     with col2:
#         loitering_alerts = st.number_input("Loitering Alerts", min_value=0, value=1)
#         nl_query         = st.text_input("NL Query (opt)",   value="")
#         nl_result        = st.text_input("NL Result (opt)",  value="")

#     st.markdown("### DETECTED SUBJECTS")
#     num_subjects = st.slider("Number of subjects", 1, 10, 3)
#     detections = []
#     for i in range(num_subjects):
#         c1, c2, c3, c4, c5, _ = st.columns(6)
#         detections.append({
#             "id":           i + 1,
#             "emotion":      c1.selectbox(f"Emotion {i+1}", ["neutral","angry","happy","sad","fear","surprise"], key=f"em_{i}"),
#             "gender":       c2.selectbox(f"Gender {i+1}",  ["Man","Woman"], key=f"gen_{i}"),
#             "age":          c3.number_input(f"Age {i+1}",  10, 80, 25, key=f"age_{i}"),
#             "dwell_seconds":c4.number_input(f"Dwell {i+1}",0, 600, 60, key=f"dw_{i}"),
#             "loitering":    c5.checkbox(f"Loiter {i+1}",   key=f"lo_{i}"),
#         })

#     st.markdown("### WEAPON DETECTIONS")
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
#             "session_id":       session_id,
#             "total_persons":    total_persons,
#             "duration_seconds": duration,
#             "loitering_alerts": loitering_alerts,
#             "weapon_detections":weapon_detections,
#             "detections":       detections,
#             "heatmap_img":      None,
#             "frame_sample":     None,
#             "nl_query":         nl_query,
#             "nl_result":        nl_result,
#         }
#         with st.spinner("GENERATING CLASSIFIED REPORT..."):
#             path = generate_report(data)

#         with open(path, "rb") as f:
#             pdf_bytes = f.read()

#         st.success("✓ Report generated successfully!")
#         st.download_button(
#             label="⬇ DOWNLOAD PDF REPORT",
#             data=pdf_bytes,
#             file_name=f"phantomeye_report_{session_id}.pdf",
#             mime="application/pdf"
#         )


# def intel_page():
#     render_session_bar()
#     back_button()
#     st.markdown('<div class="section-hdr">⚡ System Intelligence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Module health · API endpoints · model benchmarks</div>', unsafe_allow_html=True)

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("SYSTEM",   "PhantomEye")
#     c2.metric("VERSION",  "v3.0.0")
#     c3.metric("STATUS",   "ONLINE")
#     c4.metric("MODULES",  "8 ACTIVE")

#     st.markdown("<br>", unsafe_allow_html=True)

#     modules_info = [
#         ("DETECTION",  "YOLOv8-nano",    "Person detection · class 0 · confidence 0.4+ · CPU optimized"),
#         ("ANALYTICS",  "ByteTrack",      "Persistent ID tracking · heatmap · dwell time · loitering alerts"),
#         ("OSINT",      "LBPH Face",      "Privacy exposure score 0–100 · gallery match · LOW/MEDIUM/HIGH risk"),
#         ("EMOTION",    "DeepFace + TF",  "Age · Gender · Dominant Emotion · 15% face size filter"),
#         ("NL QUERY",   "Groq LLaMA 3",  "llama-3.1-8b-instant · English + Roman Urdu · JSON filter extraction"),
#         ("WEAPON",     "YOLOv8 Custom", "9 classes · mAP50 53.2% · Handgun 89.5% · Shotgun 96.3% · SMG 98.6%"),
#         ("REPORT",     "fpdf2",          "Dark cyberpunk PDF · CLASSIFIED header · threat + subject + query log"),
#         ("API",        "FastAPI",         "12 endpoints · OAS 3.1 · CORS enabled · Railway deployed"),
#     ]

#     for name, tech, desc in modules_info:
#         with st.expander(f"{'●'} {name}  ·  {tech}  ·  ACTIVE"):
#             st.markdown(f'<div class="terminal">{desc}</div>', unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.json({
#         "author":    "Abu-Sameer-66",
#         "github":    "https://github.com/Abu-Sameer-66/PhantomEye",
#         "huggingface": "https://abu-sameer-66-phantomeye.hf.space",
#         "railway_api": "https://phantomeye-production.up.railway.app",
#         "api_docs":  "https://phantomeye-production.up.railway.app/docs",
#         "stack":     ["Python 3.10", "YOLOv8", "DeepFace", "FastAPI", "Streamlit", "Groq"],
#         "status":    "online",
#         "access":    "open — no auth required",
#     })


# # ─────────────────────────────────────────────────────
# #  MAIN ROUTER
# # ─────────────────────────────────────────────────────
# def main():
#     if "page"       not in st.session_state:
#         st.session_state.page = "landing"
#     if "session_id" not in st.session_state:
#         st.session_state.session_id = "PE-" + str(uuid.uuid4())[:8].upper()

#     page = st.session_state.page

#     if   page == "landing":  landing()
#     elif page == "home":     home()
#     elif page == "DETECTION": detection_page()
#     elif page == "ANALYTICS": analytics_page()
#     elif page == "OSINT":    osint_page()
#     elif page == "EMOTION":  emotion_page()
#     elif page == "NL QUERY": nlquery_page()
#     elif page == "WEAPON":   weapon_page()
#     elif page == "REPORT":   report_page()
#     elif page == "INTEL":    intel_page()


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
    --bg-secondary: #050d15;
    --bg-card: rgba(6, 18, 32, 0.85);
    --accent-blue: #00b4ff;
    --accent-cyan: #00fff0;
    --accent-red: #ff3355;
    --accent-green: #00ff88;
    --border-glow: rgba(0, 180, 255, 0.35);
    --border-subtle: rgba(0, 180, 255, 0.12);
    --text-primary: #e8f4ff;
    --text-secondary: #7ab3d4;
    --text-dim: #3a6080;
    --grid-color: rgba(0, 180, 255, 0.035);
    --shadow-blue: 0 0 50px rgba(0, 180, 255, 0.18);
    --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.7);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,
        transparent 0%, var(--accent-blue) 20%,
        var(--accent-cyan) 50%, var(--accent-blue) 80%, transparent 100%);
    z-index: 9999;
    animation: topbar 4s ease-in-out infinite alternate;
}

@keyframes topbar {
    from { opacity: 0.6; }
    to   { opacity: 1; filter: brightness(1.4); }
}

.stApp {
    background:
        radial-gradient(ellipse at 15% 40%, rgba(0,60,130,0.18) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 15%, rgba(0,30,90,0.22) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 85%, rgba(0,40,110,0.12) 0%, transparent 60%),
        linear-gradient(180deg, #020408 0%, #030a14 100%) !important;
    min-height: 100vh;
}

.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(var(--grid-color) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-color) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none; z-index: 0;
}

.session-bar {
    display: flex; justify-content: space-between; align-items: center;
    background: rgba(0,10,20,0.7); border: 1px solid var(--border-subtle);
    border-radius: 6px; padding: 0.45rem 1.2rem; margin-bottom: 1.8rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    backdrop-filter: blur(12px);
}
.session-bar .sid { color: var(--text-secondary); }
.session-bar .sid span { color: var(--accent-blue); }
.session-bar .status { color: var(--accent-green); letter-spacing: 0.2em; }
.session-bar .status::before { content: '● '; animation: blink 1.5s infinite; }
.session-bar .badge {
    font-family: 'Rajdhani', sans-serif; font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.25em; text-transform: uppercase; color: var(--accent-cyan);
    background: rgba(0,255,240,0.08); border: 1px solid rgba(0,255,240,0.3);
    border-radius: 4px; padding: 0.15rem 0.6rem;
}

.hero-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 92vh; padding: 3rem 1rem; position: relative;
}
.hero-wrap::before {
    content: ''; position: absolute; width: 700px; height: 700px;
    background: radial-gradient(circle, rgba(0,180,255,0.07) 0%, transparent 70%);
    border-radius: 50%; top: 50%; left: 50%;
    transform: translate(-50%,-50%); animation: pulse 5s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { transform: translate(-50%,-50%) scale(1); opacity: 0.5; }
    50%      { transform: translate(-50%,-50%) scale(1.12); opacity: 1; }
}
.hero-eye {
    font-size: 5.5rem; margin-bottom: 1.5rem;
    animation: float 5s ease-in-out infinite;
    filter: drop-shadow(0 0 40px rgba(0,180,255,0.9));
}
@keyframes float {
    0%,100% { transform: translateY(0px) rotate(-2deg); }
    50%      { transform: translateY(-22px) rotate(2deg); }
}
.hero-title {
    font-family: 'Exo 2', sans-serif;
    font-size: clamp(3.5rem, 8vw, 7.5rem); font-weight: 900;
    letter-spacing: 0.15em;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-blue) 40%, var(--accent-cyan) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 0.5rem; animation: reveal 1s ease-out forwards;
}
@keyframes reveal {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: clamp(0.85rem, 2vw, 1.05rem); font-weight: 300;
    letter-spacing: 0.45em; color: var(--text-secondary);
    margin-bottom: 0.5rem; text-transform: uppercase;
}
.hero-status {
    font-size: 0.7rem; color: var(--accent-green);
    letter-spacing: 0.3em; margin-bottom: 3rem;
}
.hero-status::before { content: '● '; animation: blink 1.5s infinite; }
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.15; } }

.stats-row {
    display: flex; gap: 2rem; margin-bottom: 2.5rem;
    justify-content: center; flex-wrap: wrap;
}
.stat-item {
    text-align: center; background: var(--bg-card);
    border: 1px solid var(--border-subtle); border-radius: 10px;
    padding: 1rem 1.8rem; backdrop-filter: blur(16px); min-width: 110px;
}
.stat-value {
    font-family: 'Exo 2', sans-serif; font-size: 1.6rem; font-weight: 900;
    color: var(--accent-blue); display: block;
}
.stat-label {
    font-size: 0.62rem; letter-spacing: 0.25em; color: var(--text-dim);
    text-transform: uppercase; margin-top: 0.2rem; display: block;
}

.module-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.4rem; width: 100%; max-width: 1200px; margin: 0 auto 3rem;
}
.mod-card {
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    border-radius: 14px; padding: 2rem 1.8rem; position: relative;
    overflow: hidden; transition: all 0.4s cubic-bezier(0.23,1,0.32,1);
    backdrop-filter: blur(20px);
}
.mod-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), transparent);
    opacity: 0; transition: opacity 0.3s;
}
.mod-card::after {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at top left, rgba(0,180,255,0.1) 0%, transparent 65%);
    opacity: 0; transition: opacity 0.4s;
}
.mod-card:hover { border-color: var(--border-glow); transform: translateY(-7px); box-shadow: var(--shadow-blue), var(--shadow-card); }
.mod-card:hover::before { opacity: 1; }
.mod-card:hover::after  { opacity: 1; }
.mod-card.research-card { border-color: rgba(255,51,85,0.18); }
.mod-card.research-card:hover { border-color: rgba(255,51,85,0.55); box-shadow: 0 0 50px rgba(255,51,85,0.12), var(--shadow-card); }

.mod-icon { font-size: 2rem; margin-bottom: 1rem; display: block; }
.mod-name {
    font-family: 'Rajdhani', sans-serif; font-size: 0.95rem; font-weight: 600;
    letter-spacing: 0.2em; color: var(--accent-blue);
    text-transform: uppercase; margin-bottom: 0.5rem;
}
.mod-name.red { color: var(--accent-red); }
.mod-tag {
    display: inline-block; font-size: 0.58rem; letter-spacing: 0.18em;
    color: var(--accent-cyan); background: rgba(0,255,240,0.07);
    border: 1px solid rgba(0,255,240,0.2); border-radius: 3px;
    padding: 0.1rem 0.5rem; margin-bottom: 0.7rem; text-transform: uppercase;
}
.mod-tag.red { color: var(--accent-red); background: rgba(255,51,85,0.07); border-color: rgba(255,51,85,0.25); }
.mod-desc { font-size: 0.76rem; color: var(--text-secondary); line-height: 1.75; letter-spacing: 0.02em; }
.mod-meta {
    font-size: 0.63rem; color: var(--text-dim); margin-top: 0.9rem;
    letter-spacing: 0.04em; border-top: 1px solid var(--border-subtle); padding-top: 0.7rem;
}

.scan-line {
    width: 100%; max-width: 900px; height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), var(--accent-blue), transparent);
    margin: 2rem auto; position: relative; overflow: hidden;
}
.scan-line::after {
    content: ''; position: absolute; width: 80px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,255,240,0.9), transparent);
    animation: scan 3s linear infinite;
}
@keyframes scan { from { left: -80px; } to { left: 100%; } }

.app-header {
    font-family: 'Exo 2', sans-serif; font-size: 1.7rem; font-weight: 700;
    letter-spacing: 0.3em;
    background: linear-gradient(135deg, #fff, var(--accent-blue));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    text-align: center; padding: 1.5rem 0 0.4rem;
}
.app-sub {
    font-family: 'Rajdhani', sans-serif; font-size: 0.72rem; color: var(--text-dim);
    letter-spacing: 0.45em; text-align: center; margin-bottom: 2rem; text-transform: uppercase;
}

.stButton > button {
    font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
    letter-spacing: 0.12em !important; font-size: 0.8rem !important;
    background: var(--bg-card) !important; color: var(--accent-blue) !important;
    border: 1px solid var(--border-subtle) !important; border-radius: 8px !important;
    padding: 0.7rem 1rem !important;
    transition: all 0.3s cubic-bezier(0.23,1,0.32,1) !important;
    text-transform: uppercase !important; width: 100% !important;
}
.stButton > button:hover {
    background: rgba(0,180,255,0.1) !important; border-color: var(--accent-blue) !important;
    color: var(--accent-cyan) !important;
    box-shadow: 0 0 25px rgba(0,180,255,0.25), inset 0 0 20px rgba(0,180,255,0.05) !important;
    transform: translateY(-2px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,100,200,0.4), rgba(0,180,255,0.2)) !important;
    border-color: var(--accent-blue) !important; color: #fff !important;
    box-shadow: 0 0 25px rgba(0,180,255,0.25) !important;
}

.section-hdr {
    font-family: 'Exo 2', sans-serif; font-size: 1.25rem; font-weight: 700;
    letter-spacing: 0.25em; color: var(--accent-blue); text-transform: uppercase;
    padding: 0.5rem 0; border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 0.5rem; position: relative;
}
.section-hdr::after {
    content: ''; position: absolute; bottom: -1px; left: 0; width: 90px; height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
}
.section-hdr.red { color: var(--accent-red); }
.section-hdr.red::after { background: linear-gradient(90deg, var(--accent-red), #ff8800); }

.section-sub {
    font-size: 0.73rem; color: var(--text-secondary);
    letter-spacing: 0.15em; margin-bottom: 2rem; text-transform: uppercase;
}

.terminal {
    background: rgba(0,10,20,0.92); border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--accent-blue); border-radius: 6px;
    padding: 0.8rem 1.2rem; font-size: 0.72rem; color: var(--accent-green);
    letter-spacing: 0.15em; margin-top: 1.5rem; position: relative; overflow: hidden;
}
.terminal::before {
    content: ''; position: absolute; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,136,0.012) 2px, rgba(0,255,136,0.012) 4px);
    pointer-events: none;
}

.info-box {
    background: rgba(0,10,20,0.85); border: 1px solid var(--border-subtle);
    border-radius: 8px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;
    font-size: 0.75rem; color: var(--text-secondary); line-height: 1.85;
}
.info-box strong { color: var(--accent-blue); }

.stFileUploader {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-glow) !important;
    border-radius: 10px !important; padding: 1rem !important;
}
.stTextInput > div > div {
    background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important; color: var(--text-primary) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.stTextInput > div > div:focus-within {
    border-color: var(--accent-blue) !important; box-shadow: 0 0 15px rgba(0,180,255,0.15) !important;
}
.stSelectbox > div > div {
    background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important; color: var(--text-primary) !important;
}
.stNumberInput > div > div {
    background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div { background: var(--accent-blue) !important; }
div[data-testid="metric-container"] {
    background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important; padding: 1rem !important; transition: border-color 0.3s;
}
div[data-testid="metric-container"]:hover { border-color: var(--border-glow) !important; }
div[data-testid="metric-container"] label {
    color: var(--text-secondary) !important; font-size: 0.68rem !important;
    letter-spacing: 0.2em !important; font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
}
div[data-testid="metric-container"] div[data-testid="metric-value"] {
    color: var(--accent-blue) !important; font-family: 'Exo 2', sans-serif !important; font-weight: 700 !important;
}
div[data-testid="stDataFrame"] {
    background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important; overflow: hidden !important;
}
.stSuccess { background: rgba(0,255,136,0.07) !important; border: 1px solid rgba(0,255,136,0.28) !important; border-radius: 8px !important; color: var(--accent-green) !important; }
.stError, .stWarning { background: rgba(255,51,85,0.07) !important; border: 1px solid rgba(255,51,85,0.28) !important; border-radius: 8px !important; }
.stInfo { background: rgba(0,180,255,0.07) !important; border: 1px solid rgba(0,180,255,0.2) !important; border-radius: 8px !important; color: var(--accent-blue) !important; }

hr { border-color: var(--border-subtle) !important; margin: 1.5rem 0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 2px; }
.stSpinner > div { border-color: var(--accent-blue) transparent transparent transparent !important; }
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
.stMarkdown, .stButton, .stFileUploader { animation: fadeInUp 0.4s ease-out forwards; }
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
        <div class="sid"><span>●</span> &nbsp;SESSION: <span>{sid}</span></div>
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
      <div class="hero-sub">AI-POWERED SURVEILLANCE INTELLIGENCE SYSTEM</div>
      <div class="hero-status">[ SYSTEM ONLINE ] · OPEN ACCESS · BUILD v3.1</div>

      <div class="stats-row">
        <div class="stat-item"><span class="stat-value">10</span><span class="stat-label">Modules</span></div>
        <div class="stat-item"><span class="stat-value">97%</span><span class="stat-label">Accuracy</span></div>
        <div class="stat-item"><span class="stat-value">9</span><span class="stat-label">Weapon Classes</span></div>
        <div class="stat-item"><span class="stat-value">CPU</span><span class="stat-label">No GPU Needed</span></div>
      </div>

      <div class="scan-line"></div>

      <div class="module-grid">
        <div class="mod-card">
          <div class="mod-icon">🎯</div>
          <div class="mod-name">Person Detection</div>
          <div class="mod-tag">YOLOv8-nano</div>
          <div class="mod-desc">Real-time person detection on any uploaded image. Returns bounding boxes and per-person confidence scores. Runs entirely on CPU — no GPU required.</div>
          <div class="mod-meta">Model: yolov8n.pt · Class 0 only · Confidence threshold: 0.4 · CPU optimized</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">🔥</div>
          <div class="mod-name">Behavioral Analytics</div>
          <div class="mod-tag">ByteTrack · OpenCV</div>
          <div class="mod-desc">Upload any video and get persistent person IDs across frames, a live behavioral heatmap showing movement density, per-person dwell times, and automated loitering alerts.</div>
          <div class="mod-meta">Tracker: ByteTrack IOU · Heatmap: NumPy accumulation · Alert threshold: 60s · Max input: 15s</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">🕵️</div>
          <div class="mod-name">OSINT Audit</div>
          <div class="mod-tag">LBPH Face Recognition</div>
          <div class="mod-desc">Upload a face and receive a Privacy Exposure Score from 0 to 100. LBPH embeddings are matched against a reference gallery. Risk classified as LOW, MEDIUM, or HIGH.</div>
          <div class="mod-meta">Engine: OpenCV LBPH · Gallery: cosine similarity · Score: 0–100 · No data stored</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">🧠</div>
          <div class="mod-name">Emotion Intelligence</div>
          <div class="mod-tag">DeepFace · TensorFlow</div>
          <div class="mod-desc">Multi-face emotion analysis. Returns dominant emotion, estimated age, and gender per face. False-positive filter rejects faces smaller than 15% of frame area.</div>
          <div class="mod-meta">Backend: DeepFace · Detector: OpenCV · Min face size: 15% of frame · Multi-subject</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">💬</div>
          <div class="mod-name">NL Query Engine</div>
          <div class="mod-tag">Groq LLaMA 3</div>
          <div class="mod-desc">Type a surveillance query in plain English or Roman Urdu. LLaMA 3 extracts structured filters — emotion, gender, age, dwell time, loitering — then matches against person records.</div>
          <div class="mod-meta">Model: llama-3.1-8b-instant · Languages: English + Roman Urdu · Output: JSON filters</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">⚠️</div>
          <div class="mod-name">Weapon Detection</div>
          <div class="mod-tag">YOLOv8 Custom · 9 Classes</div>
          <div class="mod-desc">Custom YOLOv8 trained on 714 real weapon images across 9 classes. Achieves Handgun 89.5%, Shotgun 96.3%, SMG 98.6% average precision. Fires immediate threat alert on detection.</div>
          <div class="mod-meta">Classes: Handgun · Knife · Shotgun · Sniper · AR · SMG · Sword · Bazooka · GL · mAP50: 53.2%</div>
        </div>
        <div class="mod-card research-card">
          <div class="mod-icon">📊</div>
          <div class="mod-name red">Threat Momentum Score</div>
          <div class="mod-tag red">Novel Algorithm · TMS v1.0</div>
          <div class="mod-desc">Original research contribution. Accumulates threat signals over time using a compound interest model — loitering, stress emotion, rapid movement, restricted zone, gaze anomaly, group formation. Score decays when signals stop.</div>
          <div class="mod-meta">Signals: 6 · Decay half-life: 45s · Amplifier: score/200 · Levels: CLEAR / LOW / MEDIUM / HIGH / CRITICAL</div>
        </div>
        <div class="mod-card research-card">
          <div class="mod-icon">🧬</div>
          <div class="mod-name red">Behavioral DNA</div>
          <div class="mod-tag red">Novel Algorithm · BDF v1.0</div>
          <div class="mod-desc">Camera-agnostic person re-identification using behavioral signature alone. Identifies the same person across cameras without face recognition. Works through masks, hats, and low resolution.</div>
          <div class="mod-meta">Signals: gait · velocity · spatial preference · social distance · dwell zones · Match threshold: 82%</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">📄</div>
          <div class="mod-name">Intel Report</div>
          <div class="mod-tag">fpdf2 · PDF Export</div>
          <div class="mod-desc">Generate a classified PDF intelligence report from any session. Includes session overview, weapon threat log in red, per-subject behavioral records, and NL query history. Dark cyberpunk theme.</div>
          <div class="mod-meta">Library: fpdf2 · Theme: dark bg + green text · Threat: red sections · Download: immediate</div>
        </div>
        <div class="mod-card">
          <div class="mod-icon">⚡</div>
          <div class="mod-name">System Intel</div>
          <div class="mod-tag">Live Status</div>
          <div class="mod-desc">Live system dashboard. All active modules listed with tech stack, benchmark results, API endpoint reference, model filenames, and deployment metadata for full transparency.</div>
          <div class="mod-meta">Version: v3.1.0 · Deployment: HuggingFace Spaces · API: FastAPI OAS 3.1 · GitHub: open source</div>
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
    render_session_bar()
    st.markdown('<div class="app-header">👁 PHANTOMEYE</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">SELECT INTELLIGENCE MODULE · ALL SYSTEMS ACTIVE</div>', unsafe_allow_html=True)

    modules = [
        ("DETECTION", "Detection"),
        ("ANALYTICS", "Analytics"),
        ("OSINT",     "OSINT"),
        ("EMOTION",   "Emotion"),
        ("NL QUERY",  "NL Query"),
        ("WEAPON",    "Weapon"),
        ("THREAT",    "Threat Score"),
        ("BDF",       "Behavioral DNA"),
        ("REPORT",    "Report"),
        ("INTEL",     "System"),
    ]
    cols = st.columns(len(modules))
    for i, (key, label) in enumerate(modules):
        with cols[i]:
            if st.button(label, key=f"mod_{key}"):
                st.session_state.page = key
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="terminal">[ PHANTOMEYE v3.1 ] · YOLOv8 loaded · ByteTrack active · '
        'DeepFace online · Groq LLaMA connected · Weapon model ready · '
        'TMS engine active · BDF engine active · All 10 modules ONLINE</div>',
        unsafe_allow_html=True
    )


def detection_page():
    render_session_bar()
    back_button()
    st.markdown('<div class="section-hdr">Person Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">YOLOv8-nano · CPU inference · class 0 persons only · confidence 0.4</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> Upload any image and PhantomEye runs YOLOv8-nano inference on CPU.
        Configured for class 0 (person) detection only with a confidence threshold of 0.4.
        Each detected person receives a bounding box and confidence score.
        Expand the detection log below the output image to inspect raw bbox coordinates per subject.
        No GPU required — inference runs on standard CPU hardware.
    </div>
    """, unsafe_allow_html=True)
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
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> Upload a video and PhantomEye processes up to 15 seconds of footage.
        ByteTrack assigns a persistent ID to each person and maintains it across frames, even through brief
        occlusion. A NumPy heatmap accumulates every pixel position each person visits — high-activity zones
        appear red in the output. Dwell time is tracked per ID in seconds. If any person remains in one zone
        beyond the loitering threshold, an alert fires listing their assigned ID.
    </div>
    """, unsafe_allow_html=True)
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
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> Upload a face photo and PhantomEye extracts an LBPH (Local Binary Pattern
        Histogram) embedding from the detected face region. This embedding is compared against every person in the
        reference gallery using cosine similarity. The Privacy Exposure Score (0–100) reflects recognition confidence —
        a higher score means stronger match. Risk level is classified as LOW (score &lt; 40), MEDIUM (40–70), or
        HIGH (&gt; 70). All processing is in-session only — nothing is stored server-side.
    </div>
    """, unsafe_allow_html=True)
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
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> PhantomEye runs DeepFace analysis on every detected face in the uploaded image.
        For each face it returns the dominant emotion from 7 classes (angry, fear, sad, happy, surprise, neutral,
        disgust), an estimated age, and a gender classification. A false-positive filter discards any face region
        smaller than 15% of the frame area. Multiple faces in a single image are processed independently.
    </div>
    """, unsafe_allow_html=True)
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
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> Type any surveillance query in natural language — English or Roman Urdu both
        work. Groq's LLaMA 3 (llama-3.1-8b-instant) parses the intent and extracts structured filters: emotion type,
        gender, age range, minimum dwell time, and loitering status. These filters are applied against a person record
        set and matching subjects are returned in a table. First open-source surveillance system with multilingual
        NL query support including Roman Urdu.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="terminal">llama-3.1-8b-instant via Groq · JSON filter extraction · Roman Urdu supported · apply_filters() on records</div>', unsafe_allow_html=True)

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
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> A custom YOLOv8 model trained from scratch on 714 real-world weapon images
        across 9 classes. Trained on Kaggle T4 GPU. Achieves Handgun 89.5%, Shotgun 96.3%, SMG 98.6% average
        precision at mAP50 of 53.2%. Upload any image — detected weapons are highlighted with red bounding boxes
        and an immediate threat alert fires listing the weapon class and confidence score.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="terminal">weapon_detector.pt · mAP50: 53.2% · Handgun: 89.5% · Shotgun: 96.3% · SMG: 98.6% · 714 real training images</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-sub">Novel temporal threat accumulation · compound behavioral signal model · PhantomEye original research</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Research contribution:</strong> Unlike binary threat detection systems that output a single yes/no,
        TMS accumulates behavioral signals over time using a compound interest model. Each new signal contributes
        weighted to the score. When the score is already elevated, new signals contribute proportionally more —
        the amplifier effect. Score decays with a 45-second half-life when no signals arrive.
        <br><br>
        <strong>6 signals and weights:</strong> loitering (0.28) · stress emotion (0.22) · rapid movement (0.18)
        · proximity violation (0.15) · gaze anomaly (0.10) · group formation (0.07)
    </div>
    """, unsafe_allow_html=True)
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
            background:rgba(0,5,15,0.95); border:2px solid {color};
            border-radius:12px; box-shadow: 0 0 40px {color}22;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
                color:#3a6080; letter-spacing:0.35em; margin-bottom:0.75rem; text-transform:uppercase;">
                Threat Momentum Score · Person {r.person_id}
            </div>
            <div style="font-size:5rem; font-weight:900; color:{color}; font-family:'Exo 2',sans-serif; line-height:1;">
                {r.tms_score:.1f}
            </div>
            <div style="font-size:1.1rem; font-weight:700; color:{color}; letter-spacing:0.4em; margin-top:0.5rem; font-family:'Rajdhani',sans-serif;">
                {r.threat_level}
            </div>
            <div style="font-size:0.68rem; color:#3a6080; margin-top:0.75rem; font-family:'IBM Plex Mono',monospace; letter-spacing:0.1em;">
                Momentum: {r.momentum:+.2f}/frame &nbsp;|&nbsp; Time in system: {r.time_in_system}s
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
        st.markdown('<div class="terminal">Level distribution: ' + ' · '.join(f"{k}: {v}" for k, v in summary["level_distribution"].items()) + '</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-sub">Camera-agnostic re-identification · no face required · pure movement signature</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Research contribution:</strong> Identifies the same person across cameras using behavioral
        signature alone — gait rhythm, velocity profile, spatial preference zones, social distance pattern,
        and dwell locations. Works with masks, hats, and at distances where face recognition fails completely.
        When a person re-enters the scene with a new tracking ID, BDF matches them to their previous identity
        using cosine similarity on a 5-component unified behavioral feature vector. Match threshold: 82%.
        <br><br>
        <strong>5 behavioral components:</strong> gait signature (stride rhythm) · velocity profile (speed
        distribution) · spatial preference (normalized grid heatmap) · social distance average ·
        dwell zone signature (stopping locations)
    </div>
    """, unsafe_allow_html=True)
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
        <div style="padding:2rem; margin:1rem 0;
            background:rgba(0,5,15,0.95); border:2px solid {color};
            border-radius:10px; box-shadow: 0 0 30px {color}22;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
                color:#3a6080; letter-spacing:0.3em; margin-bottom:0.5rem; text-transform:uppercase;">
                Behavioral DNA Match Result · Person {r.query_id}
            </div>
            <div style="font-size:2.5rem; font-weight:900; color:{color}; font-family:'Exo 2',sans-serif;">
                {"MATCH FOUND" if r.is_match else "NO MATCH"}
            </div>
            <div style="font-size:0.75rem; color:#7ab3d4; margin-top:0.75rem; font-family:'IBM Plex Mono',monospace; line-height:1.6;">
                {r.explanation}
            </div>
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


def report_page():
    render_session_bar()
    from core.reporter import generate_report
    back_button()
    st.markdown('<div class="section-hdr">Intelligence Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Classified PDF · session data · threat log · subject records · one-click download</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> Fill in the session data fields below — total persons detected, loitering
        alerts, per-subject behavioral records with emotion and dwell time, and any weapon detections.
        Click Generate and PhantomEye produces a classified PDF using fpdf2. Dark background with green
        terminal-style text. Weapon threat sections highlighted in red. CLASSIFIED header on first page.
        Immediate download — nothing stored server-side.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="terminal">fpdf2 · dark theme · CLASSIFIED header · weapon sections in red · immediate download · zero server storage</div>', unsafe_allow_html=True)

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
        with st.spinner("Generating report..."):
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
    st.markdown('<div class="section-sub">Module registry · model benchmarks · deployment info · API reference</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SYSTEM",  "PhantomEye")
    c2.metric("VERSION", "v3.1.0")
    c3.metric("STATUS",  "ONLINE")
    c4.metric("MODULES", "10 ACTIVE")

    st.markdown("<br>", unsafe_allow_html=True)

    modules_info = [
        ("DETECTION",       "YOLOv8-nano",   "yolov8n.pt · class 0 · confidence 0.4+ · CPU only"),
        ("ANALYTICS",       "ByteTrack",     "IOU matching · NumPy heatmap · dwell time · loitering threshold: 60s"),
        ("OSINT",           "LBPH Face",     "LBPH embedding · cosine gallery search · score 0–100 · LOW/MEDIUM/HIGH"),
        ("EMOTION",         "DeepFace + TF", "7 emotion classes · age + gender · OpenCV detector · 15% min face size"),
        ("NL QUERY",        "Groq LLaMA 3",  "llama-3.1-8b-instant · English + Roman Urdu · JSON filter extraction"),
        ("WEAPON",          "YOLOv8 Custom", "9 classes · mAP50 53.2% · Handgun 89.5% · Shotgun 96.3% · SMG 98.6%"),
        ("THREAT MOMENTUM", "TMS v1.0",      "Novel · 6 signals · compound amplifier · 45s decay · 5 threat levels"),
        ("BEHAVIORAL DNA",  "BDF v1.0",      "Novel · 5 behavioral components · cosine similarity · 82% match threshold"),
        ("REPORT",          "fpdf2",         "Classified PDF · dark theme · CLASSIFIED header · threat sections in red"),
        ("API",             "FastAPI",        "OAS 3.1 · CORS enabled · uvicorn · modular routes"),
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
        "novel_contributions": ["Threat Momentum Score (TMS v1.0)", "Behavioral DNA Fingerprint (BDF v1.0)"],
        "status":              "online",
        "access":              "open",
    })


def main():
    if "page"       not in st.session_state:
        st.session_state.page = "landing"
    if "session_id" not in st.session_state:
        st.session_state.session_id = "PE-" + str(uuid.uuid4())[:8].upper()

    page = st.session_state.page

    if   page == "landing":   landing()
    elif page == "home":      home()
    elif page == "DETECTION": detection_page()
    elif page == "ANALYTICS": analytics_page()
    elif page == "OSINT":     osint_page()
    elif page == "EMOTION":   emotion_page()
    elif page == "NL QUERY":  nlquery_page()
    elif page == "WEAPON":    weapon_page()
    elif page == "THREAT":    threat_page()
    elif page == "BDF":       bdf_page()
    elif page == "REPORT":    report_page()
    elif page == "INTEL":     intel_page()


if __name__ == "__main__":
    main()
    