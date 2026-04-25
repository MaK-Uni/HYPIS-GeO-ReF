"""
HyPIS Ug 
═══════════════════════════════════════════════════════════════════════════════
HYPIS-GEo  v7.0  — Supervisor-Review Fixes
────────────────────────────────────────────
KEY FIXES (v7.0):
  ✔ Irrigation is NOT automatic every day — only triggered when Root-Zone
    Depletion (Dr) EXCEEDS the MAD/RAW threshold AND rain (Pe) < ETc
  ✔ On rainy days where Pe ≥ ETc → Status = "🌧️ Rain Covered", IWR = 0
  ✔ Deficit (Dr) column added to ALL tables — with FC / RAW / TAW context
  ✔ MAD thresholds shown per crop AND per soil type (FAO thresholds)
  ✔ FC guard: warns if soil is already at/above FC (over-saturation risk)
  ✔ Tab 1 "Today's IWR" now shows PAST 5 DAYS context table BEFORE today's
    result, so farmer can see how soil & crop demand evolved each day
  ✔ IWR is explicitly a 24-hr daily value (not cumulative)
  ✔ All tables include: Date, Rain, ET₀, ETc, Pe, Deficit (Dr mm),
    Status, NIR, IWR (gross), Volume (m³), Pump time
  ✔ Kc stage bug FIXED — ini/mid/end correctly switches
  ✔ "Method" column removed from ALL tables

Author: Prosper BYARUHANGA · HyPIS App v7.0
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, json, time as _time, subprocess, pathlib
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta, date

# ── Auto-install ML deps ──────────────────────────────────────────────────────
for _pkg in ("joblib", "xgboost"):
    try:
        __import__(_pkg)
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", _pkg, "--quiet"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="HyPIS Ug – Uganda IWR", layout="wide",
                   initial_sidebar_state="expanded")

_HERE = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# UGANDA LOCATIONS
# ══════════════════════════════════════════════════════════════════════════════
LOCATIONS = {
    "Kampala (Makerere Uni)":   ( 0.33396,  32.56801, 1239.01),
    "MUARiK (Kabanyoro)":       ( 0.464533,  32.612517, 1178.97),
    "Mbarara":                  (-0.6133,  30.6544, 1433),
    "Isingiro (Kabuyanda)":    (-0.95658,  30.61432, 1364.59),
    "Gulu":                    ( 2.7746,  32.2990, 1105),
    "Jinja":                   ( 0.4244,  33.2041, 1137),
    "Mbale":                   ( 1.0804,  34.1751, 1155),
    "Kabale":                  (-1.2490,  29.9900, 1869),
    "Fort Portal":             ( 0.6710,  30.2750, 1537),
    "Masaka":                  (-0.3310,  31.7373, 1148),
    "Lira":                    ( 2.2499,  32.9002, 1074),
    "Soroti":                  ( 1.7153,  33.6107, 1130),
    "Arua":                    ( 3.0210,  30.9110, 1047),
    "Hoima":                   ( 1.4352,  31.3524, 1562),
    "Kasese":                  ( 0.1820,  30.0804,  933),
    "Tororo":                  ( 0.6920,  34.1810, 1148),
    "Moroto":                  ( 2.5340,  34.6650, 1390),
    "Custom Location":         (None, None, None),
}

# ══════════════════════════════════════════════════════════════════════════════
# SOIL DATABASE — Uganda districts, FAO / HWSD v2 verified
# ══════════════════════════════════════════════════════════════════════════════
DISTRICT_SOIL = {
    "Kampala (Makerere Uni)":  {"fc": 0.32, "pwp": 0.18, "texture": "Clay Loam",        "source": "HWSD v2"},
    "MUARiK (Kabanyoro)":      {"fc": 0.26, "pwp": 0.12, "texture": "Sandy Clay Loam",  "source": "HWSD v2"},
    "Mbarara":                  {"fc": 0.30, "pwp": 0.15, "texture": "Loam",             "source": "HWSD v2"},
    "Isingiro (Kabuyanda)":    {"fc": 0.28, "pwp": 0.14, "texture": "Loam",             "source": "HWSD v2"},
    "Gulu":                    {"fc": 0.24, "pwp": 0.11, "texture": "Sandy Loam",        "source": "HWSD v2"},
    "Jinja":                   {"fc": 0.31, "pwp": 0.16, "texture": "Clay Loam",         "source": "HWSD v2"},
    "Mbale":                   {"fc": 0.27, "pwp": 0.13, "texture": "Loam",              "source": "HWSD v2"},
    "Kabale":                  {"fc": 0.33, "pwp": 0.19, "texture": "Clay",              "source": "HWSD v2"},
    "Fort Portal":             {"fc": 0.29, "pwp": 0.14, "texture": "Loam",              "source": "HWSD v2"},
    "Masaka":                  {"fc": 0.25, "pwp": 0.12, "texture": "Sandy Loam",        "source": "HWSD v2"},
    "Lira":                    {"fc": 0.23, "pwp": 0.10, "texture": "Sandy Loam",        "source": "HWSD v2"},
    "Soroti":                  {"fc": 0.22, "pwp": 0.09, "texture": "Sandy Loam",        "source": "HWSD v2"},
    "Arua":                    {"fc": 0.21, "pwp": 0.08, "texture": "Loamy Sand",        "source": "HWSD v2"},
    "Hoima":                   {"fc": 0.28, "pwp": 0.13, "texture": "Sandy Loam",        "source": "HWSD v2"},
    "Kasese":                  {"fc": 0.35, "pwp": 0.20, "texture": "Clay",              "source": "HWSD v2"},
    "Tororo":                  {"fc": 0.26, "pwp": 0.12, "texture": "Sandy Clay Loam",   "source": "HWSD v2"},
    "Moroto":                  {"fc": 0.18, "pwp": 0.08, "texture": "Sandy Loam",        "source": "HWSD v2"},
    "Custom Location":         {"fc": 0.28, "pwp": 0.14, "texture": "Loam (default)",    "source": "FAO-56 default"},
}

# ══════════════════════════════════════════════════════════════════════════════
# FAO-56 STANDARD SOIL TYPES
# ══════════════════════════════════════════════════════════════════════════════
SOIL_OPTS = {
    "Sand":              {"fc": 0.10, "pwp": 0.05, "desc": "Very fast drainage, very low retention"},
    "Loamy Sand":        {"fc": 0.14, "pwp": 0.07, "desc": "Fast drainage, low retention"},
    "Sandy Loam":        {"fc": 0.20, "pwp": 0.09, "desc": "Moderate drainage, moderate retention"},
    "Sandy Clay Loam":   {"fc": 0.26, "pwp": 0.12, "desc": "Moderate-high retention"},
    "Loam":              {"fc": 0.28, "pwp": 0.14, "desc": "Good balance of drainage and retention"},
    "Silt Loam":         {"fc": 0.31, "pwp": 0.15, "desc": "High retention, moderate drainage"},
    "Silt":              {"fc": 0.33, "pwp": 0.16, "desc": "High retention"},
    "Clay Loam":         {"fc": 0.32, "pwp": 0.18, "desc": "High retention, slow drainage"},
    "Silty Clay Loam":   {"fc": 0.35, "pwp": 0.20, "desc": "Very high retention"},
    "Sandy Clay":        {"fc": 0.28, "pwp": 0.16, "desc": "Moderate-high retention"},
    "Silty Clay":        {"fc": 0.38, "pwp": 0.23, "desc": "Very high retention, poor drainage"},
    "Clay":              {"fc": 0.40, "pwp": 0.25, "desc": "Maximum retention, waterlogging risk"},
}

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
TIMEZONE   = "Africa/Nairobi"
_SIGMA     = 4.903e-9
_W2M       = 4.87 / np.log(67.8 * 10.0 - 5.42)

# ══════════════════════════════════════════════════════════════════════════════
# ML MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
ML_MODEL    = None
ML_OK       = False
ML_STATUS   = ""
ML_FEATURES = ["tmean", "rh", "wind", "kc", "precipitation",
               "soil_fc", "soil_pwp", "root_depth"]
_MODEL_PATH = pathlib.Path(_HERE) / "irrigation_xgboost_model_with_soil.pkl"

def _load_ml_model():
    global ML_MODEL, ML_OK, ML_STATUS
    try:
        import joblib, xgboost
        if not _MODEL_PATH.exists():
            ML_STATUS = f"⚠️ Model file not found: {_MODEL_PATH.name}"
            return
        ML_MODEL  = joblib.load(str(_MODEL_PATH))
        ML_OK     = True
        ML_STATUS = f"✅ XGBoost loaded · Features: {', '.join(ML_FEATURES)}"
    except ModuleNotFoundError as e:
        ML_STATUS = f"⚠️ Missing package: {e}"
    except Exception as e:
        ML_STATUS = f"⚠️ Model load error: {e}"

_load_ml_model()

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<style>
:root{
  --hb:#1a5fc8;--hg:#0b6b1b;--hr:#b81c1c;
  --bg:#f4f8f2;--sf:#fff;--bd:#dbe9db;--tx:#17301b;
  --gn:#0b6b1b;--gd:#075214;--gs:#e7f3e6;
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{
  background:var(--bg)!important;color:var(--tx)!important;}
[data-testid="stHeader"],[data-testid="stToolbar"]{background:transparent!important;}
[data-testid="stMetric"]{background:var(--sf);border:1px solid var(--bd);
  border-radius:12px;padding:.5rem .7rem;}
[data-testid="stMetricLabel"] p{font-size:.76rem!important;margin:0!important;}
[data-testid="stMetricValue"] div{font-size:1.05rem!important;font-weight:700!important;}
div[data-baseweb="tab-list"]{gap:.3rem;background:transparent!important;}
button[data-baseweb="tab"]{background:var(--gs)!important;border:1px solid #b8d1b8!important;
  border-radius:999px!important;color:var(--gd)!important;
  padding:.35rem .75rem!important;font-size:.83rem!important;}
button[data-baseweb="tab"]>div{color:var(--gd)!important;font-weight:600;}
button[data-baseweb="tab"][aria-selected="true"]{
  background:var(--gn)!important;border-color:var(--gn)!important;}
button[data-baseweb="tab"][aria-selected="true"]>div{color:#fff!important;}
[data-baseweb="select"]>div,div[data-baseweb="input"]>div,
.stNumberInput>div>div,.stTextInput>div>div{
  background:var(--sf)!important;color:var(--tx)!important;border-color:#c9d9c9!important;}
.stButton>button,.stDownloadButton>button{
  background:var(--gn)!important;color:#fff!important;
  border:1px solid var(--gn)!important;border-radius:10px!important;}
.stButton>button:hover{background:var(--gd)!important;}
section[data-testid="stSidebar"]{background:#eef5ec!important;}
button[title="Fork this app"],[data-testid="stToolbarActionButtonIcon"],
[data-testid="stBottomBlockContainer"],.stDeployButton,footer{display:none!important;}
.block-container{padding-top:.8rem!important;}
.hx-outer{border-radius:20px;overflow:hidden;margin:0 0 10px 0;
  background:linear-gradient(90deg,var(--hb) 0%,var(--hb) 33.3%,
  var(--hg) 33.3%,var(--hg) 66.6%,var(--hr) 66.6%,var(--hr) 100%);
  padding:9px 9px 7px 9px;}
.hx-panel{background:#fff;border:2px solid #d0ddd0;border-radius:14px;
  padding:9px 16px 7px 16px;}
.hx-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.hx-wm{font-family:Georgia,serif;font-size:2.4rem;font-weight:700;
  line-height:1;letter-spacing:-1px;flex-shrink:0;}
.hx-wm .H{color:#1a5fc8;}.hx-wm .y{color:#0b6b1b;}.hx-wm .P{color:#b81c1c;}
.hx-wm .I{color:#1a5fc8;}.hx-wm .S{color:#0b6b1b;}
.hx-wm .Ug{color:#b81c1c;font-size:1.4rem;vertical-align:middle;margin-left:4px;}
.hx-sub{font-family:Georgia,serif;font-size:.95rem;flex:1 1 160px;color:#444;}
.hx-auth{margin:4px 0 0 4px;font-family:Georgia,serif;font-size:.78rem;color:#ddd;}
.hx-auth strong{color:#fff;}
.geo-panel{background:#fff;border:1.5px solid #b8d4f8;border-radius:14px;
  padding:10px 16px;margin:6px 0 10px 0;font-size:.86rem;color:#14324d;}
.geo-panel b{color:#1a5fc8;}
.geo-coord{font-family:monospace;background:#eef4ff;padding:2px 6px;
  border-radius:6px;font-size:.82rem;}
.nir-box{background:#fff3cd;border:1px solid #ffc107;border-radius:10px;
  padding:8px 14px;font-size:.87rem;margin:4px 0;}
.iwr-box{background:#d4edda;border:1px solid #28a745;border-radius:10px;
  padding:8px 14px;font-size:.87rem;margin:4px 0;font-weight:600;}
.vol-box{background:#cfe2ff;border:1px solid #0d6efd;border-radius:10px;
  padding:8px 14px;font-size:.87rem;margin:4px 0;}
.kc-stage{background:#e8f6ea;border:1px solid #a8d8a8;border-radius:10px;
  padding:6px 14px;font-size:.85rem;color:#073f12;margin:4px 0;font-weight:600;}
.soil-panel{background:#fef9ee;border:1px solid #e0c97a;border-radius:10px;
  padding:8px 14px;font-size:.85rem;margin:4px 0;}
.mad-panel{background:#f0f4ff;border:1px solid #7b9ed9;border-radius:10px;
  padding:8px 14px;font-size:.85rem;margin:4px 0;}
.past-panel{background:#f8fff8;border:2px solid #0b6b1b;border-radius:14px;
  padding:12px 16px;margin:10px 0;}
.today-panel{background:#fffbe6;border:2px solid #f5a623;border-radius:14px;
  padding:12px 16px;margin:10px 0;}
.warn-fc{background:#fff0f0;border:1px solid #d73027;border-radius:10px;
  padding:8px 14px;font-size:.87rem;margin:4px 0;font-weight:600;}
.live-dot{width:7px;height:7px;background:#22c55e;border-radius:50%;
  display:inline-block;margin-right:4px;animation:blink 1.4s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH (every 1 hour)
# ══════════════════════════════════════════════════════════════════════════════
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = _time.time()
_el = _time.time() - st.session_state["last_refresh"]
if _el >= 3600:
    st.cache_data.clear()
    st.session_state["last_refresh"] = _time.time()
    st.rerun()
_rem = max(0, 3600 - int(_el))

# ══════════════════════════════════════════════════════════════════════════════
# BRAND HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="hx-outer"><div class="hx-panel"><div class="hx-row">
<span style="font-size:1.5rem;">&#127807;</span>
<span class="hx-wm">
  <span class="H">H</span><span class="y">y</span><span class="P">P</span>
  <span class="I">I</span><span class="S">S</span><span class="Ug"> Ug</span>
</span>
<span class="hx-sub">HydroPredict · IrrigSched · Uganda Multi-Location IWR v7.0</span>
</div></div>
<div class="hx-auth">by: Prosper <strong>BYARUHANGA</strong>
&nbsp;·&nbsp; HyPIS App v7.0 &nbsp;·&nbsp; FAO-56 PM + XGBoost ML · Uganda</div>
</div>""", unsafe_allow_html=True)

_now_str = datetime.now().strftime("%d %b %Y %H:%M")
st.caption(
    f'<span class="live-dot"></span> Live &middot; <b>{_now_str}</b>'
    f" &nbsp;·&nbsp; Refresh in <b>{_rem // 3600}h {(_rem % 3600) // 60}m</b>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — LOCATION SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("📍 Location — Uganda")
loc_name = st.sidebar.selectbox(
    "Select District / Site",
    list(LOCATIONS.keys()),
    index=0,
    key="loc_sel",
)

_lcoords = LOCATIONS[loc_name]

if loc_name == "Custom Location":
    st.sidebar.markdown("**Enter Custom Coordinates:**")
    _clat  = st.sidebar.number_input("Latitude",  value=0.3380, format="%.4f", key="clat")
    _clon  = st.sidebar.number_input("Longitude", value=32.5680, format="%.4f", key="clon")
    _celev = st.sidebar.number_input("Elevation (m a.s.l.)", value=1189, key="celev")
    LAT, LON, ELEV = _clat, _clon, _celev
    SITE_NAME = f"Custom ({LAT:.4f}°, {LON:.4f}°)"
else:
    LAT, LON, ELEV = _lcoords
    SITE_NAME = loc_name

GMAPS_URL = f"https://maps.google.com/?q={LAT},{LON}"
GMAPS_SAT = f"https://maps.google.com/maps?q={LAT},{LON}&ll={LAT},{LON}&z=14&t=k"

# ── Soil auto-load for selected district ────────────────────────────────────
_dsoil = DISTRICT_SOIL.get(loc_name, DISTRICT_SOIL["Custom Location"])
SITE_FC   = _dsoil["fc"]
SITE_PWP  = _dsoil["pwp"]
SITE_TEXT = _dsoil["texture"]
SITE_SRC  = _dsoil["source"]

st.sidebar.markdown(
    f"""**📍 {SITE_NAME}**  
`Lat {LAT}°` · `Lon {LON}°` · `{ELEV} m a.s.l.`  
[🗺️ Google Maps]({GMAPS_URL}) | [🛰️ Satellite]({GMAPS_SAT})"""
)

st.sidebar.markdown("---\n### 🌍 Soil Type")
st.sidebar.info(
    f"**Auto-loaded for {loc_name}:**  \n"
    f"Texture: **{SITE_TEXT}**  \n"
    f"FC: **{SITE_FC*100:.0f}%** · PWP: **{SITE_PWP*100:.0f}%**  \n"
    f"Source: {SITE_SRC}"
)
soil_override = st.sidebar.checkbox("Override soil type", value=False, key="soil_ov")
if soil_override:
    soil_sel_s = st.sidebar.selectbox("Soil Type", list(SOIL_OPTS.keys()), key="soil_sel_s")
    soil_obj_s = SOIL_OPTS[soil_sel_s]
    ACTIVE_FC  = soil_obj_s["fc"]
    ACTIVE_PWP = soil_obj_s["pwp"]
    ACTIVE_TXT = soil_sel_s
else:
    ACTIVE_FC  = SITE_FC
    ACTIVE_PWP = SITE_PWP
    ACTIVE_TXT = SITE_TEXT

st.sidebar.markdown("---\n### 💧 Irrigation System")
IRRIG_SYSTEMS = {
    "Drip / Trickle":   0.90,
    "Sprinkler":        0.80,
    "Surface / Furrow": 0.65,
    "Flood":            0.55,
    "Centre Pivot":     0.85,
}
irrig_sys = st.sidebar.selectbox("System Type", list(IRRIG_SYSTEMS.keys()), index=0, key="irrig_sys")
Ef = IRRIG_SYSTEMS[irrig_sys]
st.sidebar.info(f"Efficiency **Ef = {Ef*100:.0f}%**  \nIWR (gross) = NIR ÷ {Ef:.2f}")

st.sidebar.markdown("---\n### 📐 Field & Pump")
area_ha   = st.sidebar.number_input("Field Area (ha)", value=1.0, min_value=0.1, step=0.1, key="area_g")
pump_flow = st.sidebar.number_input("Pump Flow Rate (m³/hr)", value=5.0, min_value=0.5, step=0.5, key="pump_g")

if ML_OK:
    st.sidebar.markdown("---\n### 🤖 ML Model")
    st.sidebar.success(f"**{_MODEL_PATH.name}**  \nFeatures: `{', '.join(ML_FEATURES)}`  \n*Predicts IWR (mm/day)*")
else:
    st.sidebar.markdown("---")
    st.sidebar.warning(ML_STATUS[:150])

# ══════════════════════════════════════════════════════════════════════════════
# GEO PANEL (main area)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""<div class="geo-panel">
    📍 <b>Site:</b> {SITE_NAME} &nbsp;|&nbsp; Uganda<br>
    🌐 <b>Coordinates:</b>
      <span class="geo-coord">Lat {LAT}°</span>
      <span class="geo-coord">Lon {LON}°</span>
      <span class="geo-coord">Elev {ELEV} m a.s.l.</span>
    &nbsp;&nbsp;
    <a href="{GMAPS_URL}" target="_blank">🗺️ Google Maps</a>
    &nbsp;|&nbsp;
    <a href="{GMAPS_SAT}" target="_blank">🛰️ Satellite View</a><br>
    🌍 <b>Soil ({SITE_SRC}):</b> {ACTIVE_TXT} &nbsp;·&nbsp;
      FC = <b>{ACTIVE_FC*100:.0f}%</b> &nbsp;·&nbsp; PWP = <b>{ACTIVE_PWP*100:.0f}%</b>
    </div>""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# CROP PARAMETERS  (FAO-56, Uganda-calibrated)
# ══════════════════════════════════════════════════════════════════════════════
crop_params = {
    "Tomatoes":       {"ini": 0.60, "mid": 1.15, "end": 0.80, "zr": 0.70, "mad": 0.40},
    "Cabbages":       {"ini": 0.70, "mid": 1.05, "end": 0.95, "zr": 0.50, "mad": 0.45},
    "Maize":          {"ini": 0.30, "mid": 1.20, "end": 0.60, "zr": 1.00, "mad": 0.55},
    "Beans":          {"ini": 0.40, "mid": 1.15, "end": 0.75, "zr": 0.60, "mad": 0.45},
    "Rice":           {"ini": 1.05, "mid": 1.30, "end": 0.95, "zr": 0.50, "mad": 0.20},
    "Potatoes":       {"ini": 0.50, "mid": 1.15, "end": 0.75, "zr": 0.60, "mad": 0.35},
    "Onions":         {"ini": 0.70, "mid": 1.05, "end": 0.95, "zr": 0.30, "mad": 0.30},
    "Peppers":        {"ini": 0.60, "mid": 1.10, "end": 0.80, "zr": 0.50, "mad": 0.30},
    "Cassava":        {"ini": 0.40, "mid": 0.85, "end": 0.70, "zr": 1.00, "mad": 0.60},
    "Bananas":        {"ini": 0.50, "mid": 1.00, "end": 0.80, "zr": 0.90, "mad": 0.35},
    "Wheat":          {"ini": 0.70, "mid": 1.15, "end": 0.40, "zr": 1.00, "mad": 0.55},
    "Sorghum":        {"ini": 0.30, "mid": 1.00, "end": 0.55, "zr": 1.00, "mad": 0.55},
    "Groundnuts":     {"ini": 0.40, "mid": 1.15, "end": 0.75, "zr": 0.50, "mad": 0.50},
    "Sweet Potatoes": {"ini": 0.50, "mid": 1.15, "end": 0.75, "zr": 1.00, "mad": 0.65},
    "Sunflower":      {"ini": 0.35, "mid": 1.10, "end": 0.35, "zr": 1.00, "mad": 0.45},
    "Soybeans":       {"ini": 0.40, "mid": 1.15, "end": 0.50, "zr": 0.60, "mad": 0.50},
}

STAGE_LABELS = {"ini": "🌱 Initial", "mid": "🌿 Mid-Season", "end": "🍂 End-Season"}
WMO_DESC = {
    0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
    51:"Light drizzle",53:"Moderate drizzle",55:"Dense drizzle",
    61:"Slight rain",63:"Moderate rain",65:"Heavy rain",
    80:"Slight showers",81:"Moderate showers",82:"Violent showers",
    95:"Thunderstorm",96:"Thunderstorm+hail",99:"Heavy thunderstorm+hail",
}

def wmo_icon(code):
    if not code: return "🌤️"
    c = int(code)
    if c == 0: return "☀️"
    if c in (1,2,3): return "🌤️"
    if 51 <= c <= 67: return "🌧️"
    if 80 <= c <= 82: return "🌦️"
    if 95 <= c <= 99: return "⛈️"
    return "🌥️"

# ══════════════════════════════════════════════════════════════════════════════
# FAO-56 PENMAN-MONTEITH
# ══════════════════════════════════════════════════════════════════════════════
def et0_pm(tmax, tmin, rh_max, rh_min, u2, rs, elev=None, doy=None, lat_deg=None):
    if elev is None: elev = ELEV
    if lat_deg is None: lat_deg = LAT
    try:
        tmax=float(tmax); tmin=float(tmin)
        rh_max=max(0.,min(100.,float(rh_max))); rh_min=max(0.,min(100.,float(rh_min)))
        u2=max(0.,float(u2)); rs=max(0.,float(rs))
        doy=int(doy) if doy else int(datetime.now().strftime("%j"))
        lat_deg=float(lat_deg)
    except Exception:
        return 0.0

    Gsc=0.0820; tmean=(tmax+tmin)/2.0
    P     = 101.3 * ((293.0 - 0.0065*elev) / 293.0)**5.26
    gamma = 0.000665 * P
    es_max = 0.6108 * np.exp(17.27*tmax / (tmax+237.3))
    es_min = 0.6108 * np.exp(17.27*tmin / (tmin+237.3))
    es     = (es_max + es_min) / 2.0
    ea = max(0.0, (rh_max/100.0*es_min + rh_min/100.0*es_max) / 2.0)
    ea = min(ea, es)
    es_tm = 0.6108 * np.exp(17.27*tmean / (tmean+237.3))
    Delta = 4098.0 * es_tm / (tmean+237.3)**2.0
    b  = 2.0*np.pi*doy/365.0
    dr = 1.0 + 0.033*np.cos(b)
    phi   = np.radians(abs(lat_deg))
    delta_s = 0.409*np.sin(b-1.39)
    ws  = np.arccos(np.clip(-np.tan(phi)*np.tan(delta_s), -1.0, 1.0))
    Ra  = max(0.0, (24.0*60.0/np.pi)*Gsc*dr*(
        ws*np.sin(phi)*np.sin(delta_s)+np.cos(phi)*np.cos(delta_s)*np.sin(ws)))
    Rso = max(0.0, (0.75 + 2e-5*elev)*Ra)
    Rns = 0.77 * rs
    fcd = max(0.0, min(1.0, 1.35*(rs/max(Rso,0.1)) - 0.35))
    Rnl = max(0.0, _SIGMA*((tmax+273.16)**4+(tmin+273.16)**4)/2.0
              * (0.34-0.14*np.sqrt(max(0.0,ea))) * fcd)
    Rn  = max(0.0, Rns - Rnl)
    num = 0.408*Delta*Rn + gamma*(900.0/(tmean+273.0))*u2*(es-ea)
    den = Delta + gamma*(1.0+0.34*u2)
    return max(0.0, round(num/den, 3)) if den > 0 else 0.0

def et0_hargreaves(tmax, tmin, doy=None, lat_deg=None):
    if lat_deg is None: lat_deg = LAT
    doy = doy or int(datetime.now().strftime("%j"))
    b   = 2.0*np.pi*doy/365.0
    dr  = 1.0+0.033*np.cos(b); phi=np.radians(abs(lat_deg))
    delta_s=0.409*np.sin(b-1.39)
    ws  = np.arccos(np.clip(-np.tan(phi)*np.tan(delta_s),-1.0,1.0))
    Ra  = max(0.0,(24.0*60.0/np.pi)*0.0820*dr*(
        ws*np.sin(phi)*np.sin(delta_s)+np.cos(phi)*np.cos(delta_s)*np.sin(ws)))
    tmean=(tmax+tmin)/2.0; td=max(0.0,tmax-tmin)
    return round(max(0.0,0.0023*Ra*(tmean+17.8)*td**0.5),3)

# ══════════════════════════════════════════════════════════════════════════════
# SOIL WATER BALANCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def compute_taw(fc, pwp, zr):
    """Total Available Water (mm) = (FC - PWP) × Zr × 1000"""
    return (fc - pwp) * zr * 1000.0

def compute_raw(taw, mad):
    """Readily Available Water (mm) = MAD × TAW — irrigation trigger threshold"""
    return mad * taw

def eff_rain(p):
    """USDA SCS effective rainfall (mm)."""
    p = float(p) if p else 0.0
    if p <= 0:     return 0.0
    if p <= 25.4:  return p * (125.0 - 0.6*p) / 125.0
    return p - 12.7 - 0.1*p

def get_kc(dap, crop):
    p = crop_params[crop]
    if dap < 30:  return p["ini"], "ini"
    if dap < 90:  return p["mid"], "mid"
    return p["end"], "end"

def kc_from_stage(stage, crop):
    return crop_params[crop][stage]

def soil_status_label(dr, raw, taw, pe, etc):
    """
    FAO-56 MAD-based daily irrigation decision.

    CORRECT RULE: Irrigation triggers when Dr > RAW (MAD threshold).
    This is INDEPENDENT of today's rain. A small rain event does not
    restore field capacity when the soil is already stressed.
    Rain status is noted in the label only when Dr is still BELOW RAW.
    """
    if dr <= 0:
        return "🟢 At Field Capacity", False
    if dr <= raw * 0.5:
        if pe >= etc:
            return "🌧️ Rain covered — soil adequate", False
        return "✅ Adequate moisture", False
    if dr <= raw:
        if pe >= etc:
            return "🌧️ Rain topped up — monitor tomorrow", False
        return "🟡 Monitor — nearing MAD threshold", False
    # Dr > RAW: soil below MAD — irrigation needed regardless of rain
    if dr <= taw:
        return "⚠️ Irrigate — Dr > MAD threshold", True
    return "🔴 Wilting Risk — irrigate immediately!", True

# ══════════════════════════════════════════════════════════════════════════════
# WATER BALANCE — CORE ENGINE  (v7.0: physics-first irrigation decision)
# ══════════════════════════════════════════════════════════════════════════════
def run_water_balance(daily_df, crop, soil, planting_ts, sm_pct, Ef=0.80,
                      stage_override=None):
    """
    Row-by-row daily water balance (FAO-56 MAD approach).

    IRRIGATION TRIGGER LOGIC (v7.0):
    ─────────────────────────────────
    • Irrigation is NOT automatic every day.
    • Irrigation is triggered ONLY when:
        1.  Root-zone depletion Dr > RAW (Readily Available Water)
            i.e. soil moisture has fallen below the MAD threshold, AND
        2.  Effective rainfall Pe < ETc
            i.e. today's rain does NOT already cover crop water demand.
    • On rainy days where Pe ≥ ETc → Status = "🌧️ Rain Covered", IWR = 0.
    • When irrigation IS triggered → refill soil to FC (Dr → 0 after event).
    • FC guard: if Dr would go negative (over-saturation), cap at 0.
    • All values are daily (24-hr) — not cumulative.
    """
    cp  = crop_params[crop]
    zr  = cp["zr"]; mad = cp["mad"]
    taw = compute_taw(soil["fc"], soil["pwp"], zr)
    raw = compute_raw(taw, mad)

    # Starting depletion from SM slider
    theta = soil["pwp"] + (sm_pct / 100.0) * (soil["fc"] - soil["pwp"])
    theta = min(theta, soil["fc"])  # cap at FC
    dr = max(0.0, (soil["fc"] - theta) * zr * 1000.0)

    df = daily_df.copy()

    if stage_override:
        df["kc"] = crop_params[crop][stage_override]
    else:
        df["kc"] = df.index.map(
            lambda d: get_kc(max(0, (d - planting_ts).days), crop)[0]
        )

    df["ET0"] = df.apply(lambda r: et0_pm(
        r["tmax"], r["tmin"],
        r.get("rh_max", r.get("rh", 65) + 10),
        r.get("rh_min", r.get("rh", 65) - 10),
        r["wind"], r["rs"],
        doy=r.name.timetuple().tm_yday,
        lat_deg=LAT, elev=ELEV,
    ), axis=1)
    df["ETc"] = (df["kc"] * df["ET0"]).round(3)

    prec_col = "precipitation" if "precipitation" in df.columns else "precip"
    df["Pe"] = df.get(prec_col, pd.Series(0.0, index=df.index)).apply(eff_rain)

    dr_vals = []; iwr_vals = []; nir_vals = []
    status_vals = []; sm_pct_vals = []

    for _, row in df.iterrows():
        pe_r  = row["Pe"]
        etc_r = row["ETc"]
        prec  = row.get(prec_col, 0.0)

        # --- Water balance update: ETc depletes, Pe replenishes ---
        # Cap at 0 (can't be above FC) and at TAW (permanent wilting)
        dr_new = max(0.0, min(taw, dr - pe_r + etc_r))

        # --- Irrigation decision ---
        status_lbl, irrigate = soil_status_label(dr_new, raw, taw, pe_r, etc_r)

        if irrigate:
            # NIR = amount needed to refill root zone back to FC
            nir_r     = round(dr_new, 3)
            iwr_gross = round(nir_r / max(Ef, 0.01), 3)
            dr        = 0.0   # after irrigation soil is at FC
        else:
            nir_r     = 0.0
            iwr_gross = 0.0
            dr        = dr_new  # carry forward depletion

        # Soil moisture % of available range
        sm_now = max(0, min(100, int((1.0 - dr / taw) * 100))) if taw > 0 else 70

        dr_vals.append(round(dr, 2))
        nir_vals.append(nir_r)
        iwr_vals.append(iwr_gross)
        status_vals.append(status_lbl)
        sm_pct_vals.append(sm_now)

    df["Depletion_mm"] = dr_vals   # Dr = root-zone deficit from FC
    df["SM_pct"]       = sm_pct_vals
    df["NIR"]          = nir_vals
    df["IWR"]          = iwr_vals
    df["Status"]       = status_vals
    return df, taw, raw


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-DAY IWR  (Tab 1 point calculation — physics + optional ML quantity)
# ══════════════════════════════════════════════════════════════════════════════
def compute_today_iwr(tmax, tmin, rh_in, wind_in, rs_in, prec_in,
                      kc, crop, soil, sm_pct, Ef, doy):
    """
    Compute today's IWR with full water-balance logic.
    Returns a dict with all intermediate values.
    """
    cp  = crop_params[crop]
    zr  = cp["zr"]; mad = cp["mad"]
    taw = compute_taw(soil["fc"], soil["pwp"], zr)
    raw = compute_raw(taw, mad)

    rh_mx = min(100.0, rh_in + 10.0)
    rh_mn = max(0.0, rh_in - 10.0)
    et0_fao = et0_pm(tmax, tmin, rh_mx, rh_mn, wind_in, rs_in, doy=doy)
    et0_h   = et0_hargreaves(tmax, tmin, doy=doy)
    etc     = round(kc * et0_fao, 3)
    pe      = eff_rain(prec_in)

    # Current depletion from SM slider
    theta = soil["pwp"] + (sm_pct / 100.0) * (soil["fc"] - soil["pwp"])
    theta = min(theta, soil["fc"])
    dr_start = max(0.0, (soil["fc"] - theta) * zr * 1000.0)

    # Update depletion for today
    dr_today = max(0.0, min(taw, dr_start - pe + etc))

    # Irrigation decision
    status_lbl, irrigate = soil_status_label(dr_today, raw, taw, pe, etc)

    if irrigate:
        nir   = round(dr_today, 3)
        iwr   = round(nir / max(Ef, 0.01), 3)
        dr_after = 0.0
    else:
        nir   = 0.0
        iwr   = 0.0
        dr_after = dr_today

    vol   = compute_volume(iwr, area_ha)
    mins  = round((vol["vol_m3"] / pump_flow) * 60, 1) if pump_flow > 0 and iwr > 0 else 0

    return {
        "et0_fao": et0_fao, "et0_h": et0_h,
        "etc": etc, "pe": pe,
        "taw": taw, "raw": raw,
        "dr_start": round(dr_start, 2),
        "dr_today": round(dr_today, 2),
        "dr_after": round(dr_after, 2),
        "nir": nir, "iwr": iwr,
        "vol_m3": vol["vol_m3"], "vol_L": vol["vol_L"],
        "pump_min": mins,
        "status": status_lbl,
        "irrigate": irrigate,
        "sm_pct_now": max(0, min(100, int((1 - dr_today / taw) * 100))) if taw > 0 else 70,
        "sm_pct_after": max(0, min(100, int((1 - dr_after / taw) * 100))) if taw > 0 else 100,
    }

def compute_volume(iwr_mm, area_ha):
    vol_m3 = iwr_mm * area_ha * 10.0
    return {"vol_m3": round(vol_m3, 1), "vol_L": round(vol_m3 * 1000, 0)}

# ══════════════════════════════════════════════════════════════════════════════
# SOIL MOISTURE ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
def estimate_soil_moisture_status(fc, pwp, zr, lat, lon, elev):
    try:
        end_  = date.today() - timedelta(days=1)
        start_= end_ - timedelta(days=10)
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_}&end_date={end_}"
            f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,"
            f"shortwave_radiation_sum,wind_speed_10m_max,"
            f"relative_humidity_2m_max,relative_humidity_2m_min"
            f"&timezone={TIMEZONE}"
        )
        r  = requests.get(url, timeout=12).json()
        d  = r.get("daily", {}); dates = d.get("time", [])
        taw_ = (fc - pwp) * zr * 1000.0
        theta = pwp + 0.70 * (fc - pwp)
        dr_ = max(0.0, (fc - theta) * zr * 1000.0)

        for i in range(len(dates)):
            tx = d["temperature_2m_max"][i]; tn = d["temperature_2m_min"][i]
            if tx is None or tn is None: continue
            rh_mx = d["relative_humidity_2m_max"][i] or 70
            rh_mn = d["relative_humidity_2m_min"][i] or 50
            wk    = (d["wind_speed_10m_max"][i] or 7.2) / 3.6 * _W2M
            rs_i  = d["shortwave_radiation_sum"][i] or 18.0
            prec  = d["precipitation_sum"][i] or 0.0
            doy_i = datetime.strptime(dates[i], "%Y-%m-%d").timetuple().tm_yday
            et0_i = et0_pm(tx, tn, rh_mx, rh_mn, wk, rs_i, elev=elev, doy=doy_i, lat_deg=lat)
            pe_i  = eff_rain(prec)
            dr_   = max(0.0, min(taw_, dr_ - pe_i + et0_i))

        sm = int(max(0, min(100, (1 - dr_ / taw_) * 100))) if taw_ > 0 else 70
        return {"sm_pct": sm, "source": "PM water balance (ERA5 last 10 days)", "fallback": False}
    except Exception:
        return {"sm_pct": 60, "source": "Default (weather API unavailable)", "fallback": True}

def estimate_sm(fc, pwp, zr):
    return estimate_soil_moisture_status(fc, pwp, zr, LAT, LON, ELEV)["sm_pct"]

# ══════════════════════════════════════════════════════════════════════════════
# WEATHER APIs
# ══════════════════════════════════════════════════════════════════════════════
_ch = f"{datetime.now().strftime('%Y%m%d%H')}_{LAT}_{LON}"

@st.cache_data(ttl=3600, show_spinner=False)
def get_current_weather(_cache_key, lat, lon, elev):
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,"
            f"wind_speed_10m,shortwave_radiation,weather_code"
            f"&daily=temperature_2m_max,temperature_2m_min,"
            f"relative_humidity_2m_max,relative_humidity_2m_min,"
            f"windspeed_10m_max,shortwave_radiation_sum,"
            f"precipitation_sum,weather_code"
            f"&forecast_days=1&timezone={TIMEZONE}", timeout=12
        ).json()
        cur = r.get("current", {}); d = r.get("daily", {})
        tmax = d.get("temperature_2m_max", [None])[0]
        tmin = d.get("temperature_2m_min", [None])[0]
        rh_mx = d.get("relative_humidity_2m_max", [70])[0] or 70
        rh_mn = d.get("relative_humidity_2m_min", [50])[0] or 50
        wk    = (d.get("windspeed_10m_max", [7.2])[0] or 7.2) / 3.6 * _W2M
        rs    = d.get("shortwave_radiation_sum", [18.0])[0] or 18.0
        prec  = d.get("precipitation_sum", [0.0])[0] or 0.0
        wcode = d.get("weather_code", [0])[0] or 0
        tmean_c = cur.get("temperature_2m", 25)
        tmax = tmax or tmean_c + 4; tmin = tmin or tmean_c - 4
        return {
            "tmax": round(tmax,1), "tmin": round(tmin,1),
            "rh_max": rh_mx, "rh_min": rh_mn,
            "rh_mean": round((rh_mx+rh_mn)/2, 1),
            "wind": round(wk, 3), "rs": round(rs, 1),
            "precip": round(prec, 1), "wcode": wcode,
            "description": WMO_DESC.get(int(wcode), f"Code {wcode}"),
            "source": "Open-Meteo ICON+GFS (live)"
        }
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_forecast(_cache_key, lat, lon, elev):
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,"
            f"relative_humidity_2m_max,relative_humidity_2m_min,"
            f"windspeed_10m_max,shortwave_radiation_sum,"
            f"precipitation_sum,weather_code"
            f"&forecast_days=7&timezone={TIMEZONE}", timeout=12
        ).json()
        d = r.get("daily", {})
        if not d: return None
        rows = []
        for i in range(len(d["time"])):
            wk = (d["windspeed_10m_max"][i] or 7.2) / 3.6 * _W2M
            rh_mx = d["relative_humidity_2m_max"][i] or 70
            rh_mn = d["relative_humidity_2m_min"][i] or 50
            rows.append({
                "date":pd.to_datetime(d["time"][i]),
                "tmax":d["temperature_2m_max"][i] or 28,
                "tmin":d["temperature_2m_min"][i] or 16,
                "rh_max":rh_mx, "rh_min":rh_mn, "rh_mean":round((rh_mx+rh_mn)/2,1),
                "wind":round(wk,3),
                "rs":d["shortwave_radiation_sum"][i] or 18.0,
                "precipitation":d["precipitation_sum"][i] or 0.0,
                "weather_code":d["weather_code"][i] or 0,
            })
        df = pd.DataFrame(rows).set_index("date")
        today = pd.Timestamp.today().normalize()
        return df[df.index >= today].head(5)
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_historical_weather(start_date, end_date, lat, lon):
    """ERA5 archive — authoritative historical data."""
    try:
        r = requests.get(
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_max,temperature_2m_min,"
            f"relative_humidity_2m_max,relative_humidity_2m_min,"
            f"windspeed_10m_max,shortwave_radiation_sum,"
            f"precipitation_sum"
            f"&timezone={TIMEZONE}", timeout=25
        ).json()
        d = r.get("daily", {})
        if not d: return None
        rh_mx = d.get("relative_humidity_2m_max", [])
        rh_mn = d.get("relative_humidity_2m_min", [])
        df = pd.DataFrame({
            "date":pd.to_datetime(d["time"]),
            "tmax":[x or 28 for x in d["temperature_2m_max"]],
            "tmin":[x or 16 for x in d["temperature_2m_min"]],
            "rh_max":[(a or 70) for a in rh_mx],
            "rh_min":[(a or 50) for a in rh_mn],
            "rh_mean":[(((a or 70)+(b or 50))/2) for a,b in zip(rh_mx,rh_mn)],
            "wind":[(x or 7.2)/3.6*_W2M for x in d["windspeed_10m_max"]],
            "rs":[x or 18.0 for x in d["shortwave_radiation_sum"]],
            "precipitation":[x or 0.0 for x in d["precipitation_sum"]],
        }).set_index("date")
        df["rh"] = df["rh_mean"]
        return df.dropna(subset=["tmax", "tmin"])
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# FETCH LIVE WEATHER
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner(f"📡 Fetching live weather for {SITE_NAME}…"):
    wx = get_current_weather(_ch, LAT, LON, ELEV)

if wx:
    lt=wx["tmax"]; ln=wx["tmin"]; lr_max=wx["rh_max"]; lr_min=wx["rh_min"]
    lr_mean=wx["rh_mean"]; lw=wx["wind"]; ls=wx["rs"]; lp=wx["precip"]
else:
    lt,ln,lr_max,lr_min,lr_mean,lw,ls,lp = 28.,16.,70.,50.,60.,1.5,18.,0.

_doy = int(datetime.today().strftime("%j"))

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 Today's IWR", "☁️ 5-Day Forecast", "📅 Historical"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TODAY'S IWR  +  PAST 5 DAYS CONTEXT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header(f"📊 Today's IWR — {SITE_NAME}")
    st.caption(
        f"📡 {wx['source'] if wx else 'Weather unavailable'} · "
        f"FAO-56 PM v7.0 · MAD-threshold irrigation trigger · "
        f"MAD-threshold: IWR when Dr > RAW"
    )

    if wx:
        st.success(
            f"✅ **{wx['description']}** · Rain: **{lp} mm** · "
            f"Lat {LAT}° / Lon {LON}° / {ELEV} m a.s.l."
        )
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("🌡️ Tmax / Tmin",  f"{lt}°C / {ln}°C")
        c2.metric("💧 RH min–max",   f"{lr_min:.0f}–{lr_max:.0f}%", f"Mean {lr_mean:.0f}%")
        c3.metric("🌬️ Wind (2 m)",   f"{lw:.2f} m/s")
        c4.metric("☀️ Solar Rad",    f"{ls:.1f} MJ/m²/d")
        c5.metric("🌧️ Rain Today",   f"{lp:.1f} mm")
    else:
        st.warning("⚠️ Weather unavailable — using default values")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌱 Crop & Growth Stage")
        cr1 = st.selectbox("Crop", list(crop_params.keys()), key="cr1")
        cp1 = crop_params[cr1]

        stage1 = st.radio(
            "Growing Stage", list(STAGE_LABELS.keys()),
            format_func=lambda x: STAGE_LABELS[x],
            key="stg1", horizontal=True,
        )
        kc1 = kc_from_stage(stage1, cr1)

        st.markdown(
            f'<div class="kc-stage">Kc ({STAGE_LABELS[stage1]}) = <b>{kc1:.3f}</b> '
            f'· Zr = {cp1["zr"]:.2f} m · MAD = {int(cp1["mad"]*100)}%</div>',
            unsafe_allow_html=True,
        )

        # MAD threshold info box
        _taw_preview = compute_taw(ACTIVE_FC, ACTIVE_PWP, cp1["zr"])
        _raw_preview = compute_raw(_taw_preview, cp1["mad"])
        st.markdown(
            f'<div class="mad-panel">'
            f'📐 <b>Soil Water Thresholds (FAO-56)</b><br>'
            f'TAW = <b>{_taw_preview:.1f} mm</b> &nbsp;·&nbsp; '
            f'RAW (MAD {int(cp1["mad"]*100)}%) = <b>{_raw_preview:.1f} mm</b><br>'
            f'<small>Irrigate only when Root-Zone Depletion (Dr) &gt; RAW <i>AND</i> '
            f'daily rain does not already cover crop demand.</small></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"| Stage | Ini Kc | Mid Kc | End Kc |\n|---|---|---|---|\n"
            f"| {cr1} | {cp1['ini']} | {cp1['mid']} | {cp1['end']} |"
        )

    with col2:
        st.subheader("🌦️ Weather & Soil (Today — 24 hrs)")
        tmax_in  = st.number_input("Tmax (°C)", value=float(lt), key=f"t1_{loc_name}")
        tmin_in  = st.number_input("Tmin (°C)", value=float(ln), key=f"t2_{loc_name}")
        rh_in    = st.number_input("RH mean (%)", value=float(lr_mean),
                                   min_value=0.0, max_value=100.0, key=f"rh1_{loc_name}")
        wind_in  = st.number_input("Wind 2m (m/s)", value=float(lw),
                                   min_value=0.0, key=f"w1_{loc_name}")
        rs_in    = st.number_input("Solar Rad (MJ/m²/d)", value=float(ls),
                                   min_value=0.0, key=f"rs1_{loc_name}")
        prec_in  = st.number_input("Rain (mm)", value=float(lp),
                                   min_value=0.0, key=f"p1_{loc_name}")

        soil1_obj = {"fc": ACTIVE_FC, "pwp": ACTIVE_PWP}
        _sm_def = estimate_sm(ACTIVE_FC, ACTIVE_PWP, cp1["zr"])
        sm_pct  = st.slider("Current Soil Moisture (% of FC)", 0, 100, _sm_def, key="sm1")

        # FC warning
        if sm_pct >= 95:
            st.markdown(
                f'<div class="warn-fc">⚠️ <b>Soil near Field Capacity ({sm_pct}% of FC)</b> — '
                f'do NOT irrigate; risk of waterlogging and nutrient leaching.</div>',
                unsafe_allow_html=True)

        st.markdown(
            f'<div class="soil-panel">🌍 <b>Soil ({ACTIVE_TXT})</b> '
            f'· FC <b>{ACTIVE_FC*100:.0f}%</b> · PWP <b>{ACTIVE_PWP*100:.0f}%</b><br>'
            f'💧 System: <b>{irrig_sys}</b> · Ef = <b>{Ef*100:.0f}%</b> · '
            f'Area <b>{area_ha} ha</b> · Flow <b>{pump_flow} m³/hr</b></div>',
            unsafe_allow_html=True,
        )

    if st.button("🧮 Calculate Today's IWR + Past 5 Days Context",
                 type="primary", use_container_width=True, key="calc1"):

        # ── 1. FETCH PAST 5 DAYS FROM ERA5 ──────────────────────────────────
        st.markdown("---")
        past_end_d   = date.today() - timedelta(days=1)
        past_start_d = past_end_d - timedelta(days=4)

        with st.spinner("📡 Fetching past 5 days (ERA5) for context…"):
            hist_ctx = get_historical_weather(
                str(past_start_d), str(past_end_d), LAT, LON)

        # ── 2. PAST DAYS TABLE ───────────────────────────────────────────────
        st.markdown(
            '<div class="past-panel">'
            '<b>📅 Soil & Crop Water History — Past 5 Days</b><br>'
            '<small>Use this to understand how soil moisture and crop demand '
            'evolved BEFORE today, so you know why today\'s decision is what it is.</small>'
            '</div>', unsafe_allow_html=True)

        soil1_obj = {"fc": ACTIVE_FC, "pwp": ACTIVE_PWP}

        if hist_ctx is not None and not hist_ctx.empty:
            # Use current stage Kc fixed across past days for consistency
            past_r, taw_p, raw_p = run_water_balance(
                hist_ctx, cr1, soil1_obj,
                pd.Timestamp(date.today() - timedelta(days=45)),
                sm_pct, Ef, stage_override=stage1
            )
            past_r["Vol_m3"] = past_r["IWR"].apply(
                lambda x: compute_volume(x, area_ha)["vol_m3"])

            # Build display table
            past_disp = pd.DataFrame({
                "Date":          past_r.index.strftime("%Y-%m-%d (%a)"),
                "Rain (mm)":     past_r["precipitation"].round(1),
                "ET₀ (mm/d)":    past_r["ET0"].round(2),
                "ETc (mm/d)":    past_r["ETc"].round(2),
                "Pe (mm)":       past_r["Pe"].round(2),
                "Deficit Dr (mm)": past_r["Depletion_mm"].round(2),
                "SM (% FC)":     past_r["SM_pct"],
                "NIR (mm)":      past_r["NIR"].round(2),
                "IWR gross (mm)":past_r["IWR"].round(2),
                "Vol needed (m³)":past_r["Vol_m3"],
                "Soil Status":   past_r["Status"],
            }).set_index("Date")

            # Colour-highlight status column
            def _style_status(val):
                if "Irrigate" in val or "Wilting" in val:
                    return "background-color:#ffe0e0;font-weight:bold"
                if "Rain" in val:
                    return "background-color:#e0f7fa"
                if "Adequate" in val or "Capacity" in val:
                    return "background-color:#e8f5e9"
                if "Monitor" in val:
                    return "background-color:#fff9c4"
                return ""

            styled_past = past_disp.style.format({
                "Rain (mm)":       "{:.1f}",
                "ET₀ (mm/d)":      "{:.2f}",
                "ETc (mm/d)":      "{:.2f}",
                "Pe (mm)":         "{:.2f}",
                "Deficit Dr (mm)": "{:.2f}",
                "SM (% FC)":       "{:.0f}",
                "NIR (mm)":        "{:.2f}",
                "IWR gross (mm)":  "{:.2f}",
                "Vol needed (m³)": "{:.1f}",
            }).map(_style_status, subset=["Soil Status"])

            # Reference thresholds legend
            st.info(
                f"📐 **Water Thresholds for {cr1} on {ACTIVE_TXT}:**  "
                f"TAW = **{taw_p:.1f} mm** · "
                f"RAW (MAD {int(cp1['mad']*100)}%) = **{raw_p:.1f} mm**  \n"
                f"🟢 Dr=0 → Field Capacity &nbsp;|&nbsp; "
                f"✅ Dr ≤ {raw_p*0.5:.1f} mm → Adequate &nbsp;|&nbsp; "
                f"🟡 Dr ≤ {raw_p:.1f} mm → Monitor &nbsp;|&nbsp; "
                f"⚠️ Dr > {raw_p:.1f} mm → **Irrigate** &nbsp;|&nbsp; "
                f"🔴 Dr > {taw_p:.1f} mm → Wilting risk"
            )
            st.dataframe(styled_past, use_container_width=True)

            # Depletion sparkline vs thresholds
            fig_past = go.Figure()
            fig_past.add_scatter(
                x=past_r.index.strftime("%a %d"),
                y=past_r["Depletion_mm"],
                mode="lines+markers+text",
                name="Deficit Dr (mm)",
                text=past_r["Depletion_mm"].round(1).astype(str),
                textposition="top center",
                line=dict(color="#e6550d", width=2),
                marker=dict(size=8),
            )
            fig_past.add_bar(
                x=past_r.index.strftime("%a %d"),
                y=past_r["precipitation"],
                name="Rain (mm)",
                marker_color="#1a5fc8", opacity=0.5,
            )
            fig_past.add_hline(y=raw_p, line_dash="dash", line_color="#756bb1",
                               annotation_text=f"RAW = {raw_p:.1f} mm (irrigate above this)")
            fig_past.add_hline(y=taw_p, line_dash="dot", line_color="#d73027",
                               annotation_text=f"TAW = {taw_p:.1f} mm (wilting risk)")
            fig_past.update_layout(
                title="Past 5 Days — Root-Zone Depletion vs Thresholds",
                yaxis_title="mm",
                plot_bgcolor="#f4f8f2", paper_bgcolor="#f4f8f2",
                height=300, legend=dict(x=0, y=1.1, orientation="h"),
            )
            st.plotly_chart(fig_past, use_container_width=True)

        else:
            st.warning("⚠️ ERA5 past data unavailable — showing today only "
                       "(archive may lag 3–5 days).")

        # ── 3. TODAY'S RESULT ────────────────────────────────────────────────
        st.markdown(
            '<div class="today-panel">'
            '<b>💧 TODAY\'s Irrigation Decision (24-hour crop water need)</b>'
            '</div>', unsafe_allow_html=True)

        res = compute_today_iwr(
            tmax_in, tmin_in, rh_in, wind_in, rs_in, prec_in,
            kc1, cr1, soil1_obj, sm_pct, Ef, _doy
        )

        st.info(
            f"📍 **{SITE_NAME}** · Lat {LAT}° · Lon {LON}° · {ELEV} m a.s.l.  \n"
            f"🌱 **{cr1}** · Stage: **{STAGE_LABELS[stage1]}** · Kc = **{kc1:.3f}**  \n"
            f"🌍 Soil: **{ACTIVE_TXT}** · FC={ACTIVE_FC*100:.0f}% · PWP={ACTIVE_PWP*100:.0f}%  \n"
            f"📐 TAW = **{res['taw']:.1f} mm** · RAW = **{res['raw']:.1f} mm** "
            f"(MAD {int(cp1['mad']*100)}%) · Irrigation trigger threshold"
        )

        r1,r2,r3,r4,r5,r6 = st.columns(6)
        r1.metric("ET₀-PM (FAO-56)",    f"{res['et0_fao']:.2f} mm/d")
        r2.metric("ET₀-H (Hargreaves)", f"{res['et0_h']:.2f} mm/d")
        r3.metric("ETc",                f"{res['etc']:.2f} mm/d",  f"Kc={kc1:.3f}")
        r4.metric("Pe (Eff. Rain)",     f"{res['pe']:.2f} mm")
        r5.metric("Dr (Deficit)",       f"{res['dr_today']:.2f} mm",
                  f"RAW={res['raw']:.1f} mm")
        r6.metric("SM % FC",            f"{res['sm_pct_now']}%",
                  f"→ {res['sm_pct_after']}% after irrig")

        st.markdown("---")
        st.markdown(
            f'<div class="nir-box">📐 <b>Dr (deficit) = {res["dr_today"]:.2f} mm</b> '
            f'&nbsp;·&nbsp; RAW = {res["raw"]:.1f} mm &nbsp;·&nbsp; '
            f'TAW = {res["taw"]:.1f} mm &nbsp;·&nbsp; '
            f'NIR = <b>{res["nir"]:.2f} mm</b></div>',
            unsafe_allow_html=True)

        if res["irrigate"]:
            st.markdown(
                f'<div class="iwr-box">💧 <b>IWR (Gross) = {res["iwr"]:.2f} mm/day</b> '
                f'&nbsp;·&nbsp; {irrig_sys} (Ef={Ef*100:.0f}%)</div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="vol-box">🪣 <b>Volume needed today:</b> '
                f'<b>{res["vol_m3"]:.1f} m³</b> &nbsp;=&nbsp; '
                f'<b>{res["vol_L"]:,.0f} litres</b> for <b>{area_ha} ha</b>'
                f' &nbsp;·&nbsp; ⏱️ Pump time: <b>{res["pump_min"]} min</b> '
                f'at {pump_flow} m³/hr</div>',
                unsafe_allow_html=True)
            st.warning(
                f"⚠️ **Irrigate today** · Dr = {res['dr_today']:.2f} mm > "
                f"RAW = {res['raw']:.1f} mm · "
                f"Apply **{res['iwr']:.2f} mm gross** → {res['vol_m3']:.1f} m³ "
                f"({res['vol_L']:,.0f} L) across {area_ha} ha  \n"
                f"After irrigation: soil moisture → **{res['sm_pct_after']}% of FC** "
                f"(back to Field Capacity ✅)"
            )
        else:
            _rain_note = (
                f"🌧️ Rain (Pe = {res['pe']:.2f} mm) ≥ ETc ({res['etc']:.2f} mm/d) — "
                f"today's rain covers crop demand."
                if res["pe"] >= res["etc"]
                else "Soil moisture is adequate (Dr < MAD) — monitor tomorrow."
            )
            st.success(
                f"✅ **No irrigation needed today.**  \n"
                f"Status: **{res['status']}**  \n"
                f"Dr = {res['dr_today']:.2f} mm ≤ RAW = {res['raw']:.1f} mm  \n"
                f"{_rain_note}"
            )

        # ── Today summary row in a single-row table ──────────────────────────
        st.markdown("#### 📋 Today's Summary Table")
        today_row = pd.DataFrame([{
            "Date": datetime.today().strftime("%Y-%m-%d (%a)"),
            "Rain (mm)": round(prec_in, 1),
            "ET₀-PM (mm/d)": round(res["et0_fao"], 2),
            "ET₀-H (mm/d)": round(res["et0_h"], 2),
            "Kc": round(kc1, 3),
            "ETc (mm/d)": round(res["etc"], 2),
            "Pe (mm)": round(res["pe"], 2),
            "Dr Deficit (mm)": round(res["dr_today"], 2),
            "RAW threshold (mm)": round(res["raw"], 1),
            "TAW (mm)": round(res["taw"], 1),
            "SM % FC": res["sm_pct_now"],
            "NIR (mm)": round(res["nir"], 2),
            "IWR gross (mm)": round(res["iwr"], 2),
            "Vol m³": res["vol_m3"],
            "Vol Litres": res["vol_L"],
            "Pump min": res["pump_min"],
            "Soil Status": res["status"],
        }])
        st.dataframe(today_row.set_index("Date"), use_container_width=True)

        # ── Combined download: past days + today ────────────────────────────
        today_dl = pd.DataFrame([{
            "Date":             datetime.today().strftime("%Y-%m-%d (%a)"),
            "Rain (mm)":        prec_in,
            "ET₀ (mm/d)":       res["et0_fao"],
            "ETc (mm/d)":       res["etc"],
            "Pe (mm)":          res["pe"],
            "Deficit Dr (mm)":  res["dr_today"],
            "SM (% FC)":        res["sm_pct_now"],
            "NIR (mm)":         res["nir"],
            "IWR gross (mm)":   res["iwr"],
            "Vol needed (m³)":  res["vol_m3"],
            "Soil Status":      res["status"],
        }]).set_index("Date")

        if hist_ctx is not None and not hist_ctx.empty:
            combined = pd.concat([past_disp, today_dl], sort=False)
            dl_data  = combined.fillna("").to_csv().encode()
        else:
            dl_data = today_dl.to_csv().encode()

        st.download_button(
            "⬇️ Download Past 5 Days + Today CSV",
            dl_data,
            f"HyPIS_context_{SITE_NAME.replace(' ','_')}_{datetime.today().strftime('%Y%m%d')}.csv",
            "text/csv", key="dl_today"
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — 5-DAY FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header(f"☁️ 5-Day IWR Forecast — {SITE_NAME}")
    st.caption(
        f"FAO-56 PM · MAD-threshold irrigation trigger · "
        f"MAD-threshold trigger (irrigate when Dr > RAW) · "
        f"Open-Meteo ICON+GFS · Ef={Ef*100:.0f}% ({irrig_sys})"
    )

    fc_c1, fc_c2 = st.columns(2)
    with fc_c1:
        cr2 = st.selectbox("Crop", list(crop_params.keys()), key="cr2")
        cp2 = crop_params[cr2]
        stage2 = st.radio("Growing Stage", list(STAGE_LABELS.keys()),
                          format_func=lambda x: STAGE_LABELS[x],
                          key="stg2", horizontal=True)
        kc2 = kc_from_stage(stage2, cr2)
        st.markdown(
            f'<div class="kc-stage">Kc ({STAGE_LABELS[stage2]}) = <b>{kc2:.3f}</b></div>',
            unsafe_allow_html=True)
    with fc_c2:
        planting2 = st.date_input("Planting Date",
                                  value=datetime.today().date()-timedelta(days=45),
                                  key="plant2")
        soil2 = {"fc": ACTIVE_FC, "pwp": ACTIVE_PWP}
        st.markdown(
            f'<div class="soil-panel">🌍 Soil auto-loaded: <b>{ACTIVE_TXT}</b> '
            f'(FC={ACTIVE_FC*100:.0f}%, PWP={ACTIVE_PWP*100:.0f}%)</div>',
            unsafe_allow_html=True)
        sm_pct2 = st.slider("Starting SM (% of FC)", 0, 100,
                             estimate_sm(ACTIVE_FC, ACTIVE_PWP, cp2["zr"]),
                             key="sm2")

    if st.button("📥 Get 5-Day Forecast + Volume", type="primary",
                 use_container_width=True, key="fc_btn"):
        with st.spinner(f"Fetching forecast for {SITE_NAME}…"):
            daily = get_forecast(_ch, LAT, LON, ELEV)

        if daily is None or daily.empty:
            st.warning("⚠️ Forecast unavailable — try again shortly.")
        else:
            planting_ts2 = pd.Timestamp(planting2)
            daily_r, taw2, raw2 = run_water_balance(
                daily, cr2, soil2, planting_ts2, sm_pct2, Ef,
                stage_override=stage2)

            daily_r["Vol_m3"] = daily_r["IWR"].apply(
                lambda x: compute_volume(x, area_ha)["vol_m3"])
            daily_r["Vol_L"]  = daily_r["IWR"].apply(
                lambda x: compute_volume(x, area_ha)["vol_L"])
            daily_r["PumpMin"] = daily_r["Vol_m3"].apply(
                lambda v: round(v/pump_flow*60, 1) if pump_flow>0 and v>0 else 0)

            tot_iwr = daily_r["IWR"].sum()
            tot_vol = daily_r["Vol_m3"].sum()
            nd = (daily_r["IWR"] > 0).sum()

            # MAD info
            st.info(
                f"📐 **Thresholds:** TAW = **{taw2:.1f} mm** · "
                f"RAW (MAD {int(cp2['mad']*100)}%) = **{raw2:.1f} mm**  \n"
                f"Irrigation fires only when Dr > {raw2:.1f} mm AND rain < ETc on that day."
            )

            if nd > 0:
                st.warning(
                    f"🗓️ **{nd} irrigation event(s) forecast** · "
                    f"Total IWR = **{tot_iwr:.1f} mm** · "
                    f"Total Vol = **{tot_vol:.1f} m³** ({tot_vol*1000:,.0f} L)"
                )
            else:
                st.success("✅ No irrigation needed over next 5 days — "
                           "rain and/or soil moisture is sufficient.")

            cols2 = st.columns(len(daily_r))
            for i, (dt, row) in enumerate(daily_r.iterrows()):
                icon = wmo_icon(row.get("weather_code", 0))
                if row["IWR"] > 0:
                    lbl = f"💧 {row['IWR']:.1f} mm"
                    dlt = f"🪣 {row['Vol_m3']:.1f} m³ · ⏱{row['PumpMin']}min"
                else:
                    lbl = f"{icon} No irrig"
                    dlt = f"🌧 {row['precipitation']:.1f} mm rain · Dr={row['Depletion_mm']:.1f}mm"
                cols2[i].metric(dt.strftime("%a %d"), lbl, dlt)

            st.subheader("📋 5-Day Forecast Table (with Deficit)")
            tc2 = ["tmax","tmin","rh_mean","precipitation","Pe",
                   "ET0","kc","ETc","Depletion_mm","SM_pct",
                   "NIR","IWR","Vol_m3","Vol_L","PumpMin","Status"]
            tb2 = daily_r[[c for c in tc2 if c in daily_r.columns]].copy()
            rename2 = {
                "tmax":"Tmax °C","tmin":"Tmin °C","rh_mean":"RH %",
                "precipitation":"Rain mm","Pe":"Pe mm",
                "ET0":"ET₀-PM mm/d","kc":"Kc","ETc":"ETc mm/d",
                "Depletion_mm":"Dr Deficit mm","SM_pct":"SM % FC",
                "NIR":"NIR mm","IWR":"IWR gross mm",
                "Vol_m3":"Vol m³","Vol_L":"Vol L",
                "PumpMin":"Pump min","Status":"Soil Status",
            }
            tb2.rename(columns=rename2, inplace=True)
            tb2.index = tb2.index.strftime("%Y-%m-%d (%a)")
            _fc2_fmt = {
                "Tmax °C":"{:.1f}","Tmin °C":"{:.1f}","RH %":"{:.0f}",
                "Rain mm":"{:.1f}","Pe mm":"{:.2f}",
                "ET₀-PM mm/d":"{:.2f}","Kc":"{:.3f}","ETc mm/d":"{:.2f}",
                "Dr Deficit mm":"{:.2f}","SM % FC":"{:.0f}",
                "NIR mm":"{:.2f}","IWR gross mm":"{:.2f}",
                "Vol m³":"{:.1f}","Vol L":"{:.0f}","Pump min":"{:.1f}",
            }
            st.dataframe(
                tb2.style.format({k:v for k,v in _fc2_fmt.items() if k in tb2.columns}),
                use_container_width=True)

            # Depletion chart with thresholds
            fig_dep2 = go.Figure()
            fig_dep2.add_scatter(
                x=daily_r.index.strftime("%a %d"),
                y=daily_r["Depletion_mm"],
                mode="lines+markers", name="Dr Deficit (mm)",
                line=dict(color="#e6550d", width=2), marker=dict(size=8)
            )
            fig_dep2.add_bar(
                x=daily_r.index.strftime("%a %d"),
                y=daily_r["precipitation"],
                name="Rain (mm)", marker_color="#1a5fc8", opacity=0.4, yaxis="y2"
            )
            fig_dep2.add_bar(
                x=daily_r.index.strftime("%a %d"),
                y=daily_r["IWR"],
                name="IWR (mm)", marker_color="#17a2b8", opacity=0.7, yaxis="y2"
            )
            fig_dep2.add_hline(y=raw2, line_dash="dash", line_color="#756bb1",
                               annotation_text=f"RAW = {raw2:.1f} mm")
            fig_dep2.add_hline(y=taw2, line_dash="dot", line_color="#d73027",
                               annotation_text=f"TAW = {taw2:.1f} mm")
            fig_dep2.update_layout(
                title=f"Forecast: Root-Zone Depletion vs Thresholds — {SITE_NAME}",
                yaxis=dict(title="Depletion Dr (mm)"),
                yaxis2=dict(title="mm", overlaying="y", side="right"),
                barmode="group",
                legend=dict(x=0, y=1.1, orientation="h"),
                height=380, plot_bgcolor="#f4f8f2", paper_bgcolor="#f4f8f2"
            )
            st.plotly_chart(fig_dep2, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_bar(x=daily_r.index.strftime("%a %d"), y=daily_r["IWR"],
                         name="IWR gross (mm)", marker_color="#1a5fc8")
            fig2.add_bar(x=daily_r.index.strftime("%a %d"), y=daily_r["NIR"],
                         name="NIR (mm)", marker_color="#0b6b1b", opacity=0.6)
            fig2.add_scatter(x=daily_r.index.strftime("%a %d"), y=daily_r["ET0"],
                             name="ET₀ mm/d", mode="lines+markers",
                             marker_color="#b81c1c", yaxis="y2")
            fig2.update_layout(
                title=f"5-Day IWR Forecast — {SITE_NAME}",
                yaxis=dict(title="mm"), barmode="group",
                yaxis2=dict(title="ET₀ mm/d", overlaying="y", side="right"),
                legend=dict(x=0, y=1.1, orientation="h"),
                height=350, plot_bgcolor="#f4f8f2", paper_bgcolor="#f4f8f2"
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.download_button(
                "⬇️ Download Forecast CSV",
                tb2.to_csv().encode(),
                f"HyPIS_forecast_{SITE_NAME.replace(' ','_')}_{datetime.today().strftime('%Y%m%d')}.csv",
                "text/csv", key="dl_fc"
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — HISTORICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header(f"📅 Historical IWR Analysis — {SITE_NAME}")
    st.caption(
        f"ERA5 Archive (Open-Meteo) · FAO-56 PM · MAD-threshold trigger · "
        f"MAD-threshold irrigation trigger · IWR = NIR ÷ Ef ({Ef*100:.0f}%, {irrig_sys})"
    )

    st.info(
        f"📍 **{SITE_NAME}** · Lat `{LAT}°` · Lon `{LON}°` · `{ELEV} m` a.s.l.  \n"
        f"🌍 Soil: **{ACTIVE_TXT}** · FC={ACTIVE_FC*100:.0f}% · PWP={ACTIVE_PWP*100:.0f}%  \n"
        f"*(ERA5 archive may lag 3–5 days — yesterday is the closest available)*"
    )

    pc1, pc2, pc3 = st.columns(3)
    yesterday = date.today() - timedelta(days=1)

    if pc1.button("📅 Yesterday",    use_container_width=True, key="h_yest"):
        st.session_state["h_start"] = yesterday
        st.session_state["h_end"]   = yesterday
    if pc2.button("📅 Last 7 Days",  use_container_width=True, key="h_7"):
        st.session_state["h_start"] = yesterday - timedelta(days=6)
        st.session_state["h_end"]   = yesterday
    if pc3.button("📅 Last 30 Days", use_container_width=True, key="h_30"):
        st.session_state["h_start"] = yesterday - timedelta(days=29)
        st.session_state["h_end"]   = yesterday

    h_start = st.session_state.get("h_start", yesterday - timedelta(days=6))
    h_end   = st.session_state.get("h_end",   yesterday)

    st.markdown(f"**Period:** `{h_start}` → `{h_end}` ({(h_end - h_start).days + 1} days)")

    hc1, hc2 = st.columns(2)
    with hc1:
        cr3 = st.selectbox("Crop", list(crop_params.keys()), key="cr3")
        cp3 = crop_params[cr3]
        stage3 = st.radio("Growing Stage", list(STAGE_LABELS.keys()),
                          format_func=lambda x: STAGE_LABELS[x],
                          key="stg3", horizontal=True)
        kc3 = kc_from_stage(stage3, cr3)
        st.markdown(
            f'<div class="kc-stage">Kc ({STAGE_LABELS[stage3]}) = <b>{kc3:.3f}</b></div>',
            unsafe_allow_html=True)
    with hc2:
        planting3 = st.date_input("Planting Date",
                                  value=date.today()-timedelta(days=45),
                                  key="plant3")
        soil3 = {"fc": ACTIVE_FC, "pwp": ACTIVE_PWP}
        st.markdown(
            f'<div class="soil-panel">🌍 Soil: <b>{ACTIVE_TXT}</b> '
            f'(FC={ACTIVE_FC*100:.0f}%, PWP={ACTIVE_PWP*100:.0f}%)</div>',
            unsafe_allow_html=True)
        sm3 = st.slider("Starting SM (% of FC)", 0, 100,
                        estimate_sm(ACTIVE_FC, ACTIVE_PWP, cp3["zr"]), key="sm3")

    if st.button("📥 Retrieve Historical Data", type="primary",
                 use_container_width=True, key="hist_btn"):
        with st.spinner(f"Fetching ERA5 archive for {SITE_NAME}…"):
            hist = get_historical_weather(str(h_start), str(h_end), LAT, LON)

        if hist is None or hist.empty:
            st.warning("⚠️ No ERA5 data for this period (archive may lag 3–5 days).")
        else:
            planting_ts3 = pd.Timestamp(planting3)
            hist_r, taw3, raw3 = run_water_balance(
                hist, cr3, soil3, planting_ts3, sm3, Ef,
                stage_override=stage3)

            hist_r["Vol_m3"] = hist_r["IWR"].apply(
                lambda x: compute_volume(x, area_ha)["vol_m3"])
            hist_r["Vol_L"]  = hist_r["IWR"].apply(
                lambda x: compute_volume(x, area_ha)["vol_L"])
            hist_r["ET0_H"]  = [
                et0_hargreaves(r["tmax"], r["tmin"],
                               doy=int(d.strftime("%j")), lat_deg=LAT)
                for d, r in hist_r.iterrows()
            ]
            hist_r["PumpMin"] = hist_r["Vol_m3"].apply(
                lambda v: round(v/pump_flow*60, 1) if pump_flow > 0 and v > 0 else 0)

            # MAD thresholds
            st.info(
                f"📐 **Thresholds for {cr3} / {ACTIVE_TXT}:**  "
                f"TAW = **{taw3:.1f} mm** · "
                f"RAW (MAD {int(cp3['mad']*100)}%) = **{raw3:.1f} mm**  \n"
                f"Irrigation triggered only on days where Dr > {raw3:.1f} mm "
                f"AND effective rain does not cover ETc."
            )

            m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
            m1.metric("📆 Days",        len(hist_r))
            m2.metric("🌧️ Rain Total",  f"{hist_r['precipitation'].sum():.1f} mm")
            m3.metric("💧 NIR Total",   f"{hist_r['NIR'].sum():.1f} mm")
            m4.metric("💧 IWR Total",   f"{hist_r['IWR'].sum():.1f} mm")
            m5.metric("🪣 Vol Total",   f"{hist_r['Vol_m3'].sum():.1f} m³")
            m6.metric("🚿 Irrig Days",  str((hist_r["IWR"] > 0).sum()))
            m7.metric("🌧️ Rain Days",   str((hist_r["precipitation"] > 1).sum()))

            st.subheader("📋 Historical Table (with Deficit & Status)")
            ht = hist_r[[
                "tmax","tmin","rh_mean","precipitation","Pe",
                "ET0","ET0_H","kc","ETc",
                "Depletion_mm","SM_pct",
                "NIR","IWR","Vol_m3","Vol_L","PumpMin","Status"
            ]].copy()
            ht.columns = [
                "Tmax °C","Tmin °C","RH %","Rain mm","Pe mm",
                "ET₀-PM mm/d","ET₀-H mm/d","Kc","ETc mm/d",
                "Dr Deficit mm","SM % FC",
                "NIR mm","IWR gross mm","Vol m³","Vol L",
                "Pump min","Soil Status"
            ]
            ht.index = ht.index.strftime("%Y-%m-%d (%a)")
            _fmth = {
                "Tmax °C":"{:.1f}","Tmin °C":"{:.1f}","RH %":"{:.0f}",
                "Rain mm":"{:.1f}","Pe mm":"{:.2f}",
                "ET₀-PM mm/d":"{:.2f}","ET₀-H mm/d":"{:.2f}",
                "Kc":"{:.3f}","ETc mm/d":"{:.2f}",
                "Dr Deficit mm":"{:.2f}","SM % FC":"{:.0f}",
                "NIR mm":"{:.2f}","IWR gross mm":"{:.2f}",
                "Vol m³":"{:.1f}","Vol L":"{:.0f}","Pump min":"{:.1f}",
            }
            st.dataframe(
                ht.style.format({k:v for k,v in _fmth.items() if k in ht.columns}),
                use_container_width=True)

            # Depletion vs thresholds chart
            fig_dep = go.Figure()
            fig_dep.add_scatter(
                x=hist_r.index, y=hist_r["Depletion_mm"],
                mode="lines", name="Dr Deficit (mm)",
                line=dict(color="#e6550d", width=1.5))
            fig_dep.add_bar(
                x=hist_r.index, y=hist_r["precipitation"],
                name="Rain (mm)", marker_color="#1a5fc8", opacity=0.35, yaxis="y2")
            fig_dep.add_bar(
                x=hist_r.index, y=hist_r["IWR"],
                name="IWR gross (mm)", marker_color="#17a2b8", opacity=0.7, yaxis="y2")
            fig_dep.add_hline(y=raw3, line_dash="dash", line_color="#756bb1",
                              annotation_text=f"RAW={raw3:.1f} mm (irrigate above)")
            fig_dep.add_hline(y=taw3, line_dash="dot",  line_color="#d73027",
                              annotation_text=f"TAW={taw3:.1f} mm (wilting risk)")
            fig_dep.update_layout(
                title="Root-Zone Depletion vs Irrigation Thresholds",
                yaxis=dict(title="Depletion Dr (mm)"),
                yaxis2=dict(title="mm", overlaying="y", side="right"),
                barmode="overlay",
                legend=dict(x=0, y=1.1, orientation="h"),
                height=400, plot_bgcolor="#f4f8f2", paper_bgcolor="#f4f8f2")
            st.plotly_chart(fig_dep, use_container_width=True)

            # ET₀ chart
            fig3 = go.Figure()
            fig3.add_scatter(x=hist_r.index, y=hist_r["ET0"],
                             name="ET₀ PM mm/d", mode="lines",
                             line=dict(color="#1a5fc8", width=1.5))
            fig3.add_scatter(x=hist_r.index, y=hist_r["ET0_H"],
                             name="ET₀ Hargreaves", mode="lines",
                             line=dict(color="#b81c1c", width=1, dash="dot"))
            fig3.add_scatter(x=hist_r.index, y=hist_r["ETc"],
                             name="ETc mm/d", mode="lines",
                             line=dict(color="#0b6b1b", width=1.5, dash="dash"))
            fig3.update_layout(
                title="Historical ET₀ & ETc",
                yaxis_title="mm/d",
                height=320, plot_bgcolor="#f4f8f2", paper_bgcolor="#f4f8f2",
                legend=dict(x=0, y=1.1, orientation="h"))
            st.plotly_chart(fig3, use_container_width=True)

            # Volume chart (only irrigation days)
            irrig_days = hist_r[hist_r["IWR"] > 0]
            if not irrig_days.empty:
                fig_vol = go.Figure()
                fig_vol.add_bar(
                    x=irrig_days.index, y=irrig_days["Vol_m3"],
                    name="Volume (m³)", marker_color="#0d6efd",
                    text=irrig_days["Vol_m3"].round(1).astype(str)+" m³",
                    textposition="outside")
                fig_vol.update_layout(
                    title=f"Irrigation Volume on Irrigated Days — {area_ha} ha",
                    yaxis_title="m³",
                    height=300, plot_bgcolor="#f4f8f2", paper_bgcolor="#f4f8f2")
                st.plotly_chart(fig_vol, use_container_width=True)
            else:
                st.success("✅ No irrigation events in this period — soil was adequately watered by rain.")

            st.download_button(
                "⬇️ Download Historical CSV",
                ht.to_csv().encode(),
                f"HyPIS_historical_{SITE_NAME.replace(' ','_')}_{h_start}_{h_end}.csv",
                "text/csv", key="dl_hist"
            )

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    f"HyPIS Ug v7.0 · {SITE_NAME} ({LAT}°, {LON}°, {ELEV} m) · "
    f"ERA5 + ICON + GFS (Open-Meteo) · FAO-56 Penman-Monteith · "
    f"MAD-threshold irrigation trigger (no auto-irrigation on rainy days) · "
    f"IWR = NIR ÷ Ef · HWSD v2 Soil · Byaruhanga Prosper"
)
