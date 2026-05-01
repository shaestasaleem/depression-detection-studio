import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="NeuroVoice · Depression Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;1,500&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
  --bg:#060910; --bg2:#0D1117; --bg3:#131C2E; --bg4:#1A2540;
  --teal:#00D4AA; --teal2:#00B894; --teal3:#00876E;
  --coral:#FF6B8A; --coral2:#E84393;
  --amber:#FFB830; --green:#00E676; --green2:#00C853;
  --white:#F0F4FF; --white2:#B8C5D6; --muted:#607080;
  --border:rgba(0,212,170,0.08); --border2:rgba(0,212,170,0.18);
}

.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; color: var(--white); }

[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border2) !important; }
[data-testid="stSidebar"] * { color: var(--white2) !important; font-family: 'Outfit', sans-serif !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color: var(--white) !important; }

.nv-header { background: linear-gradient(180deg,rgba(0,212,170,0.04),transparent); border-bottom: 1px solid var(--border2); padding: 1.5rem 0 1.2rem; margin-bottom: 2rem; display: flex; align-items: center; justify-content: space-between; }
.nv-logo { font-family:'Playfair Display',serif; font-size:32px; font-weight:700; color:var(--white); letter-spacing:-0.02em; }
.nv-logo em { font-style:italic; color:var(--teal); }
.nv-sub { font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--muted); letter-spacing:0.14em; text-transform:uppercase; margin-top:4px; }
.nv-badge { background:rgba(0,212,170,0.08); border:1px solid var(--border2); border-radius:30px; padding:6px 16px; font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--teal); }

.stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:2rem; }
.stat-card { background:var(--bg2); border-radius:14px; padding:1.2rem 1rem; text-align:center; position:relative; overflow:hidden; border:1px solid var(--border); }
.stat-card::before { content:''; position:absolute; top:0;left:0;right:0; height:2px; }
.stat-card.c1::before { background:linear-gradient(90deg,var(--teal),var(--teal2)); }
.stat-card.c2::before { background:linear-gradient(90deg,var(--green),var(--green2)); }
.stat-card.c3::before { background:linear-gradient(90deg,var(--amber),#FF8C00); }
.stat-card.c4::before { background:linear-gradient(90deg,var(--coral),var(--coral2)); }
.stat-val { font-family:'JetBrains Mono',monospace; font-size:26px; font-weight:600; margin-bottom:4px; }
.stat-lbl { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; font-weight:500; }

.panel-tag { font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--teal); text-transform:uppercase; letter-spacing:0.14em; margin-bottom:0.5rem; display:flex; align-items:center; gap:6px; }
.panel-tag::before { content:''; width:6px;height:6px; border-radius:50%; background:var(--teal); animation:blink 2s ease infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
.panel-title { font-family:'Playfair Display',serif; font-size:22px; font-weight:500; color:var(--white); margin-bottom:0.5rem; }
.panel-sub { font-size:13px; color:var(--muted); margin-bottom:1rem; line-height:1.6; }

div[data-testid="stFileUploader"] { background:var(--bg3) !important; border:1.5px dashed var(--border2) !important; border-radius:12px !important; }
div[data-testid="stFileUploader"] * { color:var(--white2) !important; }
div[data-testid="stFileUploader"] label { display:none !important; }

.stButton > button { background:linear-gradient(135deg,var(--teal2),var(--teal3)) !important; color:#000 !important; border:none !important; border-radius:12px !important; font-family:'Outfit',sans-serif !important; font-weight:700 !important; font-size:15px !important; padding:0.75rem 2rem !important; width:100% !important; letter-spacing:0.06em !important; text-transform:uppercase !important; box-shadow:0 4px 20px rgba(0,212,170,0.25) !important; }
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 30px rgba(0,212,170,0.35) !important; }
.stButton > button:disabled { background:var(--bg4) !important; color:var(--muted) !important; box-shadow:none !important; }

.verdict-healthy { background:linear-gradient(135deg,rgba(0,230,118,0.08),rgba(0,212,170,0.05)); border:2px solid rgba(0,230,118,0.4); border-radius:18px; padding:2rem; text-align:center; position:relative; overflow:hidden; margin-bottom:1.25rem; }
.verdict-healthy::before { content:''; position:absolute; top:0;left:0;right:0;height:3px; background:linear-gradient(90deg,transparent,var(--green),transparent); }
.verdict-depressed { background:linear-gradient(135deg,rgba(255,107,138,0.08),rgba(232,67,147,0.04)); border:2px solid rgba(255,107,138,0.4); border-radius:18px; padding:2rem; text-align:center; position:relative; overflow:hidden; margin-bottom:1.25rem; }
.verdict-depressed::before { content:''; position:absolute; top:0;left:0;right:0;height:3px; background:linear-gradient(90deg,transparent,var(--coral),transparent); }

.v-icon { font-size:56px; margin-bottom:0.5rem; }
.v-title-h { font-family:'Playfair Display',serif; font-size:32px; font-weight:700; color:var(--green); margin-bottom:0.3rem; }
.v-title-d { font-family:'Playfair Display',serif; font-size:32px; font-weight:700; color:var(--coral); margin-bottom:0.3rem; }
.v-sub-h { font-size:13px; color:rgba(0,230,118,0.65); margin-bottom:1.2rem; }
.v-sub-d { font-size:13px; color:rgba(255,107,138,0.65); margin-bottom:1.2rem; }
.v-conf-h { font-family:'JetBrains Mono',monospace; font-size:72px; font-weight:600; color:var(--green); line-height:1; }
.v-conf-d { font-family:'JetBrains Mono',monospace; font-size:72px; font-weight:600; color:var(--coral); line-height:1; }
.v-lbl { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-top:8px; }

.compare-wrap { background:var(--bg3); border:1px solid var(--border); border-radius:14px; padding:1.25rem; margin-bottom:1rem; }
.compare-title { font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1.2rem; }
.cbar-row { display:flex; align-items:center; gap:12px; margin-bottom:14px; }
.cbar-label { font-size:13px; font-weight:600; width:145px; flex-shrink:0; }
.cbar-track { flex:1; height:12px; background:var(--bg4); border-radius:6px; overflow:hidden; }
.cbar-fill-h { height:100%; border-radius:6px; background:linear-gradient(90deg,var(--green2),var(--green)); transition:width 1s ease; }
.cbar-fill-d { height:100%; border-radius:6px; background:linear-gradient(90deg,var(--coral2),var(--coral)); transition:width 1s ease; }
.cbar-pct { font-family:'JetBrains Mono',monospace; font-size:15px; font-weight:700; width:52px; text-align:right; flex-shrink:0; }
.winner-tag { font-size:9px; background:rgba(0,230,118,0.15); color:var(--green); padding:2px 7px; border-radius:4px; margin-left:6px; font-family:monospace; }
.winner-tag-d { font-size:9px; background:rgba(255,107,138,0.15); color:var(--coral); padding:2px 7px; border-radius:4px; margin-left:6px; font-family:monospace; }

.waiting-state { background:var(--bg3); border:1.5px dashed var(--border2); border-radius:16px; padding:3.5rem 2rem; text-align:center; }
.waiting-icon { font-size:52px; margin-bottom:1rem; opacity:0.35; }
.waiting-txt { font-size:15px; color:var(--muted); line-height:1.7; }

.stProgress > div > div > div { background:linear-gradient(90deg,var(--teal2),var(--green)) !important; border-radius:4px !important; }
.stProgress > div > div { background:var(--bg4) !important; border-radius:4px !important; }

.stTabs [data-baseweb="tab-list"] { background:var(--bg2) !important; border-radius:10px; gap:2px; padding:4px; border:1px solid var(--border); }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:var(--muted) !important; font-family:'Outfit',sans-serif !important; font-weight:500 !important; border-radius:8px !important; }
.stTabs [aria-selected="true"] { background:var(--bg3) !important; color:var(--teal) !important; }

.file-info { background:var(--bg3); border:1px solid var(--border); border-radius:10px; padding:10px 14px; display:flex; align-items:center; gap:10px; font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--white2); margin:10px 0; }

.disc { background:rgba(255,184,48,0.05); border:1px solid rgba(255,184,48,0.18); border-radius:10px; padding:10px 16px; font-size:11px; color:#8B6914; margin-top:1.5rem; line-height:1.6; }

[data-testid="stMetric"] { background:var(--bg3); border-radius:10px; padding:10px 14px; border:1px solid var(--border); }
[data-testid="stMetric"] label { color:var(--muted) !important; font-size:11px !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color:var(--white) !important; font-family:'JetBrains Mono',monospace !important; }

.streamlit-expanderHeader { background:var(--bg3) !important; border-radius:10px !important; color:var(--white2) !important; border:1px solid var(--border) !important; }
div[data-testid="stExpander"] { border:none !important; background:transparent !important; }
audio { filter:invert(0.8) hue-rotate(140deg); border-radius:8px; width:100%; margin-top:8px; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ─── LOAD MODELS ───
@st.cache_resource
def load_models():
    base   = os.path.dirname(__file__)
    model  = joblib.load(os.path.join(base, "best_model.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler.pkl"))
    le     = joblib.load(os.path.join(base, "label_encoder.pkl"))
    return model, scaler, le

model, scaler, le = load_models()


# ─── HELPERS ───
def extract_features(path, n_mfcc=40, sr=22050):
    y, sr_  = librosa.load(path, sr=sr)
    mfcc    = librosa.feature.mfcc(y=y, sr=sr_, n_mfcc=n_mfcc)
    feats   = np.mean(mfcc, axis=1)
    return feats, y, sr_, mfcc

def run_prediction(feats):
    scaled  = scaler.transform(feats.reshape(1, -1))
    pred    = model.predict(scaled)[0]
    label   = le.inverse_transform([pred])[0]
    try:
        proba  = model.predict_proba(scaled)[0]
        conf_h = proba[list(le.classes_).index('Healthy')]
        conf_d = proba[list(le.classes_).index('Sad_Depressed')]
    except:
        score  = model.decision_function(scaled)[0]
        conf_d = 1 / (1 + np.exp(-score))
        conf_h = 1 - conf_d
    return label, conf_h, conf_d

def dark_fig(w=9, h=2.4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')
    ax.tick_params(colors='#607080', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#1A2540')
    return fig, ax

def plt_waveform(y, sr, color):
    fig, ax = dark_fig(9, 2.2)
    t = np.linspace(0, len(y)/sr, len(y))
    ax.plot(t, y, color=color, lw=0.6, alpha=0.9)
    ax.fill_between(t, y, alpha=0.12, color=color)
    ax.set_xlabel("Time (s)", color='#607080', fontsize=9)
    ax.set_ylabel("Amplitude", color='#607080', fontsize=9)
    ax.set_title("Audio Waveform", color='#B8C5D6', fontsize=11, pad=10, fontweight='bold')
    plt.tight_layout(); return fig

def plt_mfcc_heatmap(mfcc):
    fig, ax = dark_fig(9, 3.2)
    img  = ax.imshow(mfcc, aspect='auto', origin='lower', cmap='plasma')
    cbar = plt.colorbar(img, ax=ax)
    cbar.ax.tick_params(colors='#607080', labelsize=8)
    ax.set_xlabel("Time Frames", color='#607080', fontsize=9)
    ax.set_ylabel("MFCC Index", color='#607080', fontsize=9)
    ax.set_title("MFCC Heatmap — 40 Coefficients over Time", color='#B8C5D6', fontsize=11, pad=10, fontweight='bold')
    plt.tight_layout(); return fig

def plt_mfcc_bars(feats, color):
    fig, ax = dark_fig(9, 2.8)
    clrs = [color if v >= 0 else '#607080' for v in feats]
    ax.bar(range(1, 41), feats, color=clrs, alpha=0.85, width=0.72)
    ax.axhline(0, color='#1A2540', lw=1)
    ax.set_xlabel("MFCC Coefficient Index", color='#607080', fontsize=9)
    ax.set_ylabel("Mean Value", color='#607080', fontsize=9)
    ax.set_title("MFCC Feature Vector (mean over time)", color='#B8C5D6', fontsize=11, pad=10, fontweight='bold')
    plt.tight_layout(); return fig

def plt_spectrogram(y, sr):
    fig, ax = dark_fig(9, 2.8)
    D   = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.tick_params(colors='#607080', labelsize=8)
    ax.set_title("Mel Spectrogram", color='#B8C5D6', fontsize=11, pad=10, fontweight='bold')
    ax.set_xlabel("Time (s)", color='#607080', fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color='#607080', fontsize=9)
    plt.tight_layout(); return fig


# ══════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
      <div style='font-family:"Playfair Display",serif;font-size:22px;font-weight:700;color:#F0F4FF'>Neuro<span style='color:#00D4AA'>Voice</span></div>
      <div style='font-family:"JetBrains Mono",monospace;font-size:9px;color:#607080;letter-spacing:0.12em;text-transform:uppercase;margin-top:3px'>SP Final Project · 2026</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("**👩‍🎓 Student**")
    st.markdown("""<div style='background:#131C2E;border:1px solid rgba(0,212,170,0.12);border-radius:10px;padding:12px;'>
      <div style='color:#F0F4FF;font-weight:600;font-size:14px'>Shaesta Saleem</div>
      <div style='font-family:"JetBrains Mono",monospace;font-size:11px;color:#00D4AA;margin-top:3px'>DSAI231103043</div>
      <div style='font-size:12px;color:#607080;margin-top:4px'>Speech Processing · 2026</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**🤖 Model Info**")
    st.markdown("""<table style='width:100%;font-size:12px;border-collapse:collapse'>
      <tr><td style='color:#607080;padding:5px 0'>Model</td><td style='color:#F0F4FF;text-align:right;font-family:monospace'>SVM (RBF)</td></tr>
      <tr><td style='color:#607080;padding:5px 0'>Features</td><td style='color:#F0F4FF;text-align:right;font-family:monospace'>40-dim MFCC</td></tr>
      <tr><td style='color:#607080;padding:5px 0'>Dataset</td><td style='color:#F0F4FF;text-align:right;font-family:monospace'>RAVDESS</td></tr>
      <tr><td style='color:#607080;padding:5px 0'>Samples</td><td style='color:#F0F4FF;text-align:right;font-family:monospace'>672 balanced</td></tr>
      <tr><td style='color:#607080;padding:5px 0'>Split</td><td style='color:#F0F4FF;text-align:right;font-family:monospace'>80 / 20</td></tr>
    </table>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**🎯 Classes**")
    st.markdown("""
    <div style='background:rgba(0,230,118,0.07);border:1px solid rgba(0,230,118,0.25);border-radius:8px;padding:10px 12px;margin-bottom:8px'>
      <div style='color:#00E676;font-weight:700;font-size:14px'>✓ Healthy</div>
      <div style='color:#607080;font-size:11px;margin-top:2px'>Neutral + Calm speech</div>
    </div>
    <div style='background:rgba(255,107,138,0.07);border:1px solid rgba(255,107,138,0.25);border-radius:8px;padding:10px 12px'>
      <div style='color:#FF6B8A;font-weight:700;font-size:14px'>⚠ Sad / Depressed</div>
      <div style='color:#607080;font-size:11px;margin-top:2px'>Depressive speech patterns</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**📋 11-Phase Pipeline**")
    for n,t,c in [("01","Data Load","#00D4AA"),("02","Labeling","#00D4AA"),("03","EDA","#00D4AA"),
                  ("04","Imbalance","#FFB830"),("05","Augment","#FFB830"),("06","MFCC","#00E676"),
                  ("07","Split","#00E676"),("08","Train","#4FC3F7"),("09","Evaluate","#4FC3F7"),
                  ("10","Serialize","#FF6B8A"),("11","Deploy","#FF6B8A")]:
        st.markdown(f"<div style='display:flex;align-items:center;gap:8px;padding:3px 0'><span style='font-family:monospace;font-size:11px;color:{c};font-weight:600'>{n}</span><span style='font-size:12px;color:#607080'>{t}</span></div>", unsafe_allow_html=True)


# ══════════════════════════════════════
#  HEADER
# ══════════════════════════════════════
st.markdown("""
<div class="nv-header">
  <div>
    <div class="nv-logo">Neuro<em>Voice</em></div>
    <div class="nv-sub">Acoustic Biomarker Analysis · Depression Detection · Clinical AI Research</div>
  </div>
  <div class="nv-badge">● SVM · RBF · Ready</div>
</div>""", unsafe_allow_html=True)

# STAT CARDS
st.markdown("""
<div class="stat-grid">
  <div class="stat-card c1"><div class="stat-val" style="color:#00D4AA">81.48%</div><div class="stat-lbl">Test Accuracy</div></div>
  <div class="stat-card c2"><div class="stat-val" style="color:#00E676">80.98%</div><div class="stat-lbl">CV F1 Mean</div></div>
  <div class="stat-card c3"><div class="stat-val" style="color:#FFB830">672</div><div class="stat-lbl">Training Samples</div></div>
  <div class="stat-card c4"><div class="stat-val" style="color:#FF6B8A">40</div><div class="stat-lbl">MFCC Features</div></div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════
#  MAIN COLUMNS
# ══════════════════════════════════════
col_L, col_R = st.columns([1, 1], gap="large")

with col_L:
    st.markdown('<div class="panel-tag">Audio Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Upload Audio File</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-sub">Upload any .wav file — the real trained SVM model extracts MFCC features and gives you an actual prediction.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("audio", type=["wav","mp3","ogg","flac"], label_visibility="collapsed")

    if uploaded:
        st.markdown(f"""<div class="file-info"><span>🎵</span><span>{uploaded.name}</span><span style='color:#607080;margin-left:auto'>{uploaded.size//1024} KB</span></div>""", unsafe_allow_html=True)
        st.audio(uploaded)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("🔍  Analyze Audio Now", disabled=(uploaded is None))


with col_R:
    st.markdown('<div class="panel-tag">Prediction Result</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""<div class="waiting-state">
          <div class="waiting-icon">🎙️</div>
          <div class="waiting-txt">Upload a .wav file on the left<br><br>The real SVM model will show you<br>the prediction here instantly</div>
        </div>""", unsafe_allow_html=True)

    elif analyze:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        prog   = st.progress(0)
        status = st.empty()

        for pct, msg in [(20,"🔊 Loading audio signal..."),
                         (45,"📊 Extracting 40-dim MFCC features..."),
                         (70,"⚖️  Z-score normalization..."),
                         (90,"🤖 SVM (RBF) inference..."),
                         (100,"✅ Complete!")]:
            status.markdown(f"<small style='font-family:monospace;color:#00D4AA'>{msg}</small>", unsafe_allow_html=True)
            prog.progress(pct)
            time.sleep(0.4)

        prog.empty(); status.empty()

        try:
            feats, y_audio, sr_audio, mfcc_full = extract_features(tmp_path)
            label, conf_h, conf_d               = run_prediction(feats)
            os.unlink(tmp_path)

            is_h       = (label == 'Healthy')
            conf_main  = conf_h if is_h else conf_d
            color_main = "#00E676" if is_h else "#FF6B8A"
            h_pct      = int(conf_h * 100)
            d_pct      = int(conf_d * 100)

            # BIG VERDICT
            if is_h:
                st.markdown(f"""<div class="verdict-healthy">
                  <div class="v-icon">✅</div>
                  <div class="v-title-h">Healthy Speech</div>
                  <div class="v-sub-h">Acoustic profile consistent with healthy baseline</div>
                  <div class="v-conf-h">{conf_main*100:.1f}%</div>
                  <div class="v-lbl">Model Confidence</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="verdict-depressed">
                  <div class="v-icon">⚠️</div>
                  <div class="v-title-d">Depressive Markers</div>
                  <div class="v-sub-d">Acoustic features indicate depressive speech patterns</div>
                  <div class="v-conf-d">{conf_main*100:.1f}%</div>
                  <div class="v-lbl">Model Confidence</div>
                </div>""", unsafe_allow_html=True)

            # COMPARISON BARS
            st.markdown(f"""
            <div class="compare-wrap">
              <div class="compare-title">📊 Class Probability Breakdown — Which one won?</div>
              <div class="cbar-row">
                <div class="cbar-label" style="color:#00E676">✓ Healthy{'<span class="winner-tag">▲ WINNER</span>' if is_h else ''}</div>
                <div class="cbar-track"><div class="cbar-fill-h" style="width:{h_pct}%"></div></div>
                <div class="cbar-pct" style="color:#00E676">{h_pct}%</div>
              </div>
              <div class="cbar-row">
                <div class="cbar-label" style="color:#FF6B8A">⚠ Sad/Depressed{'<span class="winner-tag-d">▲ WINNER</span>' if not is_h else ''}</div>
                <div class="cbar-track"><div class="cbar-fill-d" style="width:{d_pct}%"></div></div>
                <div class="cbar-pct" style="color:#FF6B8A">{d_pct}%</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # META CHIPS
            duration = len(y_audio)/sr_audio
            st.markdown(f"""<div style='display:flex;gap:8px;flex-wrap:wrap'>
              <span style='background:#131C2E;border:1px solid rgba(0,212,170,0.1);border-radius:6px;padding:4px 10px;font-family:monospace;font-size:11px;color:#607080'>⏱ {duration:.2f}s</span>
              <span style='background:#131C2E;border:1px solid rgba(0,212,170,0.1);border-radius:6px;padding:4px 10px;font-family:monospace;font-size:11px;color:#607080'>📡 {sr_audio} Hz</span>
              <span style='background:#131C2E;border:1px solid rgba(0,212,170,0.1);border-radius:6px;padding:4px 10px;font-family:monospace;font-size:11px;color:#607080'>🔢 {len(y_audio):,} samples</span>
            </div>""", unsafe_allow_html=True)

            st.session_state['res'] = {
                'y':y_audio,'sr':sr_audio,'mfcc':mfcc_full,
                'feats':feats,'label':label,
                'conf_h':conf_h,'conf_d':conf_d,'color':color_main
            }

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            if os.path.exists(tmp_path): os.unlink(tmp_path)

    elif 'res' in st.session_state:
        r    = st.session_state['res']
        is_h = r['label'] == 'Healthy'
        cm   = r['conf_h'] if is_h else r['conf_d']
        h_pct = int(r['conf_h']*100); d_pct = int(r['conf_d']*100)
        if is_h:
            st.markdown(f'<div class="verdict-healthy"><div class="v-icon">✅</div><div class="v-title-h">Healthy Speech</div><div class="v-conf-h">{cm*100:.1f}%</div><div class="v-lbl">Model Confidence</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-depressed"><div class="v-icon">⚠️</div><div class="v-title-d">Depressive Markers</div><div class="v-conf-d">{cm*100:.1f}%</div><div class="v-lbl">Model Confidence</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="compare-wrap"><div class="compare-title">Class Probability Breakdown</div><div class="cbar-row"><div class="cbar-label" style="color:#00E676">✓ Healthy{"<span class=winner-tag>▲ WINNER</span>" if is_h else ""}</div><div class="cbar-track"><div class="cbar-fill-h" style="width:{h_pct}%"></div></div><div class="cbar-pct" style="color:#00E676">{h_pct}%</div></div><div class="cbar-row"><div class="cbar-label" style="color:#FF6B8A">⚠ Sad/Depressed{"<span class=winner-tag-d>▲ WINNER</span>" if not is_h else ""}</div><div class="cbar-track"><div class="cbar-fill-d" style="width:{d_pct}%"></div></div><div class="cbar-pct" style="color:#FF6B8A">{d_pct}%</div></div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════
#  VISUALIZATIONS
# ══════════════════════════════════════
if 'res' in st.session_state:
    r = st.session_state['res']
    st.divider()
    st.markdown('<div class="panel-tag">Acoustic Signal Analysis</div>', unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["📈 Waveform","🔥 MFCC Heatmap","📊 MFCC Bars","🎨 Spectrogram"])
    with t1: st.pyplot(plt_waveform(r['y'],r['sr'],r['color']))
    with t2:
        st.pyplot(plt_mfcc_heatmap(r['mfcc']))
        st.caption("Each row = one MFCC coefficient over time. Brighter = higher energy.")
    with t3:
        st.pyplot(plt_mfcc_bars(r['feats'],r['color']))
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("c1 (Energy)",f"{r['feats'][0]:.2f}")
        mc2.metric("c2",f"{r['feats'][1]:.2f}")
        mc3.metric("Mean",f"{np.mean(r['feats']):.2f}")
        mc4.metric("Std Dev",f"{np.std(r['feats']):.2f}")
    with t4: st.pyplot(plt_spectrogram(r['y'],r['sr']))

    with st.expander("🔢 All 40 MFCC Coefficient Values"):
        cols = st.columns(8)
        for i,v in enumerate(r['feats']):
            c = "#00E676" if v >= 0 else "#FF6B8A"
            with cols[i%8]:
                st.markdown(f"<div style='text-align:center;padding:4px 0'><div style='font-family:monospace;font-size:9px;color:#607080'>c{i+1}</div><div style='font-family:monospace;font-size:13px;font-weight:700;color:{c}'>{v:.2f}</div></div>", unsafe_allow_html=True)

st.markdown("""<div class="disc">
  ⚠️ <strong>Research Prototype Only.</strong> Not a clinical diagnostic instrument. Do not use for medical decision-making.
  Consult a licensed mental health professional. | RAVDESS corpus · SVM (RBF) · 81.48% accuracy · DSAI231103043 · Shaesta Saleem · 2026
</div>""", unsafe_allow_html=True)
