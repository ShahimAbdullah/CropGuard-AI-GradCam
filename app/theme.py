"""
CropGuard AI — Shared Theme
Inject the premium emerald + gold glass aurora CSS into any page.
Usage:
    from theme import inject_css, hero, footer
    inject_css()
"""

import streamlit as st


# ─── CSS ─────────────────────────────────────────────────────────────────────
_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Cormorant+Garamond:wght@400;500;600;700&display=swap');

  :root {
    --bg:      #06231a;
    --bg-2:    #04170f;
    --emerald: #0d7a5f;
    --em2:     #14a37a;
    --gold:    #c9a84c;
    --gold2:   #e8c971;
    --cream:   #f5f0e0;
    --muted:   #9ec0b0;
    --line:    rgba(201,168,76,0.18);
    --glass:   rgba(245,240,224,0.04);
    --glass2:  rgba(245,240,224,0.07);
  }

  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--cream) !important;
    font-family: 'Inter', sans-serif !important;
    overflow-x: hidden;
  }
  [data-testid="stAppViewContainer"]::before {
    content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
    background:
      radial-gradient(700px 500px at 10% 0%,  rgba(20,163,122,0.22), transparent 60%),
      radial-gradient(800px 600px at 95% 10%, rgba(201,168,76,0.18), transparent 60%),
      radial-gradient(700px 700px at 50% 110%, rgba(13,122,95,0.20), transparent 60%);
    animation: aurora 22s ease-in-out infinite alternate;
    filter: blur(20px);
  }
  @keyframes aurora {
    0%   { transform: translate3d(0,0,0)       scale(1);    }
    50%  { transform: translate3d(-30px,20px,0) scale(1.05); }
    100% { transform: translate3d(20px,-15px,0) scale(1);    }
  }
  @keyframes shimmer { 0%{background-position:200% 0;} 100%{background-position:-100% 0;} }
  @keyframes fadeIn  { from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:none;} }
  @keyframes slideUp { from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;} }
  @keyframes growBar { from{width:0;} }
  @keyframes pulse   { 0%,100%{opacity:1;} 50%{opacity:.4;} }

  [data-testid="stHeader"]  { background: transparent !important; }
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stToolbar"] { display: none; }
  .block-container { padding:1.6rem 2.4rem 3rem 2.4rem !important; max-width:1380px !important; position:relative; z-index:1; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar       { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-2); }
  ::-webkit-scrollbar-thumb { background: var(--emerald); border-radius: 3px; }

  /* ── Glass base ── */
  .glass {
    background: var(--glass);
    border: 1px solid var(--line);
    border-radius: 20px;
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    box-shadow: 0 8px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
  }

  /* ── Hero ── */
  .hero { position:relative; padding:1.6rem 1.9rem; margin-bottom:1.5rem; overflow:hidden; }
  .hero::before {
    content:''; position:absolute; inset:-1px; border-radius:20px; padding:1px;
    background: linear-gradient(120deg, transparent 20%, rgba(201,168,76,0.55) 50%, transparent 80%);
    background-size:300% 100%;
    -webkit-mask:linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite:xor; mask-composite:exclude;
    animation: shimmer 6s linear infinite;
  }
  .hero-row    { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
  .hero-left   { display:flex; align-items:center; gap:1.1rem; }
  .hero-badge  {
    width:58px; height:58px; display:grid; place-items:center; font-size:1.9rem;
    border-radius:16px;
    background: linear-gradient(135deg, rgba(20,163,122,0.25), rgba(201,168,76,0.18));
    border:1px solid rgba(201,168,76,0.35);
    box-shadow: 0 0 30px rgba(20,163,122,0.25), inset 0 1px 0 rgba(255,255,255,0.08);
  }
  .hero-title {
    font-family:'Cormorant Garamond', serif;
    font-size:2.4rem; font-weight:600; line-height:1.05; letter-spacing:-0.01em;
    background: linear-gradient(90deg, var(--cream) 0%, var(--gold2) 70%);
    -webkit-background-clip:text; background-clip:text; color:transparent;
  }
  .hero-subtitle { color:var(--muted); font-size:0.9rem; margin-top:0.3rem; letter-spacing:0.01em; }
  .hero-stats    { display:flex; gap:0.5rem; flex-wrap:wrap; }

  /* ── Stat chips ── */
  .stat-chip {
    background:var(--glass2); border:1px solid var(--line); border-radius:999px;
    padding:0.4rem 0.85rem; color:var(--cream); font-size:0.74rem; font-weight:500;
    letter-spacing:0.04em; display:inline-flex; align-items:center; gap:0.4rem;
    backdrop-filter:blur(10px);
  }
  .stat-chip .dot { width:6px; height:6px; border-radius:50%; background:var(--em2); box-shadow:0 0 10px var(--em2); animation:pulse 2.4s ease-in-out infinite; }
  .stat-chip.gold { border-color:rgba(201,168,76,0.35); color:var(--gold2); }

  /* ── Byline ── */
  .byline {
    display:inline-flex; align-items:center; gap:0.5rem;
    margin-top:0.7rem; padding:0.32rem 0.85rem;
    background:var(--glass); border:1px solid var(--line); border-radius:999px;
    font-size:0.72rem; color:var(--muted); letter-spacing:0.06em;
    backdrop-filter:blur(10px);
  }
  .byline b { color:var(--gold2); font-weight:600; }
  .byline .sep { color:var(--gold); opacity:0.6; margin:0 0.2rem; }

  /* ── Nav cards ── */
  .nav-card {
    padding:1.4rem 1.6rem; cursor:pointer; transition:transform .2s, box-shadow .2s;
    animation:fadeIn 0.5s ease;
  }
  .nav-card:hover { transform:translateY(-3px); box-shadow:0 16px 50px rgba(0,0,0,0.4); }
  .nav-card .icon { font-size:1.8rem; margin-bottom:0.65rem; }
  .nav-card .title {
    font-family:'Cormorant Garamond', serif;
    font-size:1.25rem; font-weight:600; color:var(--gold2); margin-bottom:0.4rem;
  }
  .nav-card .desc  { color:var(--muted); font-size:0.85rem; line-height:1.55; }
  .nav-card .arrow { color:var(--gold); font-size:0.8rem; margin-top:0.7rem; letter-spacing:0.06em; }

  /* ── Section card / info panel ── */
  .section-card { padding:1.4rem 1.6rem; margin-bottom:1rem; animation:fadeIn 0.5s ease; }
  .section-title {
    color:var(--gold2); font-size:0.72rem; font-weight:600;
    letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.75rem;
  }

  /* ── Metric overrides ── */
  [data-testid="stMetric"] {
    background:var(--glass) !important; border:1px solid var(--line) !important;
    border-radius:14px !important; padding:1rem 1.2rem !important;
    backdrop-filter:blur(14px) !important;
  }
  [data-testid="stMetricLabel"]  { color:var(--muted)   !important; font-size:0.8rem  !important; letter-spacing:0.04em !important; }
  [data-testid="stMetricValue"]  {
    font-family:'Cormorant Garamond', serif !important;
    color:var(--cream) !important; font-size:1.8rem !important;
  }
  [data-testid="stMetricDelta"]  { font-size:0.75rem !important; }
  [data-testid="stMetricDeltaIcon"]{ display:none !important; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--bg-2) !important;
    border-right: 1px solid var(--line) !important;
  }
  [data-testid="stSidebar"] * { color: var(--cream) !important; }
  [data-testid="stSidebar"] .stCheckbox label,
  [data-testid="stSidebar"] .stSelectbox label { color: var(--muted) !important; font-size:0.85rem !important; }
  [data-testid="stSidebarHeader"] { display:none !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap:0.4rem; background:transparent; border-bottom:1px solid var(--line); margin-bottom:0.8rem;
  }
  .stTabs [data-baseweb="tab"] {
    background:transparent !important; color:var(--muted) !important;
    border-radius:10px 10px 0 0 !important; padding:0.5rem 0.9rem !important;
    font-size:0.85rem !important; font-weight:500 !important;
  }
  .stTabs [aria-selected="true"] { color:var(--gold2) !important; border-bottom:2px solid var(--gold) !important; }

  /* ── Buttons ── */
  .stButton > button, [data-testid="stDownloadButton"] > button {
    background:linear-gradient(135deg, var(--gold), #a88a36) !important;
    color:#1a1208 !important; border:none !important; border-radius:10px !important;
    font-weight:700 !important; letter-spacing:0.02em !important;
    transition:all .2s ease !important;
  }
  .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
    transform:translateY(-1px) !important; box-shadow:0 6px 20px rgba(201,168,76,0.35) !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"]            { background:transparent !important; border:none !important; padding:0 !important; }
  [data-testid="stFileUploaderDropzone"]    {
    background:var(--glass) !important; border:1.5px dashed rgba(201,168,76,0.35) !important;
    border-radius:14px !important; transition:all .3s ease; backdrop-filter:blur(10px);
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color:var(--gold) !important; background:rgba(201,168,76,0.06) !important;
    box-shadow:0 0 0 4px rgba(201,168,76,0.08), 0 0 30px rgba(201,168,76,0.12);
  }
  [data-testid="stFileUploaderDropzoneInstructions"] > div > span  { color:var(--cream) !important; font-weight:500; }
  [data-testid="stFileUploaderDropzoneInstructions"] > div > small { color:var(--muted) !important; }
  [data-testid="stFileUploader"] section button {
    background:linear-gradient(135deg, var(--gold), #a88a36) !important;
    color:#1a1208 !important; border:none !important; border-radius:10px !important; font-weight:700 !important;
  }
  .uploadedFile { background:var(--glass2) !important; border-radius:10px !important; border:1px solid var(--line) !important; }

  /* ── Inputs & selects ── */
  [data-testid="stTextInput"] input,
  [data-testid="stSelectbox"] div[data-baseweb="select"] {
    background:var(--glass2) !important; border:1px solid var(--line) !important;
    color:var(--cream) !important; border-radius:10px !important;
  }
  [data-testid="stRadio"] label { color:var(--muted) !important; font-size:0.85rem !important; }

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] { border:1px solid var(--line) !important; border-radius:12px !important; overflow:hidden !important; }
  [data-testid="stDataFrame"] thead tr th {
    background:var(--glass2) !important; color:var(--gold2) !important;
    font-size:0.78rem !important; letter-spacing:0.06em !important;
  }
  [data-testid="stDataFrame"] tbody tr td { color:var(--cream) !important; font-size:0.82rem !important; }
  [data-testid="stDataFrame"] tbody tr:nth-child(even) td { background:rgba(245,240,224,0.02) !important; }

  /* ── Progress bar ── */
  [data-testid="stProgress"] > div > div { background:var(--glass) !important; border:1px solid var(--line) !important; border-radius:999px !important; }
  [data-testid="stProgress"] > div > div > div { background:linear-gradient(90deg, var(--em2), var(--gold)) !important; border-radius:999px !important; }

  /* ── Alerts / info / warning / success ── */
  [data-testid="stInfo"], [data-testid="stSuccess"],
  [data-testid="stWarning"], [data-testid="stError"] {
    border-radius:12px !important; backdrop-filter:blur(10px) !important;
    font-size:0.87rem !important;
  }

  /* ── Code block ── */
  [data-testid="stCode"] pre {
    background:rgba(4,23,15,0.7) !important; border:1px solid var(--line) !important;
    border-radius:12px !important; color:var(--cream) !important;
    font-size:0.82rem !important;
  }

  /* ── Diagnosis card ── */
  .diag-card { padding:1.5rem 1.7rem; margin-bottom:1rem; animation:slideUp 0.5s ease; position:relative; overflow:hidden; }
  .diag-card::after {
    content:''; position:absolute; top:-40px; right:-40px; width:200px; height:200px;
    background:radial-gradient(circle, rgba(201,168,76,0.18), transparent 70%); pointer-events:none;
  }
  .diag-label   { color:var(--gold2); font-size:0.72rem; font-weight:600; letter-spacing:0.14em; text-transform:uppercase; }
  .diag-disease {
    font-family:'Cormorant Garamond', serif;
    font-size:2.1rem; font-weight:600; color:#ff9999;
    letter-spacing:-0.01em; margin:0.4rem 0 0.25rem 0; line-height:1.05; animation:fadeIn 0.7s ease;
  }
  .diag-disease.healthy { color:var(--em2) !important; }
  .diag-sub     { color:var(--muted); font-size:0.85rem; }
  .conf-val     {
    font-family:'Cormorant Garamond', serif;
    font-size:2.4rem; font-weight:600; color:var(--cream); margin:0.2rem 0 0.5rem 0; letter-spacing:-0.02em;
  }
  .conf-bar-bg  { background:rgba(0,0,0,0.3); border-radius:99px; height:6px; width:100%; overflow:hidden; border:1px solid var(--line); }
  .conf-bar-fill {
    background:linear-gradient(90deg, var(--em2), var(--gold)); border-radius:99px; height:100%;
    animation:growBar 1.1s cubic-bezier(.22,1,.36,1) forwards; box-shadow:0 0 12px rgba(201,168,76,0.4);
  }

  /* ── Treatment card ── */
  .treat-card  { padding:1.3rem 1.6rem; margin-bottom:1rem; }
  .treat-title { color:var(--gold2); font-size:0.72rem; font-weight:600; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.65rem; }
  .treat-text  { color:var(--cream); font-size:0.9rem; line-height:1.7; opacity:0.9; }

  /* ── Severity gauge ── */
  .gauge-wrap  { margin-top:1rem; }
  .gauge-track {
    position:relative; height:8px; border-radius:999px;
    background:linear-gradient(90deg, #7fd49b 0%, #e8c068 35%, #e09a3a 70%, #e07070 100%);
    box-shadow:inset 0 1px 2px rgba(0,0,0,0.4), 0 0 14px rgba(201,168,76,0.18);
    border:1px solid var(--line);
  }
  .gauge-marker {
    position:absolute; top:50%; transform:translate(-50%,-50%);
    width:14px; height:14px; border-radius:50%;
    background:#fff; box-shadow:0 0 8px rgba(0,0,0,0.5); border:2px solid rgba(0,0,0,0.3);
  }
  .gauge-scale   { display:flex; justify-content:space-between; margin-top:0.4rem; }
  .gauge-scale span { font-size:0.65rem; color:var(--muted); letter-spacing:0.05em; }
  .gauge-caption { margin-top:0.4rem; font-size:0.78rem; color:var(--muted); }

  /* ── Top-3 predictions ── */
  .pred-card   { padding:1.2rem 1.4rem; }
  .pred-title  { color:var(--gold2); font-size:0.72rem; font-weight:600; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.9rem; }
  .pred-rank   { width:28px; height:28px; border-radius:50%; background:var(--glass2); border:1px solid var(--line); display:grid; place-items:center; color:var(--gold2); font-size:0.8rem; font-weight:700; }
  .pred-name   { color:var(--cream); font-size:0.85rem; font-weight:500; margin-bottom:0.3rem; }
  .pred-bar-bg { background:rgba(0,0,0,0.3); border-radius:99px; height:5px; overflow:hidden; border:1px solid var(--line); }
  .pred-bar-fill { background:linear-gradient(90deg, var(--em2), var(--gold)); height:100%; border-radius:99px; animation:growBar 1s ease forwards; }
  .pred-pct    { color:var(--gold2); font-size:0.82rem; font-weight:600; text-align:right; }

  /* ── Warning banner ── */
  .warn-banner {
    padding:0.75rem 1.2rem; display:flex; align-items:center; gap:0.7rem;
    margin:0.8rem 0; animation:fadeIn 0.6s ease;
    border-color:rgba(232,192,104,0.3) !important;
  }
  .warn-banner span:first-child { color:var(--gold); font-size:1rem; }
  .warn-banner span:last-child  { color:var(--muted); font-size:0.82rem; line-height:1.5; }

  /* ── Image containers ── */
  .img-frame {
    border:1px solid var(--line); border-radius:14px; overflow:hidden;
    animation:fadeIn 0.5s ease; box-shadow:0 10px 32px rgba(0,0,0,0.4);
  }
  .img-frame img { width:100%; display:block; }
  .img-label { color:var(--gold2); font-size:0.85rem; font-weight:600; margin:0.9rem 0 0.4rem 0; letter-spacing:0.06em; text-transform:uppercase; }

  /* ── Timeline ── */
  .timeline-item { display:flex; align-items:flex-start; gap:1rem; padding:0.7rem 0; border-bottom:1px solid var(--line); }
  .timeline-item:last-child { border-bottom:none; }
  .timeline-icon { font-size:1.1rem; width:28px; flex-shrink:0; }
  .timeline-phase { color:var(--gold2); font-size:0.72rem; font-weight:700; letter-spacing:0.08em; margin-bottom:0.15rem; }
  .timeline-desc  { color:var(--cream); font-size:0.85rem; opacity:0.85; }

  /* ── Crop disease table ── */
  .crop-col  { padding:1rem 1.2rem; }
  .crop-name { color:var(--gold2); font-weight:600; font-size:0.95rem; margin-bottom:0.5rem; }
  .crop-disease-healthy { color:#7fd49b; font-size:0.82rem; margin-bottom:0.25rem; }
  .crop-disease-sick    { color:#ff9999; font-size:0.82rem; margin-bottom:0.25rem; }

  /* ── Empty / awaiting state ── */
  .empty-state { padding:3.5rem 2rem; text-align:center; }
  .empty-icon  { font-size:3rem; margin-bottom:1rem; opacity:0.7; }
  .empty-text  { font-family:'Cormorant Garamond', serif; font-size:1.6rem; color:var(--cream); margin-bottom:0.4rem; }
  .empty-sub   { color:var(--muted); font-size:0.88rem; }

  /* ── Footer ── */
  .app-footer {
    display:flex; flex-wrap:wrap; justify-content:center; gap:1.5rem;
    padding:1.2rem 1.6rem; margin-top:2rem; border-top:1px solid var(--line);
  }
  .footer-item { color:var(--muted); font-size:0.74rem; letter-spacing:0.06em; }
  .credits { text-align:center; padding:1.5rem 1rem 0.5rem; }
  .credits-label { color:var(--gold); font-size:0.72rem; letter-spacing:0.14em; margin-bottom:0.4rem; }
  .credits-names { color:var(--cream); font-size:0.88rem; }
  .credits-sep   { color:var(--gold); opacity:0.5; margin:0 0.4rem; }

  /* ── Matplotlib figures on dark bg ── */
  .stImage img  { border-radius:12px; border:1px solid var(--line); }
</style>
"""

_FOOTER_HTML = """
<div class="app-footer">
  <div class="footer-item">✦ ResNet50 + CBAM</div>
  <div class="footer-item">✦ PlantVillage Dataset</div>
  <div class="footer-item">✦ 10 Disease Classes</div>
  <div class="footer-item">✦ ≥94.0% Accuracy</div>
  <div class="footer-item">✦ AI335L Deep Learning Lab · Spring 2026</div>
</div>
<div class="credits">
  <div class="credits-label">✦ Designed &amp; Built By ✦</div>
  <div class="credits-names">
    Hafiz Shahim Abdullah
    <span class="credits-sep">·</span>
    Abubakar Shahid
    <span class="credits-sep">·</span>
    Wajid Hussain Khan
  </div>
</div>
"""


def inject_css() -> None:
    """Inject the shared theme CSS (call once per page, before any other st.* calls)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def footer() -> None:
    """Render the shared page footer."""
    st.markdown(_FOOTER_HTML, unsafe_allow_html=True)


def hero(icon: str, title: str, subtitle: str, chips: list[tuple[str, str]] | None = None) -> None:
    """
    Render the animated hero banner.
    chips: list of (label, extra_class) e.g. [("Model Online", ""), ("≥94.0%", "gold")]
    """
    chips_html = ""
    if chips:
        for label, cls in chips:
            dot = '<span class="dot"></span>' if not cls else ""
            chips_html += f'<span class="stat-chip {cls}">{dot}{label}</span>'
    st.markdown(
        f"""
        <div class="glass hero">
          <div class="hero-row">
            <div class="hero-left">
              <div class="hero-badge">{icon}</div>
              <div>
                <div class="hero-title">{title}</div>
                <div class="hero-subtitle">{subtitle}</div>
                <div class="byline">
                  ✦ Crafted by
                  <b>Hafiz Shahim Abdullah</b>
                  <span class="sep">·</span>
                  <b>Abubakar Shahid</b>
                  <span class="sep">·</span>
                  <b>Wajid Hussain Khan</b>
                </div>
              </div>
            </div>
            <div class="hero-stats">{chips_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
