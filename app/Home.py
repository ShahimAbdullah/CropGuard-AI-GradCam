"""
CropGuard AI — Home Page
"""
import streamlit as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from theme import inject_css, hero, footer

st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Hero ─────────────────────────────────────────────────────────────────────
hero(
    icon="🌿",
    title="CropGuard AI",
    subtitle="Explainable Deep Learning for Crop Protection · ResNet50 + CBAM · PlantVillage",
    chips=[
        ("Model Online", ""),
        ("≥94.0% Accuracy", "gold"),
        ("🧠 ResNet50+CBAM", ""),
        ("10 Classes", "gold"),
    ],
)

# ── Metrics ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test Accuracy",      "≥ 94.0%",  "+30.1 pp vs MLP")
c2.metric("Macro F1-Score",     "≥ 0.910",  "+0.355 vs MLP")
c3.metric("ROC-AUC",            "≥ 0.980",  "+0.040 vs MLP")
c4.metric("Potato Healthy F1",  "≥ 0.700",  "+0.700 via VAE")

# ── About ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass section-card" style="margin-top:1.2rem;">
  <div class="section-title">✦ About CropGuard AI</div>
  <p style="color:var(--cream); font-size:0.92rem; line-height:1.75; opacity:0.9; margin:0;">
    CropGuard AI is a deep learning system for <strong style="color:var(--gold2);">automated crop disease
    classification and treatment recommendation</strong> developed across six iterative phases at
    NASTP Institute of Information Technology (Spring&nbsp;2026).
  </p>
  <p style="color:var(--cream); font-size:0.92rem; line-height:1.75; opacity:0.9; margin:0.8rem 0 0 0;">
    <strong style="color:var(--gold2);">Core architecture:</strong> ResNet50 backbone (ImageNet pretrained)
    augmented with dual Convolutional Block Attention Modules (CBAM), trained on a 10-class subset of the
    PlantVillage dataset covering tomato, potato, pepper, and corn diseases.
  </p>
</div>
""", unsafe_allow_html=True)

# Key innovations
st.markdown("""
<div class="glass section-card" style="margin-top:0.8rem;">
  <div class="section-title">✦ Key Innovations</div>
  <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(260px,1fr)); gap:0.8rem;">
    <div style="background:var(--glass2); border:1px solid var(--line); border-radius:12px; padding:1rem;">
      <div style="color:var(--gold2); font-weight:600; font-size:0.85rem; margin-bottom:0.35rem;">🎯 CBAM Attention</div>
      <div style="color:var(--muted); font-size:0.82rem; line-height:1.55;">Localises disease lesions for superior Grad-CAM explainability</div>
    </div>
    <div style="background:var(--glass2); border:1px solid var(--line); border-radius:12px; padding:1rem;">
      <div style="color:var(--gold2); font-weight:600; font-size:0.85rem; margin-bottom:0.35rem;">🔬 VAE Augmentation</div>
      <div style="color:var(--muted); font-size:0.82rem; line-height:1.55;">Resolves Potato Healthy class imbalance (152 → 664 images, FID=42.7)</div>
    </div>
    <div style="background:var(--glass2); border:1px solid var(--line); border-radius:12px; padding:1rem;">
      <div style="color:var(--gold2); font-weight:600; font-size:0.85rem; margin-bottom:0.35rem;">⚡ Optuna HPO</div>
      <div style="color:var(--muted); font-size:0.82rem; line-height:1.55;">30 Bayesian TPE trials identify optimal hyperparameters</div>
    </div>
    <div style="background:var(--glass2); border:1px solid var(--line); border-radius:12px; padding:1rem;">
      <div style="color:var(--gold2); font-weight:600; font-size:0.85rem; margin-bottom:0.35rem;">🌿 HSV Severity</div>
      <div style="color:var(--muted); font-size:0.82rem; line-height:1.55;">Quantifies disease extent as a percentage of leaf area</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Navigation cards ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title" style="margin-top:1.4rem; padding-left:0.2rem;">✦ Navigate the Dashboard</div>', unsafe_allow_html=True)
n1, n2, n3 = st.columns(3)
with n1:
    st.markdown("""
    <div class="glass nav-card">
      <div class="icon">🔬</div>
      <div class="title">Diagnose</div>
      <div class="desc">Upload a leaf image to detect disease, visualise Grad-CAM attention,
      estimate severity, and receive a treatment plan.</div>
      <div class="arrow">→ pages/1_Diagnose</div>
    </div>
    """, unsafe_allow_html=True)
with n2:
    st.markdown("""
    <div class="glass nav-card">
      <div class="icon">📊</div>
      <div class="title">Analytics</div>
      <div class="desc">Explore the PlantVillage dataset: class distributions, RGB channel
      analysis, PCA/t-SNE feature visualisations, and augmentation demos.</div>
      <div class="arrow">→ pages/2_Analytics</div>
    </div>
    """, unsafe_allow_html=True)
with n3:
    st.markdown("""
    <div class="glass nav-card">
      <div class="icon">📈</div>
      <div class="title">Performance</div>
      <div class="desc">Review model performance across all phases, per-class F1 breakdown,
      confusion matrix, ROC curves, and robustness analysis.</div>
      <div class="arrow">→ pages/3_Performance</div>
    </div>
    """, unsafe_allow_html=True)

# ── Supported diseases ────────────────────────────────────────────────────────
st.markdown('<div class="section-title" style="margin-top:1.4rem; padding-left:0.2rem;">✦ Supported Diseases</div>', unsafe_allow_html=True)
crops = {
    ("🍅", "Tomato"):  [("Early Blight", False), ("Late Blight", False), ("Healthy", True)],
    ("🥔", "Potato"):  [("Early Blight", False), ("Late Blight", False), ("Healthy", True)],
    ("🌶️","Pepper"):  [("Bacterial Spot", False), ("Healthy", True)],
    ("🌽", "Corn"):    [("Common Rust", False), ("Healthy", True)],
}
dcols = st.columns(4)
for col, ((icon2, cname), diseases) in zip(dcols, crops.items()):
    with col:
        rows = "".join(
            f'<div class="{"crop-disease-healthy" if h else "crop-disease-sick"}">{"🟢" if h else "🔴"} {d}</div>'
            for d, h in diseases
        )
        st.markdown(f"""
        <div class="glass crop-col">
          <div class="crop-name">{icon2} {cname}</div>
          {rows}
        </div>
        """, unsafe_allow_html=True)

# ── Phase timeline ────────────────────────────────────────────────────────────
phases = [
    ("✅", "Phase 1", "Project Setup & Literature Review"),
    ("✅", "Phase 2", "EDA, Preprocessing & MLP Baseline (63.88%)"),
    ("✅", "Phase 3", "ResNet50 + CBAM CNN (92.1%)"),
    ("✅", "Phase 4", "Transfer Learning + VAE + HPO (≥94.0%)"),
    ("✅", "Phase 5", "Rigorous Evaluation, Ablations & Robustness"),
    ("🚀", "Phase 6", "Streamlit Dashboard & Final Deployment"),
]
items_html = "".join(
    f"""<div class="timeline-item">
      <div class="timeline-icon">{s}</div>
      <div>
        <div class="timeline-phase">{p}</div>
        <div class="timeline-desc">{d}</div>
      </div>
    </div>"""
    for s, p, d in phases
)
st.markdown(f"""
<div class="glass section-card" style="margin-top:1.2rem;">
  <div class="section-title">✦ Development Timeline</div>
  {items_html}
</div>
""", unsafe_allow_html=True)

footer()
