"""
CropGuard AI — Diagnose Page
"""
from __future__ import annotations
import io, sys, tempfile, os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from theme import inject_css, hero, footer

from src.model import CropGuardCNN
from src.utils import (
    CLASS_DISPLAY, TARGET_CLASSES, NUM_CLASSES,
    get_val_test_transform, denormalize, set_seed,
)
from src.severity import estimate_severity, severity_to_colour, severity_bar_label
from src.treatment import get_treatment

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diagnose | CropGuard AI", page_icon="🔬", layout="wide")
inject_css()

hero(
    icon="🔬",
    title="Disease Diagnosis",
    subtitle="Upload a leaf image · Grad-CAM attention · HSV severity · Treatment plan",
    chips=[("Model Ready", ""), ("Grad-CAM", "gold"), ("10 Classes", "")],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT   = Path(__file__).resolve().parents[2] / "checkpoints" / "cropguard_resnet50_cbam_phase3.pt"

# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model():
    set_seed(42)
    model = CropGuardCNN(num_classes=NUM_CLASSES, dropout=0.38, use_cbam=True).to(DEVICE)
    if CKPT.exists():
        ckpt = torch.load(CKPT, map_location=DEVICE)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        st.sidebar.markdown('<div style="color:#7fd49b;font-size:0.82rem;">✅ Phase 4 checkpoint loaded</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div style="color:var(--gold2);font-size:0.82rem;">⚠️ Checkpoint not found — demo mode</div>', unsafe_allow_html=True)
    model.eval()
    return model

model    = load_model()
transform = get_val_test_transform()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div style="color:var(--gold2);font-weight:600;font-size:0.9rem;letter-spacing:0.06em;margin-bottom:0.8rem;">✦ OPTIONS</div>', unsafe_allow_html=True)
show_gradcam   = st.sidebar.checkbox("Show Grad-CAM heatmap",    value=True)
show_top3      = st.sidebar.checkbox("Show top-3 predictions",   value=True)
show_treatment = st.sidebar.checkbox("Show treatment plan",       value=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass section-card">
  <div class="section-title">✦ Upload Leaf Image</div>
  <div style="color:var(--muted);font-size:0.85rem;margin-bottom:0.8rem;">
    JPEG or PNG · ideally 224×224 or larger · single leaf in frame
  </div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload leaf",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div class="glass empty-state">
      <div class="empty-icon">🍃</div>
      <div class="empty-text">Awaiting your leaf</div>
      <div class="empty-sub">Upload a leaf image above to begin diagnosis</div>
    </div>
    """, unsafe_allow_html=True)
    footer()
    st.stop()

# ── Image validation — reject non-leaf images ─────────────────────────────────
import base64, json as _json, urllib.request as _urlreq, urllib.error as _urlerr
from io import BytesIO as _BytesIO

# ── Layer 1: HSV green-pixel heuristic (instant, no network) ─────────────────
def _hsv_leaf_check(pil_img) -> tuple[bool, str]:
    """
    Returns (looks_like_leaf, reason).
    A real plant leaf has a meaningful fraction of green/yellow-green pixels.
    Skin tones, office interiors, etc. do not.
    """
    import colorsys
    img_small = pil_img.resize((64, 64)).convert("RGB")
    pixels = list(img_small.getdata())
    green_count = 0
    skin_count  = 0
    for r, g, b in pixels:
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        h_deg = h * 360
        # Green / yellow-green hue range, decent saturation & brightness
        if 40 <= h_deg <= 170 and s > 0.15 and v > 0.15:
            green_count += 1
        # Skin tone: hue 0-40 or 340-360, low-medium saturation
        if (h_deg <= 40 or h_deg >= 340) and 0.1 < s < 0.65 and v > 0.25:
            skin_count += 1

    total = len(pixels)
    green_pct = green_count / total
    skin_pct  = skin_count  / total

    # Dominant skin + low green → almost certainly a person/indoor shot
    if skin_pct > 0.20 and green_pct < 0.10:
        return False, (
            f"Image appears to contain skin tones ({skin_pct*100:.0f}% of pixels) "
            f"with very little plant-green ({green_pct*100:.0f}%). "
            "This does not look like a plant leaf."
        )
    # Very low green with no compensating factor
    if green_pct < 0.05 and skin_pct < 0.05:
        return False, (
            f"Only {green_pct*100:.0f}% green pixels detected. "
            "The image does not appear to show plant foliage."
        )
    return True, ""


# ── Layer 2: Claude Haiku visual check (urllib, no extra packages) ────────────
def _claude_leaf_check(jpeg_bytes: bytes) -> tuple[bool, str]:
    """
    Calls Claude Haiku via urllib (stdlib only).
    Returns (is_leaf, reason). Fails open on any error.
    """
    b64 = base64.b64encode(jpeg_bytes).decode()
    payload = _json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 80,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Look at this image carefully. Does it show a plant leaf "
                        "(healthy or diseased)? People, animals, objects, food, "
                        "buildings, and documents are NOT plant leaves. "
                        "Reply with ONLY valid JSON — no markdown, no extra text: "
                        '{"is_leaf": true_or_false, "reason": "one sentence"}'
                    ),
                },
            ],
        }],
    }).encode()

    req = _urlreq.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=12) as resp:
            body = _json.loads(resp.read())
        text = body["content"][0]["text"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = _json.loads(text)
        return bool(data.get("is_leaf", True)), str(data.get("reason", ""))
    except Exception:
        return True, ""   # fail open


# ── Run both layers ───────────────────────────────────────────────────────────
with st.spinner("Validating image…"):
    img_bytes = uploaded.getvalue()
    _tmp_pil  = Image.open(_BytesIO(img_bytes)).convert("RGB")

    # Layer 1 — fast pixel check
    _hsv_ok, _hsv_reason = _hsv_leaf_check(_tmp_pil)

    if _hsv_ok:
        # Layer 2 — Claude visual check (only if HSV passes)
        _buf = _BytesIO()
        _tmp_pil.save(_buf, format="JPEG", quality=85)
        is_leaf, leaf_reason = _claude_leaf_check(_buf.getvalue())
    else:
        is_leaf, leaf_reason = False, _hsv_reason

if not is_leaf:
    st.markdown(f"""
    <div class="glass" style="padding:2.5rem 2rem; text-align:center; margin-top:1rem; border-color:rgba(224,112,112,0.4);">
      <div style="font-size:3rem; margin-bottom:1rem;">🚫</div>
      <div style="font-family:'Cormorant Garamond',serif; font-size:1.8rem; color:#ff9999; margin-bottom:0.6rem;">
        Invalid Image
      </div>
      <div style="color:var(--muted); font-size:0.92rem; max-width:480px; margin:0 auto 1rem auto; line-height:1.65;">
        {leaf_reason if leaf_reason else "The uploaded image does not appear to contain a plant leaf."}
      </div>
      <div style="background:rgba(224,112,112,0.1); border:1px solid rgba(224,112,112,0.3);
                  border-radius:12px; padding:1rem 1.4rem; display:inline-block; text-align:left; max-width:440px;">
        <div style="color:#ff9999; font-size:0.75rem; font-weight:600; letter-spacing:0.1em; margin-bottom:0.5rem;">
          ✦ THIS MODEL IS TRAINED FOR
        </div>
        <div style="color:var(--cream); font-size:0.85rem; line-height:1.7;">
          🍅 Tomato leaves &nbsp;·&nbsp; 🥔 Potato leaves<br>
          🌶️ Pepper leaves &nbsp;·&nbsp; 🌽 Corn leaves<br>
          <span style="color:var(--muted); font-size:0.8rem;">10 disease classes from the PlantVillage dataset</span>
        </div>
      </div>
      <div style="color:var(--muted); font-size:0.8rem; margin-top:1.2rem;">
        Please upload a clear photo of a plant leaf to continue.
      </div>
    </div>
    """, unsafe_allow_html=True)
    footer()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
pil_img    = Image.open(uploaded).convert("RGB")
pil_224    = pil_img.resize((224, 224))
img_tensor = transform(pil_224)
input_batch = img_tensor.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(input_batch)
    probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

top3_idx   = probs.argsort()[::-1][:3]
pred_idx   = int(top3_idx[0])
pred_name  = TARGET_CLASSES[pred_idx]
pred_disp  = CLASS_DISPLAY[pred_idx]
confidence = float(probs[pred_idx])
is_healthy = "Healthy" in pred_disp

# Severity
sev_pct, sev_stage = 0.0, "Early"
if not is_healthy:
    suffix = Path(uploaded.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    sev       = estimate_severity(tmp_path)
    os.unlink(tmp_path)
    sev_pct   = sev["severity_pct"]
    sev_stage = sev["stage"]

# Severity gauge position
SEV_POS = {"Early": 12, "Moderate": 38, "Severe": 68, "Critical": 90}
sev_pos = SEV_POS.get(sev_stage, 38)
SEV_COLOR = {"Early": "#7fd49b", "Moderate": "#e8c068", "Severe": "#e09a3a", "Critical": "#e07070"}
sev_color = SEV_COLOR.get(sev_stage, "#e8c068")

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.45], gap="large")

with left:
    # Image preview
    st.markdown('<div class="img-label">Uploaded Leaf</div>', unsafe_allow_html=True)
    st.markdown('<div class="img-frame">', unsafe_allow_html=True)
    st.image(pil_img, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Grad-CAM
    if show_gradcam:
        st.markdown('<div class="img-label" style="margin-top:1rem;">Grad-CAM Attention</div>', unsafe_allow_html=True)
        try:
            from src.gradcam import GradCAMWrapper
            cam = GradCAMWrapper(model, DEVICE)
            overlay, _ = cam.explain(img_tensor, target_class=pred_idx)
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(overlay, caption="CBAM-4 spatial attention overlay", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except ImportError:
            st.markdown('<div class="glass warn-banner"><span>⚠</span><span>Install <code>pytorch-grad-cam</code> for the heatmap.</span></div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="glass warn-banner"><span>⚠</span><span>Grad-CAM error: {e}</span></div>', unsafe_allow_html=True)

with right:
    # Diagnosis card
    diag_class = "healthy" if is_healthy else ""
    st.markdown(f"""
    <div class="glass diag-card">
      <div style="display:flex; justify-content:space-between; gap:1.5rem; flex-wrap:wrap;">
        <div style="flex:1.4; min-width:220px;">
          <div class="diag-label">✦ Diagnosis</div>
          <div class="diag-disease {diag_class}">{pred_disp}</div>
          <div class="diag-sub">Identified from the uploaded leaf image</div>
          <div class="gauge-wrap">
            <div class="gauge-track"><div class="gauge-marker" style="left:{sev_pos}%;"></div></div>
            <div class="gauge-scale"><span>Early</span><span>Moderate</span><span>Severe</span><span>Critical</span></div>
            <div class="gauge-caption">Severity · <b style="color:{sev_color};">{sev_stage if not is_healthy else "None"}</b></div>
          </div>
        </div>
        <div style="flex:1; min-width:160px;">
          <div class="diag-label">Confidence</div>
          <div class="conf-val">{confidence*100:.1f}%</div>
          <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{confidence*100:.1f}%"></div></div>
          {"<div style='margin-top:0.8rem;padding:0.5rem 0.8rem;background:rgba(127,212,155,0.12);border:1px solid rgba(127,212,155,0.3);border-radius:8px;color:#7fd49b;font-size:0.82rem;'>✓ Plant appears healthy</div>" if is_healthy else f"<div style='margin-top:0.8rem;padding:0.5rem 0.8rem;background:rgba(255,153,153,0.1);border:1px solid rgba(255,153,153,0.3);border-radius:8px;color:#ff9999;font-size:0.82rem;'>{sev_pct:.1f}% leaf area affected</div>"}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Top-3
    if show_top3:
        max_pct = float(probs[top3_idx[0]]) if probs[top3_idx[0]] > 0 else 1
        rows_html = ""
        for rank, idx in enumerate(top3_idx, 1):
            disp_name = CLASS_DISPLAY[idx]
            pct = float(probs[idx])
            bar_w = round((pct / max_pct) * 100, 2)
            pct_label = f"{pct*100:.1f}%"
            display = disp_name if len(disp_name) <= 42 else disp_name[:40] + "..."
            rows_html += (
                '<div style="display:grid;grid-template-columns:32px 1fr 52px;'
                'align-items:center;gap:0.7rem;margin-bottom:0.7rem;">'
                '<div class="pred-rank">' + str(rank) + '</div>'
                '<div>'
                '<div class="pred-name">' + display + '</div>'
                '<div class="pred-bar-bg"><div class="pred-bar-fill" style="width:' + str(bar_w) + '%"></div></div>'
                '</div>'
                '<div class="pred-pct">' + pct_label + '</div>'
                '</div>'
            )
        top3_html = (
            '<div class="glass pred-card">'
            '<div class="pred-title">&#10022; Top 3 Predictions</div>'
            + rows_html
            + '</div>'
        )
        st.markdown(top3_html, unsafe_allow_html=True)

    # Treatment
    if show_treatment:
        treatment = get_treatment(pred_name, severity_stage=sev_stage if not is_healthy else "Early")
        title_label = "Care Recommendation" if is_healthy else "Recommended Treatment"
        st.markdown(f"""
        <div class="glass treat-card" style="margin-top:0.8rem;">
          <div class="treat-title">✦ {title_label}</div>
          <div class="treat-text">{treatment}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass warn-banner" style="margin-top:0.8rem;">
      <span>⚠</span>
      <span>This is an AI prediction. For high-stakes decisions, confirm with a certified plant pathologist.</span>
    </div>
    """, unsafe_allow_html=True)

# ── Download ──────────────────────────────────────────────────────────────────
treatment_text = get_treatment(pred_name, severity_stage=sev_stage if not is_healthy else "Early")
report_text = (
    f"CropGuard AI — Diagnostic Report\n{'='*50}\n"
    f"Predicted Disease : {pred_disp}\n"
    f"Confidence        : {confidence*100:.1f}%\n"
    f"Severity          : {sev_pct:.1f}% ({sev_stage})\n\n"
    f"Treatment:\n{treatment_text}\n\n"
    "Top 3 Predictions:\n"
    + "\n".join([f"  {i+1}. {CLASS_DISPLAY[idx]} — {probs[idx]*100:.2f}%" for i, idx in enumerate(top3_idx)])
    + "\n\n— Crafted by Hafiz Shahim Abdullah · Abubakar Shahid · Wajid Hussain Khan"
)

st.download_button(
    "⬇ Download Diagnostic Report",
    data=report_text,
    file_name="cropguard_report.txt",
    mime="text/plain",
)

footer()
