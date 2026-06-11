"""
CropGuard AI — Disease Severity Estimation
============================================
HSV colour-space analysis to quantify disease extent as a percentage
of total leaf area, then classify into four clinical stages.

Algorithm
---------
1. Convert BGR image to HSV
2. Mask green leaf pixels   (H: 35–85°, S/V: 40–255)
3. Mask disease pixels      (H: 5–30° — brown/yellow necrotic tissue)
4. severity_pct = disease_px / (leaf_px + disease_px)  × 100
5. Map to stage: Early (0–25%) / Moderate (25–50%) / Severe (50–75%) / Critical (75–100%)

Usage
-----
    from src.severity import estimate_severity
    result = estimate_severity("path/to/leaf.jpg")
    print(result["severity_pct"], result["stage"], result["action"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  HSV colour thresholds (calibrated on PlantVillage images)
# ─────────────────────────────────────────────────────────────────────────────

# Healthy green leaf pixels
_LEAF_HSV_LOW  = np.array([35,  40,  40], dtype=np.uint8)
_LEAF_HSV_HIGH = np.array([85, 255, 255], dtype=np.uint8)

# Brown / yellow necrotic lesion pixels
_DISEASE_HSV_LOW  = np.array([ 5,  40,  20], dtype=np.uint8)
_DISEASE_HSV_HIGH = np.array([30, 255, 200], dtype=np.uint8)

# Stage thresholds (%)
_STAGE_THRESHOLDS = [
    ("Critical", 75, 100),
    ("Severe",   50,  75),
    ("Moderate", 25,  50),
    ("Early",     0,  25),
]

_STAGE_ACTIONS: Dict[str, str] = {
    "Early":    "Monitor closely. Apply preventive treatment if recurrent.",
    "Moderate": "Apply targeted fungicide/bactericide within 24–48 hours. Inspect neighbours.",
    "Severe":   "Immediate treatment required. Prune and remove all affected tissue.",
    "Critical": "Remove and destroy infected plants immediately. Sanitise all equipment.",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Core estimation function
# ─────────────────────────────────────────────────────────────────────────────

def estimate_severity(image_path: str | Path) -> Dict:
    """
    Estimate disease severity for a single leaf image.

    Args:
        image_path: File path to a leaf image (JPEG / PNG).

    Returns:
        dict with keys:
            severity_pct  – float in [0, 100]
            stage         – "Early" | "Moderate" | "Severe" | "Critical"
            action        – recommended intervention string
            disease_px    – raw diseased pixel count
            leaf_px       – raw leaf pixel count
            overlay       – BGR image with disease region highlighted (uint8)
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {
            "severity_pct": 0.0,
            "stage": "Unknown",
            "action": "Image could not be read.",
            "disease_px": 0,
            "leaf_px": 0,
            "overlay": None,
        }

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    leaf_mask    = cv2.inRange(img_hsv, _LEAF_HSV_LOW,    _LEAF_HSV_HIGH)
    disease_mask = cv2.inRange(img_hsv, _DISEASE_HSV_LOW, _DISEASE_HSV_HIGH)

    leaf_px    = cv2.countNonZero(leaf_mask)
    disease_px = cv2.countNonZero(disease_mask)
    total_px   = leaf_px + disease_px

    severity_pct = (disease_px / max(total_px, 1)) * 100.0
    severity_pct = min(float(severity_pct), 100.0)

    stage = "Early"
    for s, lo, hi in _STAGE_THRESHOLDS:
        if severity_pct >= lo:
            stage = s
            break

    # Visual overlay: highlight disease pixels in red
    overlay = img_bgr.copy()
    overlay[disease_mask > 0] = (0, 0, 255)  # BGR red

    return {
        "severity_pct": round(severity_pct, 2),
        "stage":        stage,
        "action":       _STAGE_ACTIONS[stage],
        "disease_px":   disease_px,
        "leaf_px":      leaf_px,
        "overlay":      overlay,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Batch evaluation (for validation / calibration)
# ─────────────────────────────────────────────────────────────────────────────

def batch_severity(image_paths: list) -> list:
    """
    Run ``estimate_severity`` over a list of image paths.

    Returns a list of result dicts (same structure as single-image output).
    """
    return [estimate_severity(p) for p in image_paths]


# ─────────────────────────────────────────────────────────────────────────────
#  Stage utilities
# ─────────────────────────────────────────────────────────────────────────────

def severity_to_colour(stage: str) -> str:
    """Return a hex colour suitable for UI display."""
    return {
        "Early":    "#27ae60",   # green
        "Moderate": "#f39c12",   # amber
        "Severe":   "#e67e22",   # orange
        "Critical": "#e74c3c",   # red
        "Unknown":  "#95a5a6",   # grey
    }.get(stage, "#95a5a6")


def severity_bar_label(severity_pct: float, stage: str) -> str:
    """Short label for Streamlit progress bar."""
    return f"{severity_pct:.1f}%  ({stage})"
