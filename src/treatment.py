"""
CropGuard AI — Treatment Recommendation Database
=================================================
Expert-curated treatment protocols for all 10 PlantVillage classes.

Each entry contains:
  - cause        : Pathogen and environmental triggers
  - symptoms     : Visual identification guide
  - fungicide    : Chemical treatment options with dosage
  - organic      : Organic / biological alternatives
  - prevention   : Cultural and agronomic prevention measures

Usage
-----
    from src.treatment import get_treatment, TREATMENT_DB

    report = get_treatment("Tomato___Late_blight", severity_stage="Severe")
    print(report)
"""

from __future__ import annotations

from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Treatment Database
# ─────────────────────────────────────────────────────────────────────────────

TREATMENT_DB: Dict[str, Dict[str, str]] = {
    "Tomato___Early_blight": {
        "common_name": "Tomato Early Blight",
        "pathogen":    "Alternaria solani (fungus)",
        "cause": (
            "Warm, humid conditions (24–29°C). Spread by rain splash and "
            "wind. Favoured by drought stress followed by wet periods."
        ),
        "symptoms": (
            "Dark brown spots with concentric rings (target-board pattern) "
            "on lower / older leaves. Yellow halos around lesions. Defoliation "
            "begins from the base upward."
        ),
        "fungicide": (
            "Mancozeb 75 WP (2 g/L) or Chlorothalonil 75 WP (1.5 ml/L). "
            "Apply every 7–10 days starting at first symptom. "
            "Rotate actives to prevent resistance."
        ),
        "organic": (
            "Neem oil spray (5 ml/L water, weekly). "
            "Copper-based Bordeaux mixture (1% solution). "
            "Bacillus subtilis biocontrol (Serenade)."
        ),
        "prevention": (
            "Crop rotation — avoid Solanaceae family for 3 years. "
            "Remove and compost infected plant debris. "
            "Mulch to reduce spore splash. Stake plants for air circulation."
        ),
    },

    "Tomato___Late_blight": {
        "common_name": "Tomato Late Blight",
        "pathogen":    "Phytophthora infestans (oomycete)",
        "cause": (
            "Cool (10–20°C), wet weather. Spreads rapidly in humid conditions. "
            "Historically caused the Irish Famine (1840s)."
        ),
        "symptoms": (
            "Irregular water-soaked lesions on leaves → dark brown necrotic "
            "patches. White mold (sporangiophores) on leaf undersides in humid "
            "conditions. Rapid tissue collapse."
        ),
        "fungicide": (
            "Cymoxanil + Mancozeb (Curzate M 72 WP, 2.5 g/L). "
            "Metalaxyl (Ridomil) for systemic protection. "
            "Apply at 5–7 day intervals; strictly alternate actives."
        ),
        "organic": (
            "Copper hydroxide (1.5 g/L) preventively. "
            "Immediately remove and bag infected tissue — do not compost. "
            "Phosphorous acid (fosetyl-Al) as a biostimulant."
        ),
        "prevention": (
            "Improve air circulation — stake and prune lower foliage. "
            "Avoid overhead irrigation; use drip system. "
            "Use resistant varieties (Mountain Magic, Defiant)."
        ),
    },

    "Tomato___healthy": {
        "common_name": "Tomato Healthy",
        "pathogen":    "None — no disease detected",
        "cause":       "No disease detected.",
        "symptoms":    "Bright green, uniform leaf with no spots or lesions.",
        "fungicide": (
            "No treatment required. "
            "Apply preventive copper spray monthly during wet season."
        ),
        "organic": (
            "Maintain balanced NPK fertilisation. "
            "Ensure adequate drainage to prevent root stress. "
            "Foliar seaweed extract to boost plant immunity."
        ),
        "prevention": (
            "Regular scouting (weekly). Remove weeds that harbour pathogens. "
            "Monitor for early signs of stress or discolouration."
        ),
    },

    "Potato___Early_blight": {
        "common_name": "Potato Early Blight",
        "pathogen":    "Alternaria solani (fungus)",
        "cause": (
            "Drought stress followed by warm, humid periods (24–29°C). "
            "Primarily affects older, stressed plants."
        ),
        "symptoms": (
            "Small dark spots with yellow halos and concentric rings on "
            "older/lower leaves. Lesions enlarge and merge. "
            "Premature defoliation reduces tuber yield."
        ),
        "fungicide": (
            "Azoxystrobin 23 SC (Amistar, 1 ml/L) or "
            "Difenoconazole 25 EC (Score, 0.5 ml/L). "
            "Apply at first symptom; repeat every 10–14 days."
        ),
        "organic": (
            "Bacillus subtilis-based biocontrol (Serenade). "
            "Compost tea spray as a biostimulant. "
            "Neem oil (5 ml/L)."
        ),
        "prevention": (
            "Use certified, disease-free seed tubers. "
            "Adequate potassium nutrition strengthens cell walls. "
            "Hill soil around plants to protect tubers."
        ),
    },

    "Potato___Late_blight": {
        "common_name": "Potato Late Blight",
        "pathogen":    "Phytophthora infestans (oomycete)",
        "cause": (
            "Cool (10–20°C) and wet conditions. Spreads explosively — "
            "entire field can be destroyed within days in ideal conditions."
        ),
        "symptoms": (
            "Large, irregular brown lesions with pale green borders. "
            "Rapid tissue death. White sporulation on leaf undersides. "
            "Infected tubers show russet-brown rot."
        ),
        "fungicide": (
            "Metalaxyl-M + Mancozeb (Ridomil Gold MZ 68 WG, 2.5 g/L) — "
            "systemic + contact protection. "
            "Propamocarb HCl (Previcur N) for soil drench. "
            "Apply preventively every 5–7 days in high-risk weather."
        ),
        "organic": (
            "Destroy infected tubers — do not compost. "
            "Copper sulfate (Bordeaux mixture, 1%) preventively. "
            "Biofumigation with Brassica green manure."
        ),
        "prevention": (
            "Plant blight-resistant varieties (Sarpo Mira, Cara). "
            "Avoid dense planting — allow airflow between rows. "
            "Destroy volunteer plants and nightshade weeds."
        ),
    },

    "Potato___healthy": {
        "common_name": "Potato Healthy",
        "pathogen":    "None — no disease detected",
        "cause":       "No disease detected.",
        "symptoms":    "Healthy green leaf with no lesions or pustules.",
        "fungicide": (
            "Preventive Mancozeb spray (2 g/L) on 7-day schedule "
            "during prolonged wet periods."
        ),
        "organic": (
            "Balanced NPK with emphasis on potassium (K). "
            "Compost application for microbiome health."
        ),
        "prevention": (
            "Certify seed stock annually. "
            "3-year crop rotation with non-Solanaceae. "
            "Monitor nightshade weeds (reservoir hosts)."
        ),
    },

    "Pepper,_bell___Bacterial_spot": {
        "common_name": "Pepper Bacterial Spot",
        "pathogen":    "Xanthomonas euvesicatoria (bacterium)",
        "cause": (
            "Spread by rain splash, contaminated seed, and infected transplants. "
            "Favoured by warm (24–30°C), wet weather."
        ),
        "symptoms": (
            "Small water-soaked spots → necrotic centres with yellow halos. "
            "Raised, scabby lesions on fruit. "
            "Severe defoliation under prolonged wet conditions."
        ),
        "fungicide": (
            "Copper hydroxide + Mancozeb tank mix (1.5 g/L each). "
            "Bactericide: Streptomycin sulfate (where registered, 1 g/L). "
            "Apply every 7 days during wet periods."
        ),
        "organic": (
            "Copper sulfate spray (1 g/L). "
            "Avoid overhead irrigation. "
            "Remove and destroy infected plant material."
        ),
        "prevention": (
            "Hot water seed treatment (52°C for 30 min) before planting. "
            "Use certified pathogen-free transplants. "
            "Plant resistant varieties (Revolution, Declaration)."
        ),
    },

    "Pepper,_bell___healthy": {
        "common_name": "Pepper Healthy",
        "pathogen":    "None — no disease detected",
        "cause":       "No disease detected.",
        "symptoms":    "Deep green, smooth leaf surface with no water-soaked spots.",
        "fungicide": (
            "No treatment required. "
            "Apply preventive copper spray before rainy season."
        ),
        "organic": (
            "Foliar feeding with seaweed extract to boost plant immunity. "
            "Balanced fertilisation — avoid excess nitrogen."
        ),
        "prevention": (
            "Drip irrigation — avoid wetting foliage. "
            "Inspect incoming transplants for symptoms. "
            "Scout weekly during warm, wet periods."
        ),
    },

    "Corn_(maize)___Common_rust_": {
        "common_name": "Corn Common Rust",
        "pathogen":    "Puccinia sorghi (fungus)",
        "cause": (
            "Airborne urediniospores spreading from infected tissue. "
            "Favoured by cool temperatures (16–23°C) and >95% relative humidity."
        ),
        "symptoms": (
            "Small oval reddish-brown pustules (urediniospores) scattered on "
            "both leaf surfaces. Pustules rupture to release brick-red powdery "
            "spores. Heavily infected leaves yellow and die."
        ),
        "fungicide": (
            "Propiconazole 25 EC (Tilt, 1 ml/L) or "
            "Tebuconazole 25.9 EW (0.75 ml/L). "
            "Apply at early pustule stage; repeat after 14 days if needed."
        ),
        "organic": (
            "Sulfur dust (3 kg/ha) at early stages. "
            "Neem extract spray (5 ml/L). "
            "Remove heavily infected leaves to reduce inoculum."
        ),
        "prevention": (
            "Plant rust-resistant hybrids (SC403, DK8031). "
            "Avoid late planting — crop should reach silking before cool season. "
            "Monitor from V6 stage onwards."
        ),
    },

    "Corn_(maize)___healthy": {
        "common_name": "Corn Healthy",
        "pathogen":    "None — no disease detected",
        "cause":       "No disease detected.",
        "symptoms":    "Uniform green leaf with no pustules, lesions, or chlorosis.",
        "fungicide": (
            "No treatment required. "
            "Scout weekly from V6 to silking; intervene only if pustules appear."
        ),
        "organic": (
            "Balanced N-P-K; avoid excess nitrogen (increases disease susceptibility). "
            "Adequate spacing (60–75 cm) for air circulation."
        ),
        "prevention": (
            "Proper plant spacing for airflow. "
            "Weed control — wild grasses can harbour rust. "
            "Use certified hybrid seed."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  Formatted report generator
# ─────────────────────────────────────────────────────────────────────────────

_URGENCY_MAP: Dict[str, str] = {
    "Early":    "🟢 LOW — Monitor and apply preventive treatment.",
    "Moderate": "🟡 MEDIUM — Apply targeted treatment within 24–48 hours.",
    "Severe":   "🟠 HIGH — Immediate treatment required. Prune affected tissue.",
    "Critical": "🔴 CRITICAL — Remove and destroy infected plants immediately.",
    "Unknown":  "⚪ UNKNOWN — Please re-examine the image.",
}


def get_treatment(
    class_name: str,
    severity_stage: str = "Early",
    as_dict: bool = False,
) -> str | dict:
    """
    Retrieve a formatted treatment recommendation report.

    Args:
        class_name:     PlantVillage folder name (key in TREATMENT_DB).
        severity_stage: One of "Early" | "Moderate" | "Severe" | "Critical".
        as_dict:        If True, return raw dict instead of formatted string.

    Returns:
        Formatted string report, or raw dict if *as_dict* is True.
    """
    info = TREATMENT_DB.get(class_name)
    if info is None:
        # Fallback: try display name lookup
        from src.utils import DISPLAY_TO_CLASS
        class_name = DISPLAY_TO_CLASS.get(class_name, class_name)
        info = TREATMENT_DB.get(class_name)

    if info is None:
        return "⚠️  No treatment data available for this class."

    if as_dict:
        return {**info, "urgency": _URGENCY_MAP.get(severity_stage, "")}

    return (
        f"Disease    : {info['common_name']}\n"
        f"Pathogen   : {info['pathogen']}\n"
        f"Urgency    : {_URGENCY_MAP.get(severity_stage, '')}\n"
        f"\nCause      : {info['cause']}\n"
        f"\nSymptoms   : {info['symptoms']}\n"
        f"\nFungicide  : {info['fungicide']}\n"
        f"\nOrganic    : {info['organic']}\n"
        f"\nPrevention : {info['prevention']}"
    )


def list_diseases() -> list:
    """Return a list of all 10 disease/healthy class names."""
    return list(TREATMENT_DB.keys())
