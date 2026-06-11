# CropGuard AI — Deployment Guide

## Local Development

### 1. Clone and Install

```bash
git clone https://github.com/Shahim01/CropGuard-AI.git
cd CropGuard-AI

# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate cropguard-ai

# Option B: pip
pip install -r requirements.txt
```

### 2. Download Model Weights

```bash
# Download from HuggingFace Hub (recommended — avoids large git files)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Shahim01/CropGuard-AI',
    filename='best_cropguard_phase4.pt',
    local_dir='./checkpoints'
)
"

# OR place weights manually at:
# checkpoints/best_cropguard_phase4.pt
```

### 3. Run the Streamlit Dashboard

```bash
streamlit run app/Home.py
```

Open `http://localhost:8501` in your browser.

---

## Docker Deployment

```bash
# Build image
docker build -t cropguard-ai .

# Run with CPU
docker run -p 8501:8501 \
  -e MODEL_CHECKPOINT=/app/checkpoints/best_cropguard_phase4.pt \
  cropguard-ai

# Run with GPU (requires nvidia-container-toolkit)
docker run --gpus all -p 8501:8501 \
  -e MODEL_CHECKPOINT=/app/checkpoints/best_cropguard_phase4.pt \
  cropguard-ai

# Health check
curl http://localhost:8501/healthz
```

---

## Cloud Hosting (Streamlit Community Cloud)

1. Fork the repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your fork, branch `main`, and entry point `app/Home.py`.
4. In **Secrets**, add:
   ```toml
   MODEL_CHECKPOINT = "/mount/src/cropguard-ai/checkpoints/best_cropguard_phase4.pt"
   ```
5. Click **Deploy**.

**Model size note:** Free Streamlit Cloud has a 1 GB RAM limit. The ResNet50+CBAM
checkpoint (~100 MB) fits comfortably. If RAM is exceeded, switch to the
MobileNetV2 deployment model (`checkpoints/mobilenet_cropguard.pt`, ~14 MB).

---

## HuggingFace Spaces

```bash
# Install HF CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Create Space
huggingface-cli repo create cropguard-ai --type space --space-sdk streamlit

# Push
git remote add hf https://huggingface.co/spaces/Shahim01/cropguard-ai
git push hf main
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CHECKPOINT` | `./checkpoints/best_cropguard_phase4.pt` | Path to .pt weights |
| `STREAMLIT_SERVER_PORT` | `8501` | Streamlit port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |

---

## Performance in Production

| Metric | Value |
|--------|-------|
| Model load time (cold start) | ~8–12 seconds (CPU) |
| Single inference | ~15–30 ms (CPU), ~5–10 ms (GPU) |
| E2E (upload → result) | ~2–5 seconds |
| Max file size | 10 MB (enforced in app) |
| Concurrent users | Up to 5 (free tier) |

---

## Reproducing Results from Scratch

```bash
# 1. Download PlantVillage dataset
kaggle datasets download abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/raw/

# 2. Run EDA
jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb

# 3. Train MLP baseline
jupyter nbconvert --to notebook --execute notebooks/02_MLP_Baseline.ipynb

# 4. Train ResNet50 + CBAM (GPU recommended, ~90 min)
jupyter nbconvert --to notebook --execute notebooks/03_ResNet50_CNN.ipynb

# 5. CBAM attention study
jupyter nbconvert --to notebook --execute notebooks/04_CBAM_Attention.ipynb

# 6. VAE augmentation
jupyter nbconvert --to notebook --execute notebooks/05_VAE_Augmentation.ipynb

# 7. Grad-CAM visualization
jupyter nbconvert --to notebook --execute notebooks/06_GradCAM.ipynb

# 8. Final evaluation (Phase 5)
jupyter nbconvert --to notebook --execute notebooks/07_Evaluation.ipynb

# 9. Launch dashboard
streamlit run app/Home.py
```

Total compute: ~15 GPU-hours (predominantly Phase 4 HPO).

---

## Troubleshooting

**Model file too large for hosting:**
```bash
# Quantize to INT8 (reduces size ~4×, ~1–2% accuracy drop)
python -c "
import torch
from src.model import CropGuardCNN
model = CropGuardCNN()
model.load_state_dict(torch.load('checkpoints/best_cropguard_phase4.pt')['model_state'])
quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
torch.save(quantized.state_dict(), 'checkpoints/cropguard_quantized.pt')
"
```

**CUDA out of memory:**
Reduce batch size in `src/utils.py`:
```python
BATCH_SIZE = 16  # default 32
```

**Streamlit app crashes on upload:**
Check file type (must be JPEG or PNG) and size (< 10 MB).
The app validates input before inference — a stack trace indicates a
preprocessing bug. File a GitHub issue with the error message.
