# ─────────────────────────────────────────────────────────────────────────────
#  CropGuard AI — Dockerfile
#  Phase 6 Deployment  |  AI335L Deep Learning Lab  |  Spring 2026
# ─────────────────────────────────────────────────────────────────────────────
#
#  Build:   docker build -t cropguard-ai .
#  Run:     docker run -p 8501:8501 --env-file .env cropguard-ai
#  Health:  curl http://localhost:8501/healthz
#
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libgomp1 \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash cropguard
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/        ./src/
COPY app/        ./app/
COPY checkpoints/ ./checkpoints/

# Streamlit config
RUN mkdir -p /app/.streamlit
COPY .streamlit/config.toml /app/.streamlit/config.toml 2>/dev/null || \
    echo '[server]\nheadless = true\nport = 8501\nenableCORS = false\n[browser]\ngatherUsageStats = false' \
    > /app/.streamlit/config.toml

# Set ownership
RUN chown -R cropguard:cropguard /app
USER cropguard

# Environment variables (override via --env-file or -e flags)
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV MODEL_CHECKPOINT=/app/checkpoints/cropguard_resnet50_cbam_phase3.pt
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/healthz || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "app/Home.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
