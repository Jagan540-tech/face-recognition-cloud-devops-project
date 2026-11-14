# syntax=docker/dockerfile:1
# Multi-stage build. Final image = python:3.12-slim + a venv with 4 runtime packages
# (flask, gunicorn, opencv-python-headless, numpy) + two small ONNX models. No compilers, no apt
# packages, no TensorFlow.

# ---- stage 1: fetch + verify the pretrained models (stdlib only; hashes pinned in the script)
FROM python:3.12-slim AS models
WORKDIR /build
COPY scripts/download_models.py .
RUN python download_models.py --dir /models

# ---- stage 2: install Python dependencies into a venv (binary wheels only -> no compiler needed)
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install --only-binary=:all: -r requirements.txt \
 && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + \
 && rm -rf /opt/venv/lib/python3.12/site-packages/pip/_vendor/*/tests 2>/dev/null || true

# ---- stage 3: runtime
FROM python:3.12-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FR_MODELS_DIR=/app/models \
    FR_DATA_DIR=/data \
    FR_PORT=5000 \
    FR_S3_BUCKET=jaganops-face-recognition-data-080705554676 \
    FR_S3_REGION=ap-south-1
# Registration data lives in Amazon S3 (FR_S3_BUCKET). It is restored at start-up and every new
# registration is uploaded. Credentials come from the EC2 instance role - NEVER put keys in this file.
# Running without S3 (local development): docker run -e FR_S3_BUCKET= ...

# Non-root user. /data is the ONLY writable location. It is TEMPORARY scratch space in the container
# (NOT a volume): S3 is the permanent copy and /data is rebuilt from it every time the app starts.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser \
 && mkdir -p /data /app/models \
 && chown appuser:appuser /data

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=models /models /app/models
RUN chmod 755 /app/models && chmod 644 /app/models/*.onnx
COPY app.py config.py gunicorn.conf.py ./
COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts

USER appuser
EXPOSE 5000

# Calls the real /health endpoint (stdlib only, so no curl is needed in the image).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os,sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('FR_PORT','5000'), timeout=4).status == 200 else 1)"]

# ONE worker on purpose (in-memory registration sessions + embedding store); see gunicorn.conf.py.
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:create_app()"]
