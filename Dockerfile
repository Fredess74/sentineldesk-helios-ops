# Stage 1: Build clean React + Vite Web UI
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python 3.11 FastAPI + ML/FAISS Runtime
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY evaluate.py ./evaluate.py
COPY artifacts/holdout_metrics.json artifacts/kb_audit_summary.json artifacts/index_meta.json ./artifacts/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Build zero-leakage 80/20 split, FAISS KB index, and calibrated local ML heads at image build time
RUN python backend/data_prep.py && \
    python backend/indexing.py && \
    python backend/train_ml.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
