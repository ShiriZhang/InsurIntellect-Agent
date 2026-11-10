# ---------- base stage: heavy wheels installed once (chromadb / scipy / numpy) ----------
# Splitting the heavy DB layer from the light deps so changes to small packages
# don't trigger a ~500s chromadb rebuild.
FROM python:3.11-slim-bookworm AS heavy-deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_NO_COMPILE=1
WORKDIR /app

COPY <<'EOF' /tmp/requirements-heavy.txt
chromadb>=0.5.0
langchain-chroma>=0.1.0
numpy>=1.24.0
PyMuPDF>=1.28.0
jieba>=0.42.0
rank_bm25>=0.2.2
EOF
RUN pip install --no-cache-dir --no-compile -r /tmp/requirements-heavy.txt

# ---------- light deps stage: smaller packages, change more often ----------
FROM heavy-deps AS deps

WORKDIR /app

COPY requirements-docker.txt .
# Strip the heavy packages already installed above to avoid re-resolving them.
RUN grep -vE '^(chromadb|langchain-chroma|numpy|PyMuPDF|jieba|rank_bm25)' requirements-docker.txt > /tmp/requirements-light.txt \
    && pip install --no-cache-dir --no-compile -r /tmp/requirements-light.txt

# ---------- runtime stage: small app layer on top of cached deps ----------
FROM deps AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8766

WORKDIR /app

# site-packages and python binary already inherited from the deps stage chain.
COPY . .

EXPOSE 8766

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; port=os.getenv('PORT','8766'); urllib.request.urlopen(f'http://127.0.0.1:{port}/api/v1/health/live', timeout=5).read()"

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8766}"]

# Optional complete runtime. Build with --target full and mount prepared cache.
FROM runtime AS full
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "torch>=2.2.0" --index-url ${TORCH_INDEX_URL}
COPY requirements-models.txt .
RUN pip install --no-cache-dir -r requirements-models.txt

# Keep the default build lightweight.
FROM runtime AS default
