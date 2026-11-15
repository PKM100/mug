# MUG medical QA app: FastAPI backend + built React UI in one image.
#   docker build -t mug .
#   docker run --rm -v mug-data:/data mug python ../scripts/download_index.py --data-dir /data   # once, ~20-30 min
#   docker run --rm -p 127.0.0.1:8000:8000 -v mug-data:/data -e OPENAI_API_KEY mug              # http://127.0.0.1:8000
FROM node:20-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    MUG_DATA_DIR=/data \
    MUG_AUDIT_DB=/data/audit.db
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt \
 && python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"
COPY backend/ backend/
COPY scripts/ scripts/
COPY --from=ui /ui/dist frontend/dist
RUN useradd --create-home --uid 1000 mug && mkdir -p /data && chown -R mug /data /opt/fastembed
USER mug
VOLUME /data
WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
