FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

ENV RAILWAY_ENVIRONMENT=production

EXPOSE 8080

# Optional: container-level healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD wget -qO- http://127.0.0.1:$PORT/health || exit 1

# Use shell form so $PORT expands (Railway sets PORT automatically)
CMD sh -c "python -m uvicorn scripts.production_server:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips='*'"
