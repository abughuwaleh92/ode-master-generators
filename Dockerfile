FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

ENV PORT=8080 RAILWAY_ENVIRONMENT=production

EXPOSE 8080

CMD ["sh", "-c", "python -m uvicorn scripts.production_server:app --host 0.0.0.0 --port $PORT"]
