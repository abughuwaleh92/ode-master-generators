# Use a slim Python base
FROM python:3.11-slim

WORKDIR /app

# System dependencies (build + runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install minimal deps for API (no GPU ML in this image)
RUN pip install --no-cache-dir -r railway/requirements-railway.txt

# Expose PORT (Railway sets PORT env var)
ENV PORT=8000
CMD [ "uvicorn", "scripts.production_server:app", "--host", "0.0.0.0", "--port", "8000" ]
