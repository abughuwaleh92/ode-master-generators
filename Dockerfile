FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy code
COPY . /app

# If you have a prebuilt GUI at ode_gui_bundle/dist, the server will serve it.
# If you don't, this env can be omitted or left; the server logs a warning and continues.
ENV GUI_BUNDLE_DIR="ode_gui_bundle/dist" \
    PORT=8080 \
    RAILWAY_ENVIRONMENT=production \
    ENABLE_WEBSOCKET=true \
    PUBLIC_READ=true \
    API_KEYS="railway-key,dev-key" \
    ALLOWED_ORIGINS="https://ode-gui.up.railway.app,http://localhost:3000"

EXPOSE 8080

CMD ["uvicorn", "scripts.production_server:app", "--host", "0.0.0.0", "--port", "8080", "--lifespan", "on", "--workers", "1"]
