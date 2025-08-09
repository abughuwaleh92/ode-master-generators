# =========================
# 1) GUI builder (optional)
#    Build only the SPA bundle with Node
# =========================
FROM node:20-alpine AS gui-builder

WORKDIR /gui

# Install a few build basics to keep node-gyp happy if needed
RUN apk add --no-cache python3 make g++ git

# Copy only package files first for better layer caching
COPY ode_gui_bundle/package*.json ./ 

# Install deps (ci when lockfile exists, else install)
RUN if [ -f package-lock.json ]; then npm ci; else npm i; fi

# Now copy the rest of the GUI
COPY ode_gui_bundle/ ./

# Build the Vite/React app
# (If your build needs envs like VITE_API_URL, set them as --build-arg and pass to ENV here)
RUN npm run build


# =========================
# 2) Python deps builder (cacheable)
#    Create a venv with all Python deps
# =========================
FROM python:3.11-slim AS api-deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal system deps for building wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage docker layer cache
COPY requirements.txt .

# Use a self-contained venv to simplify runtime stage
RUN python -m venv /opt/venv \
 && . /opt/venv/bin/activate \
 && pip install --upgrade pip \
 && pip install -r requirements.txt

# =========================
# 3) Runtime
#    Copy app + venv, optionally copy built GUI
# =========================
FROM python:3.11-slim AS runtime

# --- Runtime envs (tweak at deploy time in Railway Variables) ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    RAILWAY_ENVIRONMENT=production \
    ENABLE_WEBSOCKET=true \
    PUBLIC_READ=true \
    API_KEYS="railway-key,dev-key" \
    ALLOWED_ORIGINS="https://ode-gui.up.railway.app,http://localhost:3000"

# Optional: set this to true at build to copy in the GUI bundle
ARG BUILD_GUI=false
ENV BUILD_GUI=${BUILD_GUI}

# Put venv on PATH
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install system deps needed at runtime (keep minimal)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy the prepared Python env
COPY --from=api-deps /opt/venv /opt/venv

# Copy your API code
COPY . /app

# If BUILD_GUI=true, copy the GUI bundle produced in stage 1
# The API tolerates a missing GUI (it checks for index.html), so this is safe either way.
# We still always run the COPY; the gui-builder stage is always built and will have /gui/dist.
# If your repo doesn't include ode_gui_bundle/, keep BUILD_GUI=false to avoid build failures.
COPY --from=gui-builder /gui/dist /app/ode_gui_bundle/dist

# Hint the server where the GUI bundle is (it also auto-detects)
ENV GUI_BUNDLE_DIR="/app/ode_gui_bundle/dist"

EXPOSE 8080

# Healthcheck (optional). Uncomment if you want Docker-level checks.
# RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
# HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
#   CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# Use Uvicorn to run the FastAPI app inside the container
# replace the previous CMD with this:
CMD ["python", "-m", "scripts.production_server"]
