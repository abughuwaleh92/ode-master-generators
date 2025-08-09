# ---------- Stage 1: Build the React GUI ----------
FROM node:20-alpine AS frontend
WORKDIR /ui

# Only copy the GUI sources to leverage caching
COPY ode_gui_bundle/package.json ode_gui_bundle/package-lock.json* ./ 
# If you use pnpm/yarn, copy the relevant lockfile instead and adjust install cmd
RUN npm ci || npm i

# Copy the rest of the GUI
COPY ode_gui_bundle/ ./

# Build the SPA (outputs to /ui/dist)
RUN npm run build

# ---------- Stage 2: Python backend + static GUI ----------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    RAILWAY_ENVIRONMENT=production \
    GUI_BUNDLE_DIR=/app/ode_gui_bundle/dist

WORKDIR /app

# System deps you might need (e.g., for scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App source
COPY . /app

# Bring the built GUI into the expected path
# This assumes your server serves from $GUI_BUNDLE_DIR
RUN mkdir -p /app/ode_gui_bundle/dist
COPY --from=frontend /ui/dist /app/ode_gui_bundle/dist

EXPOSE 8080
CMD ["python", "-m", "scripts.production_server"]
