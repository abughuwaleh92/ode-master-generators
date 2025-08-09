##--------------------------------------------------------------------
## Dockerfile for ODE Master Generators: API + GUI (Railway deploy)
##
## This multi‑stage Docker build installs and compiles the React
## front‑end located in `ode_gui_bundle` and installs all Python
## dependencies required to run the FastAPI production server defined
## in `scripts/production_server.py`.  The built GUI is copied into
## the final image under `ode_gui_bundle/dist`, and the API knows
## to serve static assets from this directory via the `GUI_BUNDLE_DIR`
## environment variable.  Railway automatically exposes the container
## on port 8080; you can override this by setting the `PORT` env var.

### -------------------------------------------------------------------
### 1. Build stage: compile the React GUI using Node
###
# Use an Alpine‑based Node image to build the SPA efficiently.
FROM node:18-alpine AS gui-builder

## Set the working directory for building the GUI
WORKDIR /app/gui

## Copy package manifest files first to leverage Docker layer caching
COPY ode_gui_bundle/package*.json ./

## Install front‑end dependencies.  `npm ci` uses the lock file if
## present and ensures repeatable builds.  We pass `--legacy-peer-deps`
## to accommodate packages that may have peer dependency mismatches.
RUN npm ci --legacy-peer-deps

## Copy the rest of the GUI sources into the build context
COPY ode_gui_bundle/ ./

## Build the production SPA.  This generates a `dist` directory
## containing static HTML/JS/CSS assets under `/app/gui/dist`.
RUN npm run build

### -------------------------------------------------------------------
### 2. Runtime stage: install Python dependencies and serve the API
###
# Use a slim Python base image.  Python 3.11 is specified in
# `requirements.txt` and aligns with Railway’s default runtime.
FROM python:3.11-slim

## Prevent Python from writing `.pyc` files and enable unbuffered
## logging for consistent output in containers.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

## Install system packages necessary for compiling native Python
## extensions and for the built in healthcheck.  `build-essential`
## provides gcc/g++ and related tools; `curl` is used in the health
## check; `git` is included for optional dependencies that fetch from
## Git repositories.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

## Set work directory for application code
WORKDIR /app

## Copy dependency manifest first for caching and install the Python
## packages.  Upgrading pip ensures features like PEP 517 builds.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

## Copy the remainder of the repository into the container.  All
## Python modules, scripts and static files live relative to `/app`.
COPY . .

## Copy the compiled front‑end assets from the build stage into the
## runtime image.  The assets live at `/app/gui/dist` in the build
## stage; we place them into `ode_gui_bundle/dist` so that
## `production_server.py` can automatically locate them.  If you
## choose to mount a different GUI directory, set the `GUI_BUNDLE_DIR`
## environment variable accordingly.
COPY --from=gui-builder /app/gui/dist ./ode_gui_bundle/dist

## (Optional) create a non‑root user for security and ensure
## ownership of application files.  Railway runs images as root by
## default, but dropping privileges is a best practice.
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app

## Switch to the non‑root user
USER appuser

## Expose the port used by the FastAPI application.  Railway binds
## automatically to the port specified by the `PORT` environment
## variable (default 8080).  Exposing it here helps documentation and
## local testing.
EXPOSE 8080

## Default environment variables.  These can be overridden by
## Railway configuration or local `.env` files.  `GUI_BUNDLE_DIR`
## points to the directory containing `index.html` for the SPA.
ENV PORT=8080 \
    ENVIRONMENT=production \
    GUI_BUNDLE_DIR=ode_gui_bundle/dist

## Provide a simple healthcheck.  Railway monitors this endpoint to
## determine container health.  FastAPI defines `/health` in
## `production_server.py`.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

## The default command launches the production FastAPI server.  The
## `production_server.py` script uses Uvicorn internally when run
## directly.  It also sets up CORS, API key middleware and mounts
## the GUI bundle after API routes.
CMD ["python", "scripts/production_server.py"]
