# Dockerfile - Working version for Railway deployment
# This is a unified Dockerfile that can build both API and GUI
# Use --build-arg SERVICE=api or SERVICE=gui to select

ARG SERVICE=api

# ============================================
# API Build
# ============================================
FROM python:3.11-slim AS api

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create necessary directories
RUN mkdir -p data models ml_data logs

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PORT=8080
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run command
CMD ["python", "scripts/production_server.py"]

# ============================================
# GUI Build
# ============================================
FROM node:18-alpine AS gui-builder

WORKDIR /app

# Copy package files
COPY ode_gui_bundle/package*.json ./

# Install dependencies
RUN npm ci

# Copy GUI source
COPY ode_gui_bundle/ .

# Build the application
RUN npm run build

# ============================================
# GUI Runtime
# ============================================
FROM nginx:alpine AS gui

# Copy built files from builder
COPY --from=gui-builder /app/dist /usr/share/nginx/html

# Create nginx configuration using echo commands (Docker-safe)
RUN echo 'server {' > /etc/nginx/conf.d/default.conf && \
    echo '    listen 80;' >> /etc/nginx/conf.d/default.conf && \
    echo '    server_name localhost;' >> /etc/nginx/conf.d/default.conf && \
    echo '    root /usr/share/nginx/html;' >> /etc/nginx/conf.d/default.conf && \
    echo '    index index.html;' >> /etc/nginx/conf.d/default.conf && \
    echo '' >> /etc/nginx/conf.d/default.conf && \
    echo '    gzip on;' >> /etc/nginx/conf.d/default.conf && \
    echo '    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json;' >> /etc/nginx/conf.d/default.conf && \
    echo '' >> /etc/nginx/conf.d/default.conf && \
    echo '    location /assets/ {' >> /etc/nginx/conf.d/default.conf && \
    echo '        expires 1y;' >> /etc/nginx/conf.d/default.conf && \
    echo '        add_header Cache-Control "public, immutable";' >> /etc/nginx/conf.d/default.conf && \
    echo '    }' >> /etc/nginx/conf.d/default.conf && \
    echo '' >> /etc/nginx/conf.d/default.conf && \
    echo '    location /config.js {' >> /etc/nginx/conf.d/default.conf && \
    echo '        add_header Cache-Control "no-store, no-cache, must-revalidate";' >> /etc/nginx/conf.d/default.conf && \
    echo '    }' >> /etc/nginx/conf.d/default.conf && \
    echo '' >> /etc/nginx/conf.d/default.conf && \
    echo '    location / {' >> /etc/nginx/conf.d/default.conf && \
    echo '        try_files $uri $uri/ /index.html;' >> /etc/nginx/conf.d/default.conf && \
    echo '    }' >> /etc/nginx/conf.d/default.conf && \
    echo '}' >> /etc/nginx/conf.d/default.conf

# Create entrypoint script using echo commands
RUN echo '#!/bin/sh' > /docker-entrypoint.sh && \
    echo 'cat > /usr/share/nginx/html/config.js << EOJS' >> /docker-entrypoint.sh && \
    echo 'window.ODE_CONFIG = {' >> /docker-entrypoint.sh && \
    echo '  API_BASE: "'${API_BASE:-}'",' >> /docker-entrypoint.sh && \
    echo '  API_KEY: "'${API_KEY:-}'",' >> /docker-entrypoint.sh && \
    echo '  WS: '${ENABLE_WEBSOCKET:-true} >> /docker-entrypoint.sh && \
    echo '};' >> /docker-entrypoint.sh && \
    echo 'EOJS' >> /docker-entrypoint.sh && \
    echo 'nginx -g "daemon off;"' >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh

EXPOSE 80

ENTRYPOINT ["/docker-entrypoint.sh"]

# ============================================
# Final stage selection based on SERVICE arg
# ============================================
FROM ${SERVICE} AS final
