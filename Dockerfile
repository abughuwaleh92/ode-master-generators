# Dockerfile - Unified Dockerfile for both API and GUI
# Uses build arguments to determine which service to build

ARG SERVICE_TYPE=api

# ============================================
# API Build Stage
# ============================================
FROM python:3.11-slim AS api-build

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

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# ============================================
# GUI Build Stage
# ============================================
FROM node:18-alpine AS gui-build

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
# GUI Runtime Stage
# ============================================
FROM nginx:alpine AS gui-runtime

# Copy built files from gui-build stage
COPY --from=gui-build /app/dist /usr/share/nginx/html

# Create nginx config
RUN cat > /etc/nginx/conf.d/default.conf << 'EOF'
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    gzip on;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json;

    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location /config.js {
        add_header Cache-Control "no-store, no-cache, must-revalidate";
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
EOF

# Create runtime config script
RUN cat > /docker-entrypoint.sh << 'EOF'
#!/bin/sh
cat > /usr/share/nginx/html/config.js << EOJS
window.ODE_CONFIG = {
  API_BASE: '${API_BASE:-}',
  API_KEY: '${API_KEY:-}',
  WS: ${ENABLE_WEBSOCKET:-true}
};
EOJS
nginx -g 'daemon off;'
EOF

RUN chmod +x /docker-entrypoint.sh

# ============================================
# Final Stage Selection
# ============================================
FROM ${SERVICE_TYPE}-${SERVICE_TYPE == 'api' ? 'build' : 'runtime'} AS final

# API-specific configuration
FROM api-build AS api-final
ENV PORT=8080
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1
EXPOSE ${PORT}
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1
CMD ["python", "scripts/production_server.py"]

# GUI-specific configuration
FROM gui-runtime AS gui-final
EXPOSE 80
ENTRYPOINT ["/docker-entrypoint.sh"]

# Select final image based on SERVICE_TYPE
FROM ${SERVICE_TYPE}-final
