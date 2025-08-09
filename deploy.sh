#!/bin/bash
# Railway deployment troubleshooting script

set -e

echo "🔧 Railway Deployment Troubleshooting"
echo "======================================"

# Check current directory
if [ ! -f "scripts/production_server.py" ]; then
    echo "❌ Error: Run this script from the project root directory"
    exit 1
fi

echo "📁 Project structure looks good"

# Option 1: Try minimal nixpacks
echo ""
echo "🎯 OPTION 1: Minimal Nixpacks Configuration"
echo "--------------------------------------------"

cat > nixpacks.toml << 'EOF'
[variables]
NIXPACKS_PYTHON_VERSION = "3.11"

[nixpkgs]
packages = ["python311", "python311Packages.pip"]

[phases.install]
cmds = ["pip install -r requirements.txt"]

[phases.build]
cmds = ["cd ode_gui_bundle && npm ci && npm run build"]

[start]
cmd = "python scripts/production_server.py"
EOF

cat > railway.toml << 'EOF'
[build]
builder = "nixpacks"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "always"

[deploy.env]
ENVIRONMENT = "production"
ENABLE_WEBSOCKET = "true"
PUBLIC_READ = "true"
PORT = "8080"
GUI_BUNDLE_DIR = "ode_gui_bundle/dist"
EOF

echo "✅ Created minimal nixpacks.toml and railway.toml"

# Option 2: Create Dockerfile alternative
echo ""
echo "🐳 OPTION 2: Dockerfile Alternative (if nixpacks fails)"
echo "------------------------------------------------------"

cat > Dockerfile << 'EOF'
# Multi-stage build
FROM node:18-alpine AS gui-builder
WORKDIR /app/gui
COPY ode_gui_bundle/package*.json ./
RUN npm ci
COPY ode_gui_bundle/ ./
RUN npm run build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=gui-builder /app/gui/dist ./ode_gui_bundle/dist
RUN adduser --disabled-password appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8080
CMD ["python", "scripts/production_server.py"]
EOF

cat > railway-docker.toml << 'EOF'
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "always"

[deploy.env]
ENVIRONMENT = "production"
ENABLE_WEBSOCKET = "true"
PUBLIC_READ = "true"
PORT = "8080"
GUI_BUNDLE_DIR = "ode_gui_bundle/dist"
EOF

echo "✅ Created Dockerfile and railway-docker.toml"

# Create minimal requirements.txt
echo ""
echo "📦 Creating minimal requirements.txt"
echo "------------------------------------"

cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
numpy==1.24.3
sympy==1.12
prometheus-client==0.19.0
PyYAML==6.0.1
Jinja2==3.1.2
EOF

echo "✅ Created minimal requirements.txt"

# Check package.json
echo ""
echo "📋 Checking GUI configuration"
echo "-----------------------------"

if [ -f "ode_gui_bundle/package.json" ]; then
    echo "✅ package.json found"
    
    # Check if build script exists
    if grep -q '"build"' ode_gui_bundle/package.json; then
        echo "✅ Build script found in package.json"
    else
        echo "⚠️  Warning: No build script found in package.json"
        echo "   Add this to package.json scripts section:"
        echo '   "build": "vite build"'
    fi
else
    echo "❌ ode_gui_bundle/package.json not found"
    echo "   Creating basic package.json..."
    
    mkdir -p ode_gui_bundle
    cat > ode_gui_bundle/package.json << 'EOF'
{
  "name": "ode-gui",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "build": "echo 'No GUI build needed' && mkdir -p dist && echo '<h1>GUI Placeholder</h1>' > dist/index.html"
  }
}
EOF
    echo "✅ Created basic package.json with placeholder build"
fi

echo ""
echo "🚀 Deployment Options"
echo "====================="
echo ""
echo "OPTION A: Try Nixpacks first"
echo "-----------------------------"
echo "railway up"
echo ""
echo "OPTION B: If nixpacks fails, try Docker"
echo "---------------------------------------"
echo "mv railway.toml railway-nixpacks.toml"
echo "mv railway-docker.toml railway.toml"
echo "railway up"
echo ""
echo "OPTION C: Deploy without GUI first"
echo "----------------------------------"
echo "# Remove GUI build from nixpacks.toml phases.build"
echo "# Set GUI_BUNDLE_DIR to empty or comment out"
echo "railway up"

echo ""
echo "🔍 Debugging Commands"
echo "====================="
echo "railway logs          # View deployment logs"
echo "railway status        # Check deployment status"
echo "railway variables     # Check environment variables"
echo "railway shell         # Access deployed container"

echo ""
echo "✅ Troubleshooting setup complete!"
echo "   Try the deployment options above in order."
