#!/bin/bash

# Build GUI
echo "Building GUI..."
cd ode_gui_bundle
npm install
npm run build

# Ensure server can find GUI
export GUI_BUNDLE_DIR="$(pwd)/dist"

# Deploy to Railway
echo "Deploying to Railway..."
railway up

echo "Deployment complete!"
