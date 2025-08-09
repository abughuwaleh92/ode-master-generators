#!/bin/bash
# deploy.sh - Deploy to Railway

set -e

echo "🚀 Deploying ODE Master Generators to Railway"

# Function to check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check if Railway CLI is installed
    if ! command -v railway &> /dev/null; then
        echo "❌ Railway CLI not found. Installing..."
        npm install -g @railway/cli
    fi
    
    # Check if logged in to Railway
    if ! railway whoami &> /dev/null; then
        echo "📝 Please login to Railway:"
        railway login
    fi
}

# Function to setup Railway project
setup_railway_project() {
    echo "Setting up Railway project..."
    
    # Initialize Railway project if not exists
    if [ ! -f ".railway" ]; then
        railway init
    fi
}

# Function to deploy API service
deploy_api() {
    echo "📦 Deploying API service..."
    
    # Create API service if not exists
    railway service create ode-api || true
    
    # Set API environment variables
    railway variables set \
        ENVIRONMENT=production \
        PORT=8080 \
        ENABLE_WEBSOCKET=true \
        PUBLIC_READ=false \
        API_KEYS="${API_KEYS:-$(openssl rand -hex 32)}" \
        --service ode-api
    
    # Deploy API
    railway up --service ode-api --dockerfile Dockerfile.api
    
    # Get API URL
    API_URL=$(railway status --service ode-api --json | jq -r '.url')
    echo "✅ API deployed at: $API_URL"
}

# Function to deploy GUI service
deploy_gui() {
    echo "📦 Deploying GUI service..."
    
    # Create GUI service if not exists
    railway service create ode-gui || true
    
    # Set GUI environment variables
    railway variables set \
        API_BASE="$API_URL" \
        ENABLE_WEBSOCKET=true \
        --service ode-gui
    
    # Deploy GUI
    railway up --service ode-gui --dockerfile Dockerfile.gui
    
    # Get GUI URL
    GUI_URL=$(railway status --service ode-gui --json | jq -r '.url')
    echo "✅ GUI deployed at: $GUI_URL"
}

# Function to setup Redis (if needed)
setup_redis() {
    echo "📦 Setting up Redis..."
    
    # Add Redis plugin
    railway plugin create redis || true
    
    # Get Redis URL
    REDIS_URL=$(railway variables get REDIS_URL)
    
    # Update API with Redis URL
    railway variables set REDIS_URL="$REDIS_URL" --service ode-api
}

# Main deployment flow
main() {
    check_prerequisites
    setup_railway_project
    
    # Optional: Setup Redis
    read -p "Do you want to add Redis caching? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_redis
    fi
    
    # Deploy services
    deploy_api
    deploy_gui
    
    echo "
    ✨ Deployment Complete! ✨
    
    API URL: $API_URL
    GUI URL: $GUI_URL
    
    Next steps:
    1. Update ALLOWED_ORIGINS for API: railway variables set ALLOWED_ORIGINS=$GUI_URL --service ode-api
    2. Generate API keys: railway variables set API_KEYS=your-secure-keys --service ode-api
    3. Monitor logs: railway logs --service ode-api
    "
}

# Run main function
main
