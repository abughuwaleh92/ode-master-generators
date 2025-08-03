# scripts/deploy.sh
#!/bin/bash
# Production deployment script

# Benefits:
# - Zero-downtime deployment
# - Automatic rollback on failure
# - Health checks
# - Load balancing

set -e

# Configuration
DEPLOY_ENV=${1:-"production"}
APP_NAME="ode-generator"
DEPLOY_DIR="/opt/$APP_NAME"
BACKUP_DIR="/opt/backups/$APP_NAME"
NGINX_CONFIG="/etc/nginx/sites-available/$APP_NAME"

echo "Deploying ODE Generator to $DEPLOY_ENV"

# 1. Pre-deployment checks
echo "Running pre-deployment checks..."

# Check disk space
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    echo "ERROR: Disk usage too high: $DISK_USAGE%"
    exit 1
fi

# Check if services are running
if ! systemctl is-active --quiet nginx; then
    echo "ERROR: Nginx is not running"
    exit 1
fi

# 2. Backup current deployment
echo "Creating backup..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/backup_$TIMESTAMP"
mkdir -p "$BACKUP_PATH"

if [ -d "$DEPLOY_DIR" ]; then
    cp -r "$DEPLOY_DIR" "$BACKUP_PATH/"
    echo "Backup created at $BACKUP_PATH"
fi

# 3. Update code
echo "Updating code..."
cd "$DEPLOY_DIR"

# Git pull (or copy from CI/CD artifacts)
git fetch origin
git checkout $DEPLOY_ENV
git pull origin $DEPLOY_ENV

# 4. Install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt --upgrade

# 5. Run migrations/updates
echo "Running database migrations..."
python scripts/migrate_db.py

# 6. Run tests
echo "Running deployment tests..."
python -m pytest tests/deployment_tests.py -v

# 7. Build static assets
echo "Building static assets..."
python scripts/build_assets.py

# 8. Update configuration
echo "Updating configuration..."
cp "config/config.$DEPLOY_ENV.yaml" "config.yaml"

# 9. Reload services with zero downtime
echo "Reloading services..."

# Start new workers
for i in {1..4}; do
    systemctl start "ode-worker-new-$i"
done

# Wait for new workers to be ready
sleep 10

# Health check new workers
for i in {1..4}; do
    if ! curl -f "http://localhost:800$i/health" > /dev/null 2>&1; then
        echo "ERROR: New worker $i failed health check"
        # Rollback
        for j in {1..4}; do
            systemctl stop "ode-worker-new-$j" || true
        done
        exit 1
    fi
done

# Switch nginx to new workers
cp "$NGINX_CONFIG.new" "$NGINX_CONFIG"
nginx -t && nginx -s reload

# Stop old workers
for i in {1..4}; do
    systemctl stop "ode-worker-old-$i" || true
done

# Rename new workers to current
for i in {1..4}; do
    systemctl stop "ode-worker-new-$i"
    mv "/etc/systemd/system/ode-worker-new-$i.service" "/etc/systemd/system/ode-worker-$i.service"
    systemctl daemon-reload
    systemctl start "ode-worker-$i"
done

# 10. Post-deployment checks
echo "Running post-deployment checks..."

# API health check
if ! curl -f "https://api.ode-generator.com/health" > /dev/null 2>&1; then
    echo "ERROR: API health check failed"
    # Rollback
    ./scripts/rollback.sh "$BACKUP_PATH"
    exit 1
fi

# Generate test ODE
TEST_RESPONSE=$(curl -s -X POST "https://api.ode-generator.com/api/v1/generate" \
    -H "X-API-Key: $DEPLOY_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"generator": "L1", "function": "sine", "count": 1}')

if [ -z "$TEST_RESPONSE" ]; then
    echo "ERROR: Test generation failed"
    ./scripts/rollback.sh "$BACKUP_PATH"
    exit 1
fi

# 11. Clean up old backups
echo "Cleaning old backups..."
find "$BACKUP_DIR" -name "backup_*" -mtime +7 -exec rm -rf {} +

# 12. Send notification
echo "Sending deployment notification..."
curl -X POST "$SLACK_WEBHOOK" \
    -H "Content-Type: application/json" \
    -d "{
        \"text\": \"ODE Generator deployed successfully to $DEPLOY_ENV\",
        \"attachments\": [{
            \"color\": \"good\",
            \"fields\": [
                {\"title\": \"Environment\", \"value\": \"$DEPLOY_ENV\", \"short\": true},
                {\"title\": \"Version\", \"value\": \"$(git rev-parse --short HEAD)\", \"short\": true},
                {\"title\": \"Deployed by\", \"value\": \"$USER\", \"short\": true},
                {\"title\": \"Timestamp\", \"value\": \"$(date)\", \"short\": true}
            ]
        }]
    }"

echo "Deployment complete!"