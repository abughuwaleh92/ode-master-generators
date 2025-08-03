# scripts/auto_backup.sh
#!/bin/bash
# Automated backup system for datasets and models

# Benefits:
# - Prevents data loss
# - Version control for datasets
# - Compressed storage
# - Cloud sync support

BACKUP_DIR="/backup/ode_project"
RETENTION_DAYS=30

# Create timestamped backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/backup_$TIMESTAMP"
mkdir -p "$BACKUP_PATH"

# Backup datasets
echo "Backing up datasets..."
tar -czf "$BACKUP_PATH/datasets.tar.gz" *.jsonl

# Backup models
echo "Backing up models..."
tar -czf "$BACKUP_PATH/models.tar.gz" *.pth *.pkl

# Backup configurations
cp -r config.yaml generators/ "$BACKUP_PATH/"

# Create backup manifest
cat > "$BACKUP_PATH/manifest.txt" << EOF
Backup Date: $(date)
Datasets: $(ls -1 *.jsonl | wc -l) files
Models: $(ls -1 *.pth *.pkl 2>/dev/null | wc -l) files
Total Size: $(du -sh "$BACKUP_PATH" | cut -f1)
EOF

# Clean old backups
find "$BACKUP_DIR" -name "backup_*" -mtime +$RETENTION_DAYS -exec rm -rf {} +

# Optional: Sync to cloud
# aws s3 sync "$BACKUP_PATH" "s3://your-bucket/ode-backups/$TIMESTAMP/"

echo "Backup complete: $BACKUP_PATH"