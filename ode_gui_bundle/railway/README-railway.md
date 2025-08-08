# Railway Deployment Notes

## One-service deployment (API only)
1) Ensure your repository includes `scripts/production_server.py`.
2) Commit the `gui/` folder to your repo root.
3) Commit `railway/Procfile`, `railway/requirements-railway.txt`, and `railway/Dockerfile`.
4) On Railway:
   - Create a new project from your GitHub repo.
   - Set service to "Dockerfile".
   - Set environment variables:
     - `API_KEYS` e.g. `my-dev-key`
     - `RAILWAY_ENVIRONMENT=production`
     - `ENABLE_WEBSOCKET=true`
     - (Optional) Attach a Redis plugin → this will set `REDIS_URL` automatically.
5) Deploy.

## ML-enabled build (larger image)
Use `railway/requirements-railway-ml.txt` in the Dockerfile to include PyTorch and Transformers, or create a second Railway service dedicated to ML endpoints.
