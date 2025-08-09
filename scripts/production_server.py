# scripts/production_server.py
"""
Production FastAPI server for ODE Master Generators System
Optimized for Railway deployment with full GUI integration support.
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import asyncio
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import logging
from contextlib import asynccontextmanager
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import sympy as sp
from fastapi import (
    FastAPI, HTTPException, Depends, Request, Response,
    WebSocket, WebSocketDisconnect
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis

# Ensure project root is importable (when run as `python -m scripts.production_server`)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -----------------------------
# Optional project imports
# -----------------------------
try:
    from pipeline.generator import ODEDatasetGenerator
    from verification.verifier import ODEVerifier
    from utils.config import ConfigManager
    from utils.features import FeatureExtractor
    from core.types import GeneratorType, VerificationMethod, ODEInstance
    from core.functions import AnalyticFunctionLibrary
    from core.symbols import SYMBOLS  # noqa: F401
    CORE_IMPORTS_AVAILABLE = True
except Exception as e:
    logging.warning(f"Core imports not available: {e}")
    CORE_IMPORTS_AVAILABLE = False

    # Minimal fallbacks to keep API responsive in demo mode
    class GeneratorType(Enum):
        LINEAR = "linear"
        NONLINEAR = "nonlinear"

    class VerificationMethod(Enum):
        SUBSTITUTION = "substitution"
        CHECKODESOL = "checkodesol"
        NUMERIC = "numeric"
        FAILED = "failed"
        PENDING = "pending"

# Optional ML stack
try:
    import torch
    from ml_pipeline.train_ode_generator import ODEGeneratorTrainer
    from ml_pipeline.models import ODEPatternNet, ODETransformer, ODEVAE  # noqa: F401
    from ml_pipeline.evaluation import ODEEvaluator  # noqa: F401
    from ml_pipeline.utils import prepare_ml_dataset, load_pretrained_model, generate_novel_odes
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False
    logging.warning("ML pipeline not available")

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Environment
# -----------------------------
PORT = int(os.getenv("PORT", "8080"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
# Comma-separated list of valid API keys, e.g. "dev-key,railway-key,abc123"
API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "dev-key,test-key,railway-key").split(",") if k.strip()]
ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "development")
ENABLE_WEBSOCKET = os.getenv("ENABLE_WEBSOCKET", "true").lower() == "true"

# -----------------------------
# Redis (with in-memory fallback)
# -----------------------------
try:
    if REDIS_URL.startswith(("redis://", "rediss://")):
        redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=5)
        redis_client.ping()
        REDIS_AVAILABLE = True
    else:
        raise ConnectionError("Invalid Redis URL")
except Exception as e:
    logger.warning(f"Redis not available: {e}. Using in-memory cache.")
    REDIS_AVAILABLE = False

    class InMemoryCache:
        def __init__(self):
            self.cache: Dict[str, str] = {}
            self.ttls: Dict[str, float] = {}

        def get(self, key: str) -> Optional[str]:
            if key in self.ttls and time.time() > self.ttls[key]:
                self.cache.pop(key, None)
                self.ttls.pop(key, None)
            return self.cache.get(key)

        def set(self, key: str, value: str, ex: Optional[int] = None):
            self.cache[key] = value
            if ex:
                self.ttls[key] = time.time() + ex

        def setex(self, key: str, ttl: int, value: str):
            self.set(key, value, ex=ttl)

        def delete(self, *keys: str):
            for k in keys:
                self.cache.pop(k, None)
                self.ttls.pop(k, None)

        def keys(self, pattern: str = "*"):
            import fnmatch
            if pattern == "*":
                return list(self.cache.keys())
            return [k for k in self.cache.keys() if fnmatch.fnmatch(k, pattern)]

        def incr(self, key: str) -> int:
            val = int(self.cache.get(key, "0")) + 1
            self.cache[key] = str(val)
            return val

        def expire(self, key: str, ttl: int):
            if key in self.cache:
                self.ttls[key] = time.time() + ttl

    redis_client = InMemoryCache()

# -----------------------------
# Prometheus metrics
# -----------------------------
ode_generation_counter = Counter("ode_generation_total", "Total ODEs generated", ["generator", "function"])
verification_counter = Counter("ode_verification_total", "Total verifications", ["method", "result"])
generation_time_histogram = Histogram("ode_generation_duration_seconds", "ODE generation time")
active_jobs_gauge = Gauge("active_jobs", "Number of active jobs")
api_request_counter = Counter("api_requests_total", "Total API requests", ["endpoint", "method", "status"])
api_request_duration = Histogram("api_request_duration_seconds", "API request duration", ["endpoint"])
websocket_connections = Gauge("websocket_connections", "Active WebSocket connections")
dataset_size_gauge = Gauge("dataset_size_bytes", "Total size of datasets")

# -----------------------------
# Pydantic base
# -----------------------------
class APIModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow fields named "model_*" etc.

# -----------------------------
# Schemas
# -----------------------------
class ODEGenerationRequest(APIModel):
    generator: str = Field(..., description="Generator name (e.g., L1, N1)")
    function: str = Field(..., description="Function name (e.g., sine, exponential)")
    parameters: Optional[Dict[str, float]] = None
    count: int = Field(1, ge=1, le=100)
    verify: bool = True
    stream: bool = False

class BatchGenerationRequest(APIModel):
    generators: List[str]
    functions: List[str]
    samples_per_combination: int = Field(5, ge=1, le=50)
    parameter_ranges: Optional[Dict[str, List[float]]] = None
    verify: bool = True
    save_dataset: bool = True
    dataset_name: Optional[str] = None

class ODEVerificationRequest(APIModel):
    ode: str
    solution: str
    method: str = Field("substitution", description="Verification method")
    timeout: int = Field(30, ge=1, le=300)

class MLTrainingRequest(APIModel):
    dataset: str
    model_type: str = Field(..., description="Model type: pattern_net, transformer, vae")
    epochs: int = Field(50, ge=1, le=1000)
    batch_size: int = Field(32, ge=8, le=256)
    learning_rate: float = Field(0.001, ge=0.00001, le=0.1)
    early_stopping: bool = True
    validation_split: float = Field(0.2, ge=0.1, le=0.5)

class MLGenerationRequest(APIModel):
    model_path: str
    n_samples: int = Field(10, ge=1, le=1000)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    generators: Optional[List[str]] = None
    functions: Optional[List[str]] = None

class JobStatus(APIModel):
    job_id: str
    status: str
    progress: float
    results: Optional[Any] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    eta: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CreateDatasetBody(APIModel):
    odes: List[Dict[str, Any]]
    name: Optional[str] = None

# -----------------------------
# Job manager
# -----------------------------
class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.job_queues: Dict[str, deque] = defaultdict(deque)

    async def create_job(self, job_type: str, params: Dict[str, Any], priority: int = 5) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        job_data = {
            "id": job_id,
            "type": job_type,
            "params": params,
            "status": "queued",
            "priority": priority,
            "progress": 0.0,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "results": None,
            "error": None,
            "metadata": {},
            "retries": 0,
        }
        self.jobs[job_id] = job_data
        self.job_queues[job_type].append(job_id)
        active_jobs_gauge.inc()
        if REDIS_AVAILABLE:
            redis_client.setex(f"job:{job_id}", 3600, json.dumps(job_data))
        return job_id

    async def update_job(self, job_id: str, updates: Dict[str, Any]):
        if job_id not in self.jobs:
            return
        self.jobs[job_id].update(updates)
        self.jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        if REDIS_AVAILABLE:
            redis_client.setex(f"job:{job_id}", 3600, json.dumps(self.jobs[job_id]))

    async def complete_job(self, job_id: str, results: Any):
        await self.update_job(job_id, {
            "status": "completed",
            "progress": 100.0,
            "results": results,
            "completed_at": datetime.utcnow().isoformat(),
        })
        active_jobs_gauge.dec()

    async def fail_job(self, job_id: str, error: str, retry: bool = True):
        job = self.jobs.get(job_id)
        if not job:
            return
        if retry and job["retries"] < 3:
            job["retries"] += 1
            await self.update_job(job_id, {"status": "queued", "error": f"Retry {job['retries']}/3: {error}"})
            self.job_queues[job["type"]].append(job_id)
        else:
            await self.update_job(job_id, {"status": "failed", "error": error, "completed_at": datetime.utcnow().isoformat()})
            active_jobs_gauge.dec()

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        if job_id in self.jobs:
            return self.jobs[job_id]
        if REDIS_AVAILABLE:
            raw = redis_client.get(f"job:{job_id}")
            if raw:
                return json.loads(raw)
        return None

    async def list_jobs(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j["status"] == status]
        jobs.sort(key=lambda x: x["created_at"], reverse=True)
        return jobs[:limit]

    def get_queue_stats(self) -> Dict[str, int]:
        return {jt: len(q) for jt, q in self.job_queues.items()}

# -----------------------------
# WebSocket manager
# -----------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        websocket_connections.inc()
        logger.info(f"WebSocket client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            for topic in list(self.subscriptions.keys()):
                self.subscriptions[topic].discard(client_id)
            websocket_connections.dec()
            logger.info(f"WebSocket client {client_id} disconnected")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def broadcast(self, message: str, topic: str = "general"):
        for client_id in self.subscriptions.get(topic, []):
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_text(message)
                except Exception:
                    self.disconnect(client_id)

    def subscribe(self, client_id: str, topic: str):
        self.subscriptions[topic].add(client_id)

    def unsubscribe(self, client_id: str, topic: str):
        self.subscriptions[topic].discard(client_id)

# -----------------------------
# Core ODE service
# -----------------------------
class ODEService:
    def __init__(self):
        self.config = ConfigManager() if CORE_IMPORTS_AVAILABLE else None
        self.generator = None
        self.verifier = None
        self.feature_extractor = None
        self.function_library: Dict[str, Any] = {}
        self.working_generators: Dict[str, Dict[str, Any]] = {"linear": {}, "nonlinear": {}}
        self._initialize()

    def _initialize(self):
        if CORE_IMPORTS_AVAILABLE:
            try:
                self.generator = ODEDatasetGenerator(config=self.config)
                self.verifier = ODEVerifier(self.config.config.get("verification", {}))
                self.feature_extractor = FeatureExtractor()
                self.function_library = AnalyticFunctionLibrary.get_safe_library()

                self.working_generators = self.generator.test_generators()
                logger.info(
                    f"Initialized {len(self.working_generators['linear'])} linear and "
                    f"{len(self.working_generators['nonlinear'])} nonlinear generators"
                )
            except Exception as e:
                logger.error(f"Failed to initialize ODE services: {e}")
                self._setup_demo_mode()
        else:
            self._setup_demo_mode()

    def _setup_demo_mode(self):
        logger.warning("Running in demo mode")
        self.working_generators = {
            "linear": {"L1": None, "L2": None, "L3": None, "L4": None},
            "nonlinear": {"N1": None, "N2": None, "N3": None},
        }
        self.function_library = {"identity": None, "sine": None, "cosine": None, "exponential": None, "quadratic": None}

    def get_available_generators(self) -> List[str]:
        return list(self.working_generators["linear"].keys()) + list(self.working_generators["nonlinear"].keys())

    def get_available_functions(self) -> List[str]:
        return list(self.function_library.keys())

    async def generate_ode(self, generator: str, function: str, params: Optional[Dict] = None) -> Optional[Dict]:
        if not CORE_IMPORTS_AVAILABLE or not self.generator:
            # Demo payload
            return {
                "id": str(uuid.uuid4()),
                "generator": generator,
                "function": function,
                "ode": f"Eq(Derivative(y(x), x, 2) + y(x), pi*{function}(x))",
                "solution": f"pi*{function}(x)",
                "verified": True,
                "complexity": 100,
                "parameters": params or {},
            }
        try:
            gen_type = "linear" if generator in self.working_generators["linear"] else "nonlinear"
            ode_instance: Optional[ODEInstance] = self.generator.generate_single_ode(
                generator=self.working_generators[gen_type][generator],
                gen_type=gen_type,
                gen_name=generator,
                f_key=function,
                ode_id=0,
            )
            if ode_instance:
                return {
                    "id": str(ode_instance.id),
                    "generator": ode_instance.generator_name,
                    "function": ode_instance.function_name,
                    "ode": ode_instance.ode_symbolic,
                    "solution": ode_instance.solution_symbolic,
                    "verified": ode_instance.verified,
                    "complexity": ode_instance.complexity_score,
                    "parameters": ode_instance.parameters,
                    "verification_confidence": ode_instance.verification_confidence,
                    "initial_conditions": ode_instance.initial_conditions,
                }
        except Exception as e:
            logger.error(f"Error generating ODE: {e}")
        return None

    async def verify_ode(self, ode: str, solution: str, method: str = "substitution") -> Dict[str, Any]:
        if not CORE_IMPORTS_AVAILABLE or not self.verifier:
            return {"verified": False, "method": method, "confidence": 0.0, "error": "Verification service not available"}
        try:
            ode_expr = sp.sympify(ode)
            sol_expr = sp.sympify(solution)
            verified, ver_method, confidence = self.verifier.verify(ode_expr, sol_expr)
            return {"verified": verified, "method": ver_method.value, "confidence": confidence}
        except Exception as e:
            return {"verified": False, "method": method, "confidence": 0.0, "error": str(e)}

# -----------------------------
# App lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting ODE Master Generators Server on port {PORT}")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Redis: {'Connected' if REDIS_AVAILABLE else 'Using in-memory cache'}")
    logger.info(f"ML Pipeline: {'Available' if ML_AVAILABLE else 'Not available'}")

    app.state.ode_service = ODEService()
    app.state.job_manager = JobManager()
    app.state.ws_manager = ConnectionManager()

    for d in ["data", "models", "logs", "checkpoints", "ml_data"]:
        Path(d).mkdir(exist_ok=True)

    asyncio.create_task(job_processor(app))
    asyncio.create_task(metrics_collector(app))
    yield
    logger.info("Shutting down server...")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="ODE Master Generators API",
    description="Production API for ODE generation, verification, datasets, jobs, ML, and metrics",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Security
# -----------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key

# -----------------------------
# Request metrics middleware
# -----------------------------
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    api_request_counter.labels(endpoint=request.url.path, method=request.method, status=response.status_code).inc()
    api_request_duration.labels(endpoint=request.url.path).observe(duration)
    return response

# -----------------------------
# Health & Info & Metrics
# -----------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "ode_generation": CORE_IMPORTS_AVAILABLE,
            "ml_pipeline": ML_AVAILABLE,
            "cache": REDIS_AVAILABLE,
            "websocket": ENABLE_WEBSOCKET,
        },
    }

@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/api/info")
async def api_info():
    return {
        "name": "ODE Master Generators API",
        "version": "3.0.0",
        "status": "operational",
        "environment": ENVIRONMENT,
        "features": {
            "core": CORE_IMPORTS_AVAILABLE,
            "ml": ML_AVAILABLE,
            "redis": REDIS_AVAILABLE,
            "websocket": ENABLE_WEBSOCKET,
        },
    }

# -----------------------------
# ODE generation / verification
# -----------------------------
@app.get("/api/generators")
async def list_generators(api_key: str = Depends(verify_api_key)):
    s = app.state.ode_service
    return {
        "linear": list(s.working_generators["linear"].keys()),
        "nonlinear": list(s.working_generators["nonlinear"].keys()),
        "all": s.get_available_generators(),
    }

@app.get("/api/functions")
async def list_functions(api_key: str = Depends(verify_api_key)):
    s = app.state.ode_service
    return {"functions": s.get_available_functions(), "count": len(s.function_library)}

@app.post("/api/generate")
async def generate_odes(request: ODEGenerationRequest, api_key: str = Depends(verify_api_key)):
    s = app.state.ode_service
    if request.generator not in s.get_available_generators():
        raise HTTPException(400, f"Unknown generator: {request.generator}")
    if request.function not in s.get_available_functions():
        raise HTTPException(400, f"Unknown function: {request.function}")
    job_id = await app.state.job_manager.create_job("generation", request.dict())
    return {"job_id": job_id, "status": "queued", "check_status_url": f"/api/jobs/{job_id}"}

@app.post("/api/batch_generate")
async def batch_generate(request: BatchGenerationRequest, api_key: str = Depends(verify_api_key)):
    s = app.state.ode_service
    invalid_g = [g for g in request.generators if g not in s.get_available_generators()]
    invalid_f = [f for f in request.functions if f not in s.get_available_functions()]
    if invalid_g:
        raise HTTPException(400, f"Unknown generators: {invalid_g}")
    if invalid_f:
        raise HTTPException(400, f"Unknown functions: {invalid_f}")
    job_id = await app.state.job_manager.create_job("batch_generation", request.dict(), priority=10)
    total_expected = len(request.generators) * len(request.functions) * request.samples_per_combination
    return {"job_id": job_id, "status": "queued", "total_expected": total_expected, "check_status_url": f"/api/jobs/{job_id}"}

@app.post("/api/verify")
async def verify_ode(request: ODEVerificationRequest, api_key: str = Depends(verify_api_key)):
    s = app.state.ode_service
    result = await s.verify_ode(request.ode, request.solution, request.method)
    verification_counter.labels(method=result.get("method", request.method), result="success" if result.get("verified") else "failed").inc()
    return result

# -----------------------------
# Datasets
# -----------------------------
@app.get("/api/datasets")
async def list_datasets(api_key: str = Depends(verify_api_key)):
    datasets = []
    data_dir = Path("data")
    if data_dir.exists():
        for fp in data_dir.glob("*.jsonl"):
            stat = fp.stat()
            metadata = None
            if REDIS_AVAILABLE:
                raw = redis_client.get(f"dataset:{fp.stem}")
                if raw:
                    metadata = json.loads(raw)
            datasets.append({
                "name": fp.stem,
                "path": str(fp),
                "size_bytes": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "metadata": metadata,
            })
    return {"datasets": datasets, "count": len(datasets)}

@app.post("/api/datasets/create")
async def create_dataset(body: CreateDatasetBody, api_key: str = Depends(verify_api_key)):
    name = body.name or f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    fp = Path("data") / f"{name}.jsonl"
    fp.parent.mkdir(parents=True, exist_ok=True)

    with open(fp, "w") as f:
        for ode in body.odes:
            f.write(json.dumps(ode) + "\n")

    verified_count = sum(1 for ode in body.odes if ode.get("verified", False))
    verification_rate = verified_count / len(body.odes) if body.odes else 0.0

    metadata = {
        "name": name,
        "path": str(fp),
        "size": len(body.odes),
        "verified_count": verified_count,
        "verification_rate": verification_rate,
        "created_at": datetime.utcnow().isoformat(),
    }

    if REDIS_AVAILABLE:
        redis_client.setex(f"dataset:{name}", 86400, json.dumps(metadata))

    try:
        dataset_size_gauge.inc(fp.stat().st_size)
    except Exception:
        pass

    return metadata

@app.get("/api/datasets/{name}/download")
async def download_dataset(name: str, format: str = "jsonl", api_key: str = Depends(verify_api_key)):
    fp = Path("data") / f"{name}.jsonl"
    if not fp.exists():
        raise HTTPException(404, "Dataset not found")

    if format == "jsonl":
        return FileResponse(fp, filename=f"{name}.jsonl")
    elif format == "csv":
        df = pd.read_json(fp, lines=True)
        tmp = Path("/tmp") / f"{name}.csv"
        df.to_csv(tmp, index=False)
        return FileResponse(tmp, filename=f"{name}.csv")
    else:
        raise HTTPException(400, f"Unsupported format: {format}")

# -----------------------------
# Jobs
# -----------------------------
@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, api_key: str = Depends(verify_api_key)):
    job = await app.state.job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatus(
        job_id=job["id"],
        status=job["status"],
        progress=job["progress"],
        results=job.get("results"),
        error=job.get("error"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        eta=calculate_eta(job) if job["status"] == "running" else None,
        metadata=job.get("metadata", {}),
    )

@app.get("/api/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 100, api_key: str = Depends(verify_api_key)):
    jobs = await app.state.job_manager.list_jobs(status=status, limit=limit)
    return {
        "jobs": [
            JobStatus(
                job_id=j["id"],
                status=j["status"],
                progress=j["progress"],
                results=None,  # omit heavy results in listing
                error=j.get("error"),
                created_at=j["created_at"],
                updated_at=j["updated_at"],
                metadata=j.get("metadata", {}),
            )
            for j in jobs
        ],
        "count": len(jobs),
        "queue_stats": app.state.job_manager.get_queue_stats(),
    }

@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str, api_key: str = Depends(verify_api_key)):
    job = await app.state.job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] in ["completed", "failed"]:
        raise HTTPException(400, f"Cannot cancel {job['status']} job")
    await app.state.job_manager.update_job(job_id, {"status": "cancelled", "error": "Cancelled by user"})
    active_jobs_gauge.dec()
    return {"message": "Job cancelled"}

# -----------------------------
# ML endpoints (optional)
# -----------------------------
@app.post("/api/ml/train")
async def train_ml_model(request: MLTrainingRequest, api_key: str = Depends(verify_api_key)):
    if not ML_AVAILABLE:
        raise HTTPException(503, "ML pipeline not available")

    dataset_path = Path("data") / f"{request.dataset}.jsonl"
    if not dataset_path.exists():
        raise HTTPException(404, f"Dataset {request.dataset} not found")

    job_id = await app.state.job_manager.create_job("ml_training", request.dict(), priority=15)
    return {"job_id": job_id, "status": "queued", "check_status_url": f"/api/jobs/{job_id}"}

@app.post("/api/ml/generate")
async def generate_with_ml(request: MLGenerationRequest, api_key: str = Depends(verify_api_key)):
    if not ML_AVAILABLE:
        raise HTTPException(503, "ML pipeline not available")

    model_path = Path("models") / request.model_path
    if not model_path.exists():
        raise HTTPException(404, f"Model {request.model_path} not found")

    job_id = await app.state.job_manager.create_job("ml_generation", request.dict())
    return {"job_id": job_id, "status": "queued", "check_status_url": f"/api/jobs/{job_id}"}

@app.get("/api/ml/models")
async def list_ml_models(api_key: str = Depends(verify_api_key)):
    models = []
    models_dir = Path("models")
    if models_dir.exists():
        for p in models_dir.glob("*.pth"):
            stat = p.stat()
            meta = {}
            meta_path = p.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            models.append({
                "name": p.stem,
                "path": str(p),
                "size_bytes": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "metadata": meta,
            })
    return {"models": models, "count": len(models), "ml_available": ML_AVAILABLE}

@app.get("/api/stats")
async def get_statistics(api_key: str = Depends(verify_api_key)):
    total_generated_24h = int(redis_client.get("stats:generated_24h") or 0) if REDIS_AVAILABLE else 0
    verification_rate = float(redis_client.get("stats:verification_rate") or 0.0) if REDIS_AVAILABLE else 0.0
    active_jobs = sum(1 for j in app.state.job_manager.jobs.values() if j["status"] in ["running", "queued"])
    return {
        "status": "operational",
        "statistics": {
            "total_generated_24h": total_generated_24h,
            "verification_rate": verification_rate,
            "active_jobs": active_jobs,
            "total_jobs": len(app.state.job_manager.jobs),
        },
        "capabilities": {
            "generators": len(app.state.ode_service.get_available_generators()),
            "functions": len(app.state.ode_service.get_available_functions()),
            "ml_enabled": ML_AVAILABLE,
            "redis_enabled": REDIS_AVAILABLE,
        },
        "queue_stats": app.state.job_manager.get_queue_stats(),
    }

# -----------------------------
# WebSocket endpoint (optional)
# -----------------------------
if ENABLE_WEBSOCKET:
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        await app.state.ws_manager.connect(websocket, client_id)
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                t = msg.get("type")
                if t == "subscribe":
                    topic = msg.get("topic", "general")
                    app.state.ws_manager.subscribe(client_id, topic)
                    await websocket.send_text(json.dumps({"type": "subscribed", "topic": topic}))
                elif t == "unsubscribe":
                    topic = msg.get("topic", "general")
                    app.state.ws_manager.unsubscribe(client_id, topic)
                elif t == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
        except WebSocketDisconnect:
            app.state.ws_manager.disconnect(client_id)

# -----------------------------
# Background processors
# -----------------------------
async def process_generation_job(app: FastAPI, job_id: str, request: ODEGenerationRequest):
    try:
        await app.state.job_manager.update_job(job_id, {"status": "running", "started_at": datetime.utcnow().isoformat()})
        results: List[Dict[str, Any]] = []
        svc = app.state.ode_service

        with generation_time_histogram.time():
            for i in range(request.count):
                ode_data = await svc.generate_ode(request.generator, request.function, request.parameters)
                if ode_data:
                    if request.verify:
                        ode_data["verification"] = await svc.verify_ode(ode_data["ode"], ode_data["solution"])
                    results.append(ode_data)
                    ode_generation_counter.labels(generator=request.generator, function=request.function).inc()

                # progress after each attempt
                await app.state.job_manager.update_job(job_id, {
                    "progress": ((i + 1) / max(1, request.count)) * 100.0,
                    "metadata": {"current": i + 1, "total": request.count},
                })

                if request.stream and ENABLE_WEBSOCKET:
                    await app.state.ws_manager.broadcast(
                        json.dumps({"type": "ode_generated", "job_id": job_id, "ode": ode_data}),
                        topic=f"job:{job_id}",
                    )

        if REDIS_AVAILABLE:
            redis_client.incr("stats:generated_24h")
            if results:
                verified = sum(1 for r in results if r.get("verified", False))
                rate = verified / len(results)
                redis_client.set("stats:verification_rate", str(rate))

        await app.state.job_manager.complete_job(job_id, results)
    except Exception as e:
        logger.error(f"Generation job {job_id} failed: {e}\n{traceback.format_exc()}")
        await app.state.job_manager.fail_job(job_id, str(e))

async def process_batch_generation_job(app: FastAPI, job_id: str, request: BatchGenerationRequest):
    try:
        await app.state.job_manager.update_job(job_id, {"status": "running"})
        results: List[Dict[str, Any]] = []
        svc = app.state.ode_service
        total = len(request.generators) * len(request.functions) * request.samples_per_combination
        current = 0

        for g in request.generators:
            for f in request.functions:
                for _ in range(request.samples_per_combination):
                    params: Dict[str, Any] = {}
                    if request.parameter_ranges:
                        for k, values in request.parameter_ranges.items():
                            if isinstance(values, list) and values:
                                params[k] = np.random.choice(values)

                    ode_data = await svc.generate_ode(g, f, params)
                    if ode_data:
                        if request.verify:
                            ode_data["verification"] = await svc.verify_ode(ode_data["ode"], ode_data["solution"])
                        results.append(ode_data)

                    current += 1
                    await app.state.job_manager.update_job(job_id, {
                        "progress": (current / max(1, total)) * 100.0,
                        "metadata": {"current": current, "total": total, "generator": g, "function": f},
                    })

        dataset_name = None
        if request.save_dataset and results:
            dataset_name = request.dataset_name or f"batch_{job_id[:8]}"
            ds_path = Path("data") / f"{dataset_name}.jsonl"
            ds_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ds_path, "w") as f:
                for ode in results:
                    f.write(json.dumps(ode) + "\n")

            metadata = {
                "name": dataset_name,
                "path": str(ds_path),
                "size": len(results),
                "generators": request.generators,
                "functions": request.functions,
                "created_at": datetime.utcnow().isoformat(),
            }
            if REDIS_AVAILABLE:
                redis_client.setex(f"dataset:{dataset_name}", 86400, json.dumps(metadata))

        await app.state.job_manager.complete_job(job_id, {
            "total_generated": len(results),
            "dataset_name": dataset_name,
            "summary": {
                "verified": sum(1 for r in results if r.get("verified", False)),
                "generators": list({r["generator"] for r in results}),
                "functions": list({r["function"] for r in results}),
            },
        })
    except Exception as e:
        logger.error(f"Batch generation job {job_id} failed: {e}")
        await app.state.job_manager.fail_job(job_id, str(e))

async def process_ml_training_job(app: FastAPI, job_id: str, request: MLTrainingRequest):
    if not ML_AVAILABLE:
        await app.state.job_manager.fail_job(job_id, "ML pipeline not available")
        return
    try:
        await app.state.job_manager.update_job(job_id, {"status": "running"})

        dataset_path = Path("data") / f"{request.dataset}.jsonl"
        ml_dir = Path("ml_data") / job_id
        ml_dir.mkdir(parents=True, exist_ok=True)

        data_paths = prepare_ml_dataset(str(dataset_path), output_dir=str(ml_dir), test_split=0.2, val_split=request.validation_split)
        trainer = ODEGeneratorTrainer(dataset_path=str(dataset_path), features_path=data_paths["train"])

        model_path = None
        if request.model_type == "pattern_net":
            model = trainer.train_pattern_network(epochs=request.epochs, batch_size=request.batch_size)
            model_path = f"models/{job_id}_pattern.pth"
        elif request.model_type == "transformer":
            model = trainer.train_language_model(epochs=request.epochs, batch_size=request.batch_size)
            model_path = f"models/{job_id}_transformer.pth"
        else:
            raise ValueError(f"Unsupported model_type: {request.model_type}")

        if model_path:
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_path)
            meta = {
                "model_type": request.model_type,
                "dataset": request.dataset,
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "created_at": datetime.utcnow().isoformat(),
            }
            with open(Path(model_path).with_suffix(".json"), "w") as f:
                json.dump(meta, f)

        await app.state.job_manager.complete_job(job_id, {"model_path": model_path, "training_completed": True})
    except Exception as e:
        logger.error(f"ML training job {job_id} failed: {e}")
        await app.state.job_manager.fail_job(job_id, str(e))

async def process_ml_generation_job(app: FastAPI, job_id: str, request: MLGenerationRequest):
    if not ML_AVAILABLE:
        await app.state.job_manager.fail_job(job_id, "ML pipeline not available")
        return
    try:
        await app.state.job_manager.update_job(job_id, {"status": "running"})

        model = load_pretrained_model(
            model_type="pattern_net",  # could infer from filename if desired
            checkpoint_path=f"models/{request.model_path}",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        generated_odes = generate_novel_odes(
            model=model,
            n_samples=request.n_samples,
            generators=request.generators,
            functions=request.functions,
            temperature=request.temperature,
        )

        await app.state.job_manager.complete_job(job_id, {"generated_odes": generated_odes, "count": len(generated_odes)})
    except Exception as e:
        logger.error(f"ML generation job {job_id} failed: {e}")
        await app.state.job_manager.fail_job(job_id, str(e))

# Background workers
async def job_processor(app: FastAPI):
    while True:
        try:
            for job_type, queue in list(app.state.job_manager.job_queues.items()):
                if not queue:
                    continue
                job_id = queue.popleft()
                job = await app.state.job_manager.get_job(job_id)
                if not job or job["status"] != "queued":
                    continue

                if job_type == "generation":
                    asyncio.create_task(process_generation_job(app, job_id, ODEGenerationRequest(**job["params"])))
                elif job_type == "batch_generation":
                    asyncio.create_task(process_batch_generation_job(app, job_id, BatchGenerationRequest(**job["params"])))
                elif job_type == "ml_training":
                    asyncio.create_task(process_ml_training_job(app, job_id, MLTrainingRequest(**job["params"])))
                elif job_type == "ml_generation":
                    asyncio.create_task(process_ml_generation_job(app, job_id, MLGenerationRequest(**job["params"])))
                else:
                    logger.warning(f"Unknown job type: {job_type}")

            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Job processor error: {e}")
            await asyncio.sleep(5)

async def metrics_collector(app: FastAPI):
    while True:
        try:
            await asyncio.sleep(60)
            if REDIS_AVAILABLE:
                active = sum(1 for j in app.state.job_manager.jobs.values() if j["status"] in ["running", "queued"])
                redis_client.set("metrics:active_jobs", str(active))
        except Exception as e:
            logger.error(f"Metrics collector error: {e}")

# -----------------------------
# Utilities
# -----------------------------
def calculate_eta(job: Dict[str, Any]) -> Optional[str]:
    if job.get("status") != "running" or job.get("progress", 0) == 0:
        return None
    started_at = datetime.fromisoformat(job.get("started_at", job["created_at"]))
    elapsed = (datetime.utcnow() - started_at).total_seconds()
    progress = job.get("progress", 0.0)
    if progress <= 0:
        return None
    total_est = elapsed / (progress / 100.0)
    remaining = total_est - elapsed
    return (datetime.utcnow() + timedelta(seconds=remaining)).isoformat()

# =====================================================================
# GUI mounting (IMPORTANT: last, so it does NOT intercept /api/*)
# =====================================================================
from fastapi.responses import HTMLResponse  # noqa: E402

GUI_BUNDLE_DIR_ENV = os.getenv("GUI_BUNDLE_DIR", "").strip()
CANDIDATES: List[Path] = []
if GUI_BUNDLE_DIR_ENV:
    CANDIDATES.append(Path(GUI_BUNDLE_DIR_ENV))
repo_root = Path(__file__).resolve().parents[1]
CANDIDATES += [
    repo_root / "ode_gui_bundle",
    repo_root / "ode_gui_bundle" / "dist",
    repo_root / "ode_gui_bundle" / "build",
    repo_root / "gui" / "gui" / "dist",
]
BUNDLE_DIR: Optional[Path] = None
for root in CANDIDATES:
    if (root / "index.html").exists():
        BUNDLE_DIR = root
        break

if BUNDLE_DIR:
    logger.info(f"Serving GUI from: {BUNDLE_DIR}")

    @app.get("/config.js", include_in_schema=False)
    async def config_js():
        api_base = os.getenv("PUBLIC_API_BASE", "")  # e.g., "https://ode-api.up.railway.app"
        api_key = os.getenv("PUBLIC_API_KEY", "")    # expose only if intentionally public
        ws_enabled = os.getenv("ENABLE_WEBSOCKET", "true").lower() == "true"
        content = "window.ODE_CONFIG=" + json.dumps({"API_BASE": api_base, "API_KEY": api_key, "WS": ws_enabled}) + ";"
        return Response(content, media_type="application/javascript")

    assets_dir = BUNDLE_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    else:
        logger.warning("GUI found but no assets/ folder. If index.html references /assets/*, those will 404.")

    # Mount SPA at root with HTML fallback (no manual catch-all route)
    app.mount("/", StaticFiles(directory=BUNDLE_DIR, html=True), name="ui")

else:
    logger.warning(
        "GUI bundle not found. Looked in:\n  - " + "\n  - ".join(str(p) for p in CANDIDATES) +
        "\nSet GUI_BUNDLE_DIR env var to the folder containing index.html."
    )

    @app.get("/", include_in_schema=False)
    async def root_info():
        return {
            "name": "ODE Master Generators API",
            "version": "3.0.0",
            "status": "operational",
            "environment": ENVIRONMENT,
            "features": {
                "core": CORE_IMPORTS_AVAILABLE,
                "ml": ML_AVAILABLE,
                "redis": REDIS_AVAILABLE,
                "websocket": ENABLE_WEBSOCKET,
            },
        }

# -----------------------------
# Local entry point
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info" if ENVIRONMENT == "production" else "debug")
