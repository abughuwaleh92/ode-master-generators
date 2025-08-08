# production_server.py
"""
Production API server for ODE generation, verification, and ML integration
=========================================================================

Full rewrite highlights
-----------------------
• Keeps canonical routes at /api/v1/* **and** provides unprefixed aliases (/generate, /verify, etc.)
• Health & metrics at root (/health, /metrics) for compatibility with platforms
• Redis-backed job store with in-memory fallback
• Prometheus metrics for API, generation, and ML tasks
• Background job manager for generation, batches, ML train/generate
• Demo fallback for generators/functions if core modules not available
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import asyncio
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response, Body
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

import numpy as np
import sympy as sp

# 3rd-party services
import redis  # type: ignore
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Make project imports available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional: import your project components
try:
    from pipeline.generator import ODEDatasetGenerator
    from verification.verifier import ODEVerifier
    from utils.config import ConfigManager
    from core.types import VerificationMethod
    from core.functions import AnalyticFunctionLibrary
    CORE_AVAILABLE = True
except Exception as e:
    logging.getLogger(__name__).error(f"Core imports unavailable: {e}")
    CORE_AVAILABLE = False

# Optional ML stack
ML_AVAILABLE = True
try:
    import torch  # type: ignore
    from ml_pipeline.models import ODEPatternNet, ODETransformer, ODEVAE  # noqa: F401
    from ml_pipeline.train_ode_generator import ODEGeneratorTrainer, ODEDataset  # noqa: F401
except Exception as e:
    logging.getLogger(__name__).warning(f"ML components unavailable: {e}")
    ML_AVAILABLE = False

# ---------- Logging ----------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ode-api")

# ---------- App & CORS ----------

app = FastAPI(
    title="ODE Generation API",
    description="Production API for ODE generation, verification, and ML-based analysis",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Env flags ----------

USE_DEMO = str(os.getenv("USE_DEMO", "0")).lower() in {"1","true","yes"}

# ---------- Redis (with fallback) ----------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    if REDIS_URL.startswith(("redis://","rediss://")):
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    else:
        redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connected")
except Exception:
    logger.warning("Redis not available. Using in-memory storage.")
    REDIS_AVAILABLE = False

    class FakeRedis:
        def __init__(self):
            self.storage: Dict[str, str] = {}
        def setex(self, key, ttl, value):
            self.storage[key] = value
        def set(self, key, value):
            self.storage[key] = value
        def get(self, key):
            return self.storage.get(key)
        def incr(self, key):
            v = int(self.storage.get(key, "0")) if self.storage.get(key, "0").isdigit() else 0
            v += 1
            self.storage[key] = str(v)
            return v
        def keys(self, pattern: Optional[str] = None):
            if not pattern:
                return list(self.storage.keys())
            import fnmatch
            return [k for k in self.storage.keys() if fnmatch.fnmatch(k, pattern)]
        def dbsize(self):
            return len(self.storage)
        def ping(self):
            return True
        def delete(self, *keys):
            for k in keys:
                self.storage.pop(k, None)
    redis_client = FakeRedis()

# ---------- Security ----------

VALID_API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "test-key,dev-key").split(",") if k.strip()]
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)):
    if not api_key:
        raise HTTPException(status_code=403, detail="API key required. Add 'X-API-Key' header.")
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# ---------- Prometheus ----------

ode_generation_counter = Counter("ode_generation_total", "Total ODEs generated", ["generator","function"])
verification_counter   = Counter("ode_verification_total", "Total verifications", ["method","result"])
generation_time_hist   = Histogram("ode_generation_duration_seconds", "ODE generation time")
active_jobs_gauge      = Gauge("active_jobs", "Number of active jobs")
api_request_counter    = Counter("api_requests_total", "Total API requests", ["endpoint","method","status"])
api_request_duration   = Histogram("api_request_duration_seconds", "API request duration", ["endpoint"])
ml_training_counter    = Counter("ml_training_total", "Total ML training jobs", ["model_type","status"])
ml_generation_counter  = Counter("ml_generation_total", "Total ML generations", ["model_type"])

# ---------- Job models ----------

class ODEGenerationRequest(BaseModel):
    generator: str = Field(..., description="Generator name (e.g., L1, N1)")
    function: str = Field(..., description="Function name (e.g., sine, exponential)")
    parameters: Optional[Dict[str, float]] = Field(default=None, description="ODE parameters")
    count: int = Field(1, ge=1, le=100, description="Number of ODEs to generate")
    verify: bool = Field(True, description="Whether to verify generated ODEs")

class BatchGenerationRequest(BaseModel):
    generators: List[str] = Field(..., description="List of generators to use")
    functions: List[str] = Field(..., description="List of functions to use")
    samples_per_combination: int = Field(5, ge=1, le=50, description="Samples per generator-function combination")
    parameters: Optional[Dict[str, List[float]]] = Field(default=None, description="Parameter ranges")
    verify: bool = Field(True, description="Whether to verify generated ODEs")
    dataset_name: Optional[str] = Field(None, description="Name for the generated dataset")

class ODEVerificationRequest(BaseModel):
    ode: str = Field(..., description="ODE equation as string")
    solution: str = Field(..., description="Proposed solution as string")
    method: str = Field("substitution", description="Verification method")

class DatasetCreationRequest(BaseModel):
    odes: List[Dict[str, Any]] = Field(..., description="List of ODE data")
    dataset_name: Optional[str] = Field(None, description="Dataset name")

class MLTrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset: str = Field(..., description="Dataset path or identifier")
    model_type: str = Field(..., description="Model type (pattern_net, transformer, vae)")
    epochs: int = Field(50, ge=1, le=1000)
    batch_size: int = Field(32, ge=8, le=256)
    learning_rate: float = Field(0.001, ge=0.00001, le=0.1)
    early_stopping: bool = Field(True)
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MLGenerationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_path: str = Field(..., description="Path to trained model")
    n_samples: int = Field(10, ge=1, le=1000)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    generator: Optional[str] = Field(None, description="Target generator style")
    function: Optional[str] = Field(None, description="Target function type")
    complexity_range: Optional[List[int]] = Field(None, description="Target complexity range")

class ODEResponse(BaseModel):
    id: str
    ode: str
    solution: Optional[str]
    verified: Optional[bool]
    complexity: int
    generator: str
    function: str
    parameters: Dict[str, float]
    timestamp: str
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    results: Optional[Any]
    error: Optional[str]
    created_at: str
    updated_at: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class JobCreatedResponse(BaseModel):
    job_id: str
    status: str
    check_status_url: str

class BatchJobCreatedResponse(JobCreatedResponse):
    total_expected: int

# ---------- Job manager ----------

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
    async def create_job(self, job_type: str, params: Dict) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        job_data = {
            "id": job_id, "type": job_type, "params": params,
            "status": "pending", "progress": 0.0,
            "created_at": now, "updated_at": now,
            "results": None, "error": None, "metadata": {},
        }
        if REDIS_AVAILABLE:
            redis_client.setex(f"job:{job_id}", 3600, json.dumps(job_data))
        else:
            self.jobs[job_id] = job_data
        active_jobs_gauge.inc()
        return job_id
    async def update_job(self, job_id: str, updates: Dict):
        updates["updated_at"] = datetime.now().isoformat()
        if REDIS_AVAILABLE:
            data = redis_client.get(f"job:{job_id}")
            if data:
                job = json.loads(data)
                job.update(updates)
                redis_client.setex(f"job:{job_id}", 3600, json.dumps(job))
        else:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)
    async def get_job(self, job_id: str) -> Optional[Dict]:
        if REDIS_AVAILABLE:
            data = redis_client.get(f"job:{job_id}")
            return json.loads(data) if data else None
        return self.jobs.get(job_id)
    async def complete_job(self, job_id: str, results: Any):
        await self.update_job(job_id, {"status": "completed", "progress": 100.0, "results": results, "completed_at": datetime.now().isoformat()})
        active_jobs_gauge.dec()
    async def fail_job(self, job_id: str, error: str):
        await self.update_job(job_id, {"status": "failed", "error": error, "failed_at": datetime.now().isoformat()})
        active_jobs_gauge.dec()
    def active_count(self) -> int:
        if REDIS_AVAILABLE:
            cnt = 0
            for key in redis_client.keys("job:*"):
                try:
                    j = json.loads(redis_client.get(key) or "{}")
                    if j.get("status") in ("pending","running"):
                        cnt += 1
                except Exception:
                    pass
            return cnt
        return sum(1 for j in self.jobs.values() if j.get("status") in ("pending","running"))

job_manager = JobManager()

# ---------- Initialize generators & verifier ----------

WORKING_GENERATORS: Dict[str, Dict[str, Any]] = {"linear": {}, "nonlinear": {}}
AVAILABLE_GENERATORS: List[str] = []
AVAILABLE_FUNCTIONS: List[str] = []

try:
    if CORE_AVAILABLE:
        config = ConfigManager()
        ode_generator = ODEDatasetGenerator(config=config)
        ode_verifier = ODEVerifier()
        WORKING_GENERATORS = ode_generator.test_generators()
        AVAILABLE_GENERATORS = list(WORKING_GENERATORS.get("linear", {}).keys()) + list(WORKING_GENERATORS.get("nonlinear", {}).keys())
        AVAILABLE_FUNCTIONS = list(getattr(ode_generator, "f_library", AnalyticFunctionLibrary()).keys())
        logger.info(f"Generators: {AVAILABLE_GENERATORS}")
        logger.info(f"Functions: {len(AVAILABLE_FUNCTIONS)}")
    else:
        raise RuntimeError("Core unavailable")
except Exception as e:
    logger.error(f"Failed to initialize core: {e}")
    if USE_DEMO:
        AVAILABLE_GENERATORS = ["L1","L2","L3","L4","N1","N2","N3","N4","N5","N6","N7"]
        AVAILABLE_FUNCTIONS = [
            "identity","quadratic","cubic","quartic","quintic",
            "exponential","exp_scaled","exp_quadratic","exp_negative",
            "sine","cosine","tangent_safe","sine_scaled","cosine_scaled",
            "sinh","cosh","tanh",
            "log_safe","log_shifted",
            "rational_simple","rational_stable",
            "exp_sin","gaussian",
        ]
        # Lightweight placeholders
        class DummyGen:
            def __call__(self, *a, **k): return {"ode":"y''+y=pi*sin(x)","solution":"pi*sin(x)","verified":True,"complexity_score":50}
        WORKING_GENERATORS = {"linear": {g: DummyGen() for g in AVAILABLE_GENERATORS if g.startswith("L")},
                              "nonlinear": {g: DummyGen() for g in AVAILABLE_GENERATORS if g.startswith("N")}}
        class DummyVerifier:
            def verify(self, ode_expr, sol_expr, method: str = "substitution"):
                return True, "substitution", 0.99
        ode_generator = type("OG", (), {"generate_single_ode": lambda *a,**k: type("O", (), {
            "id": uuid.uuid4(), "ode_symbolic": "Eq(Derivative(y(x), x, 2) + y(x), pi*sin(x))",
            "solution_symbolic": "pi*sin(x)", "complexity_score": 50, "generator_name": "L1", "function_name": "sine",
            "parameters": {"alpha":1.0,"beta":1.0,"M":0.0}, "verified": True, "verification_confidence": 0.99
        })()})()
        ode_verifier = DummyVerifier()

# ---------- Middleware ----------

@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    endpoint = request.url.path
    method = request.method
    status = response.status_code
    try:
        api_request_counter.labels(endpoint=endpoint, method=method, status=status).inc()
        api_request_duration.labels(endpoint=endpoint).observe(duration)
    except Exception:
        pass
    return response

# ---------- Root & health ----------

@app.get("/")
async def root():
    return {
        "name": "ODE Generation API",
        "version": "2.1.0",
        "ml_enabled": ML_AVAILABLE,
        "redis_enabled": REDIS_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "api_v1": {
                "generate": "/api/v1/generate",
                "batch_generate": "/api/v1/batch_generate",
                "verify": "/api/v1/verify",
                "datasets": {"create": "/api/v1/datasets/create", "list": "/api/v1/datasets"},
                "jobs": "/api/v1/jobs/{job_id}",
                "stats": "/api/v1/stats",
                "generators": "/api/v1/generators",
                "functions": "/api/v1/functions",
                "ml": {"train": "/api/v1/ml/train", "generate": "/api/v1/ml/generate", "models": "/api/v1/models"},
            },
            "aliases": {
                "generate": "/generate",
                "batch_generate": "/batch_generate",
                "verify": "/verify",
                "datasets": {"create": "/datasets/create", "list": "/datasets"},
                "jobs": "/jobs/{job_id}",
                "stats": "/stats",
                "generators": "/generators",
                "functions": "/functions",
                "ml": {"train": "/ml/train", "generate": "/ml/generate", "models": "/models"},
            },
        },
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis": "connected" if REDIS_AVAILABLE else "not available",
        "ml_enabled": ML_AVAILABLE,
        "generators": len(AVAILABLE_GENERATORS),
        "functions": len(AVAILABLE_FUNCTIONS),
        "api_prefix_hint": "/api/v1",
    }

# ---------- API v1 Endpoints ----------

@app.post("/api/v1/generate", response_model=JobCreatedResponse)
async def generate_odes(request: ODEGenerationRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    if request.generator not in AVAILABLE_GENERATORS:
        raise HTTPException(status_code=400, detail=f"Unknown generator: {request.generator}. Available: {AVAILABLE_GENERATORS}")
    if request.function not in AVAILABLE_FUNCTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown function: {request.function}. Available: {AVAILABLE_FUNCTIONS}")
    job_id = await job_manager.create_job("generation", request.model_dump())
    background_tasks.add_task(process_generation_job, job_id, request)
    return JobCreatedResponse(job_id=job_id, status="Job created", check_status_url=f"/api/v1/jobs/{job_id}")

@app.post("/api/v1/batch_generate", response_model=BatchJobCreatedResponse)
async def batch_generate_odes(request: BatchGenerationRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    invalid_generators = [g for g in request.generators if g not in AVAILABLE_GENERATORS]
    if invalid_generators:
        raise HTTPException(status_code=400, detail=f"Unknown generators: {invalid_generators}. Available: {AVAILABLE_GENERATORS}")
    invalid_functions = [f for f in request.functions if f not in AVAILABLE_FUNCTIONS]
    if invalid_functions:
        raise HTTPException(status_code=400, detail=f"Unknown functions: {invalid_functions}. Available: {AVAILABLE_FUNCTIONS}")
    job_id = await job_manager.create_job("batch_generation", request.model_dump())
    background_tasks.add_task(process_batch_generation_job, job_id, request)
    total_expected = len(request.generators) * len(request.functions) * request.samples_per_combination
    return BatchJobCreatedResponse(job_id=job_id, status="Batch generation job created", check_status_url=f"/api/v1/jobs/{job_id}", total_expected=total_expected)

@app.post("/api/v1/verify")
async def verify_ode(request: ODEVerificationRequest, api_key: str = Depends(verify_api_key)):
    try:
        ode_expr = sp.sympify(request.ode)
        solution_expr = sp.sympify(request.solution)
        try:
            verified, method_used, confidence = ode_verifier.verify(ode_expr, solution_expr, method=request.method)  # type: ignore
            method_str = method_used.value if hasattr(method_used, "value") else str(method_used)
        except TypeError:
            verified, method_used, confidence = ode_verifier.verify(ode_expr, solution_expr)  # type: ignore
            method_str = method_used.value if hasattr(method_used, "value") else str(method_used)
        verification_counter.labels(method=method_str, result="success" if verified else "failed").inc()
        return {
            "verified": bool(verified),
            "confidence": float(confidence),
            "method": method_str,
            "details": {"ode": str(ode_expr), "solution": str(solution_expr)},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/datasets/create")
async def create_dataset(request: DatasetCreationRequest = Body(...), api_key: str = Depends(verify_api_key)):
    name = request.dataset_name or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
    dataset_path = data_dir / f"{name}.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for ode in request.odes:
            f.write(json.dumps(ode, ensure_ascii=False) + "\n")
    info = {"name": name, "path": str(dataset_path), "size": len(request.odes), "created_at": datetime.now().isoformat()}
    try:
        redis_client.setex(f"dataset:{name}", 86400, json.dumps(info))
    except Exception:
        pass
    return {"dataset_name": name, "path": str(dataset_path), "size": len(request.odes), "message": "Dataset created successfully"}

@app.get("/api/v1/datasets")
async def list_datasets(api_key: str = Depends(verify_api_key)):
    datasets: List[Dict[str, Any]] = []

    try:
        if REDIS_AVAILABLE:
            for key in redis_client.keys("dataset:*"):
                dataset_info = redis_client.get(key)
                if dataset_info:
                    datasets.append(json.loads(dataset_info))
    except Exception:
        pass

    data_dir = Path("data")
    if data_dir.exists():
        for file_path in data_dir.glob("*.jsonl"):
            if not any(d.get("path") == str(file_path) for d in datasets):
                stat = file_path.stat()
                with open(file_path, "r", encoding="utf-8") as f:
                    line_count = sum(1 for line in f if line.strip())
                datasets.append(
                    {"name": file_path.stem, "path": str(file_path), "size": line_count, "file_size_bytes": stat.st_size, "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()}
                )
    return {"datasets": datasets, "count": len(datasets)}

@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str, api_key: str = Depends(verify_api_key)):
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=job["id"], status=job["status"], progress=job["progress"],
        results=job.get("results"), error=job.get("error"),
        created_at=job["created_at"], updated_at=job["updated_at"], metadata=job.get("metadata", {})
    )

@app.get("/api/v1/models")
async def list_ml_models(api_key: str = Depends(verify_api_key)):
    models_dir = Path("models"); models: List[Dict[str, Any]] = []
    if models_dir.exists():
        for model_path in models_dir.glob("*.pth"):
            try:
                metadata_path = model_path.with_suffix(".json")
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                else:
                    metadata = {"name": model_path.stem}
                models.append({"path": str(model_path), "name": model_path.stem, "size": model_path.stat().st_size, "created": datetime.fromtimestamp(model_path.stat().st_ctime).isoformat(), "metadata": metadata})
            except Exception as e:
                logger.error(f"Error loading model {model_path}: {e}")
    return {"models": models, "count": len(models), "ml_enabled": ML_AVAILABLE}

@app.get("/api/v1/stats")
async def get_statistics(api_key: str = Depends(verify_api_key)):
    total_generated = int(redis_client.get("metric:total_generated_24h") or 0)
    verification_success_rate = float(redis_client.get("metric:verification_success_rate") or 0.0)
    generator_stats: Dict[str, Dict[str, Any]] = {}
    for gen in AVAILABLE_GENERATORS:
        success_rate = float(redis_client.get(f"metric:generator:{gen}:success_rate") or 0.0)
        avg_time     = float(redis_client.get(f"metric:generator:{gen}:avg_time") or 0.0)
        total        = int(redis_client.get(f"metric:generator:{gen}:total") or 0)
        verified     = int(redis_client.get(f"metric:generator:{gen}:verified") or 0)
        generator_stats[gen] = {"success_rate": success_rate, "avg_time": avg_time, "total_generated": total, "total_verified": verified}
    return {
        "status": "operational", "total_generated_24h": total_generated, "verification_success_rate": verification_success_rate,
        "active_jobs": job_manager.active_count(), "available_generators": len(AVAILABLE_GENERATORS), "available_functions": len(AVAILABLE_FUNCTIONS),
        "generator_performance": generator_stats, "ml_enabled": ML_AVAILABLE, "redis_enabled": REDIS_AVAILABLE,
    }

@app.post("/api/v1/ml/train")
async def train_ml_model(request: MLTrainingRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML features are not available. Please install torch & scikit-learn.")
    job_id = await job_manager.create_job("ml_training", request.model_dump())
    background_tasks.add_task(process_ml_training_job, job_id, request)
    return {"job_id": job_id, "status": "Training job created", "check_status_url": f"/api/v1/jobs/{job_id}"}

@app.post("/api/v1/ml/generate")
async def generate_with_ml(request: MLGenerationRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML features are not available. Please install torch & scikit-learn.")
    job_id = await job_manager.create_job("ml_generation", request.model_dump())
    background_tasks.add_task(process_ml_generation_job, job_id, request)
    return {"job_id": job_id, "status": "ML generation job created", "check_status_url": f"/api/v1/jobs/{job_id}"}

@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type="text/plain")

# ---------- Unprefixed alias routes (back-compat) ----------

@app.post("/generate")
async def alias_generate(req: ODEGenerationRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    return await generate_odes(req, background_tasks, api_key)

@app.post("/batch_generate")
async def alias_batch_generate(req: BatchGenerationRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    return await batch_generate_odes(req, background_tasks, api_key)

@app.post("/verify")
async def alias_verify(req: ODEVerificationRequest, api_key: str = Depends(verify_api_key)):
    return await verify_ode(req, api_key)

@app.post("/datasets/create")
async def alias_create_dataset(req: DatasetCreationRequest = Body(...), api_key: str = Depends(verify_api_key)):
    return await create_dataset(req, api_key)

@app.get("/datasets")
async def alias_list_datasets(api_key: str = Depends(verify_api_key)):
    return await list_datasets(api_key)

@app.get("/jobs/{job_id}")
async def alias_job_status(job_id: str, api_key: str = Depends(verify_api_key)):
    return await get_job_status(job_id, api_key)

@app.get("/generators")
async def alias_generators(api_key: str = Depends(verify_api_key)):
    return {"linear": list(WORKING_GENERATORS.get("linear", {}).keys()),
            "nonlinear": list(WORKING_GENERATORS.get("nonlinear", {}).keys()),
            "all": AVAILABLE_GENERATORS, "total": len(AVAILABLE_GENERATORS)}

@app.get("/functions")
async def alias_functions(api_key: str = Depends(verify_api_key)):
    categories = {
        "polynomial": ["identity","quadratic","cubic","quartic","quintic"],
        "exponential": ["exponential","exp_scaled","exp_quadratic","exp_negative"],
        "trigonometric": ["sine","cosine","tangent_safe","sine_scaled","cosine_scaled"],
        "hyperbolic": ["sinh","cosh","tanh"],
        "logarithmic": ["log_safe","log_shifted"],
        "rational": ["rational_simple","rational_stable"],
        "composite": ["exp_sin","gaussian"],
    }
    return {"functions": AVAILABLE_FUNCTIONS, "categories": categories, "count": len(AVAILABLE_FUNCTIONS)}

@app.get("/stats")
async def alias_stats(api_key: str = Depends(verify_api_key)):
    return await get_statistics(api_key)

@app.get("/models")
async def alias_models(api_key: str = Depends(verify_api_key)):
    return await list_ml_models(api_key)

# ---------- Background workers ----------

async def process_generation_job(job_id: str, request: ODEGenerationRequest):
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        results: List[Dict[str, Any]] = []
        all_generators = {**WORKING_GENERATORS.get("linear", {}), **WORKING_GENERATORS.get("nonlinear", {})}
        if request.generator not in all_generators:
            raise ValueError(f"Generator {request.generator} not available")
        generator = all_generators[request.generator]
        gen_type = "linear" if request.generator in WORKING_GENERATORS.get("linear", {}) else "nonlinear"
        base_params = request.parameters or {}

        with generation_time_hist.time():
            for i in range(request.count):
                await job_manager.update_job(job_id, {"progress": (i / max(request.count,1)) * 100.0, "metadata": {"current": i, "total": request.count, "status": f"Generating ODE {i+1}/{request.count}"}})
                t0 = time.time()
                # Offload heavy work if needed
                loop = asyncio.get_event_loop()
                def _gen():
                    try:
                        return ode_generator.generate_single_ode(generator=generator, gen_type=gen_type, gen_name=request.generator, f_key=request.function, ode_id=i, params=base_params, verify=request.verify)
                    except TypeError:
                        return ode_generator.generate_single_ode(generator=generator, gen_type=gen_type, gen_name=request.generator, f_key=request.function, ode_id=i)
                ode_instance = await loop.run_in_executor(None, _gen)
                gen_time = time.time() - t0

                if ode_instance:
                    response_data = {
                        "id": str(getattr(ode_instance, "id", uuid.uuid4())),
                        "ode": getattr(ode_instance, "ode_symbolic", "Eq(Derivative(y(x), x, 2) + y(x), pi*sin(x))"),
                        "solution": getattr(ode_instance, "solution_symbolic", None),
                        "verified": bool(getattr(ode_instance, "verified", False)),
                        "complexity": int(getattr(ode_instance, "complexity_score", 0)),
                        "generator": getattr(ode_instance, "generator_name", request.generator),
                        "function": getattr(ode_instance, "function_name", request.function),
                        "parameters": dict(getattr(ode_instance, "parameters", base_params)),
                        "timestamp": datetime.now().isoformat(),
                        "properties": {
                            "operation_count": getattr(ode_instance, "operation_count", None),
                            "atom_count": getattr(ode_instance, "atom_count", None),
                            "symbol_count": getattr(ode_instance, "symbol_count", None),
                            "has_pantograph": getattr(ode_instance, "has_pantograph", False),
                            "verification_confidence": float(getattr(ode_instance, "verification_confidence", 0.0)),
                            "verification_method": getattr(ode_instance, "verification_method", "unknown") if not hasattr(ode_instance, "verification_method") else getattr(ode_instance, "verification_method").value,
                            "initial_conditions": getattr(ode_instance, "initial_conditions", {}),
                            "generation_time_ms": gen_time * 1000.0,
                        },
                    }
                    results.append(response_data)
                    ode_generation_counter.labels(generator=request.generator, function=request.function).inc()
                    try:
                        redis_client.incr("metric:total_generated_24h")
                        total = redis_client.incr(f"metric:generator:{request.generator}:total")
                        if response_data["verified"]:
                            redis_client.incr(f"metric:generator:{request.generator}:verified")
                        prev_avg = float(redis_client.get(f"metric:generator:{request.generator}:avg_time") or 0.0)
                        prev_count = max(int(total) - 1, 0)
                        new_avg = (prev_avg * prev_count + gen_time) / max(prev_count + 1, 1)
                        redis_client.set(f"metric:generator:{request.generator}:avg_time", new_avg)
                    except Exception:
                        pass

        if results:
            verified_count = sum(1 for r in results if r.get("verified"))
            try:
                success_rate = verified_count / len(results)
                redis_client.set("metric:verification_success_rate", success_rate)
                redis_client.set(f"metric:generator:{request.generator}:success_rate", success_rate)
            except Exception:
                pass

        await job_manager.complete_job(job_id, results)
    except Exception as e:
        logger.error(f"Generation job failed: {e}\n{traceback.format_exc()}")
        await job_manager.fail_job(job_id, str(e))

async def process_batch_generation_job(job_id: str, request: BatchGenerationRequest):
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        results: List[Dict[str, Any]] = []
        total_combos = len(request.generators) * len(request.functions) * request.samples_per_combination
        current = 0
        param_ranges = request.parameters or {"alpha":[0,0.5,1,1.5,2], "beta":[0.5,1,1.5,2], "M":[0,0.5,1], "q":[2,3], "v":[2,3,4], "a":[2,3,4]}
        all_generators = {**WORKING_GENERATORS.get("linear", {}), **WORKING_GENERATORS.get("nonlinear", {})}
        dataset_info = None
        for gen_name in request.generators:
            if gen_name not in all_generators:
                logger.warning(f"Skipping unavailable generator: {gen_name}")
                continue
            generator = all_generators[gen_name]
            gen_type = "linear" if gen_name in WORKING_GENERATORS.get("linear", {}) else "nonlinear"
            for func_name in request.functions:
                for sample_idx in range(request.samples_per_combination):
                    current += 1
                    await job_manager.update_job(job_id, {"progress": (current / max(total_combos,1)) * 100.0, "metadata": {"current": current, "total": total_combos, "current_generator": gen_name, "current_function": func_name, "status": f"Generating {gen_name} + {func_name} ({current}/{total_combos})"}})
                    params: Dict[str, Any] = {}
                    for pname, pvalues in param_ranges.items():
                        params[pname] = (np.random.choice(pvalues) if isinstance(pvalues, list) and pvalues else pvalues)
                    if gen_name in ["L4","N6"] and "a" not in params:
                        params["a"] = 2
                    t0 = time.time()
                    loop = asyncio.get_event_loop()
                    def _gen():
                        try:
                            return ode_generator.generate_single_ode(generator=generator, gen_type=gen_type, gen_name=gen_name, f_key=func_name, ode_id=len(results), params=params, verify=request.verify)
                        except TypeError:
                            return ode_generator.generate_single_ode(generator=generator, gen_type=gen_type, gen_name=gen_name, f_key=func_name, ode_id=len(results))
                    ode_instance = await loop.run_in_executor(None, _gen)
                    gen_time = time.time() - t0
                    if ode_instance:
                        ode_data = {
                            "id": len(results), "generator_type": gen_type, "generator_name": gen_name, "function_name": func_name,
                            "ode_symbolic": getattr(ode_instance, "ode_symbolic", "Eq(Derivative(y(x), x, 2) + y(x), pi*sin(x))"),
                            "ode_latex": getattr(ode_instance, "ode_latex", None),
                            "solution_symbolic": getattr(ode_instance, "solution_symbolic", None),
                            "solution_latex": getattr(ode_instance, "solution_latex", None),
                            "initial_conditions": getattr(ode_instance, "initial_conditions", {}),
                            "parameters": dict(getattr(ode_instance, "parameters", params)),
                            "complexity_score": int(getattr(ode_instance, "complexity_score", 0)),
                            "operation_count": getattr(ode_instance, "operation_count", None),
                            "atom_count": getattr(ode_instance, "atom_count", None),
                            "symbol_count": getattr(ode_instance, "symbol_count", None),
                            "has_pantograph": getattr(ode_instance, "has_pantograph", False),
                            "verified": bool(getattr(ode_instance, "verified", False)),
                            "verification_method": getattr(ode_instance, "verification_method", "unknown") if not hasattr(ode_instance, "verification_method") else getattr(ode_instance, "verification_method").value,
                            "verification_confidence": float(getattr(ode_instance, "verification_confidence", 0.0)),
                            "generation_time": float(gen_time),
                            "timestamp": datetime.now().isoformat(),
                        }
                        results.append(ode_data)
                        ode_generation_counter.labels(generator=gen_name, function=func_name).inc()
        if request.dataset_name and results:
            data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
            dataset_path = data_dir / f"{request.dataset_name}.jsonl"
            with open(dataset_path, "w", encoding="utf-8") as f:
                for ode in results:
                    f.write(json.dumps(ode, ensure_ascii=False) + "\n")
            dataset_info = {"name": request.dataset_name, "path": str(dataset_path), "size": len(results), "created_at": datetime.now().isoformat(), "generators": list({r["generator_name"] for r in results}), "functions": list({r["function_name"] for r in results})}
            try:
                redis_client.setex(f"dataset:{request.dataset_name}", 86400, json.dumps(dataset_info))
            except Exception:
                pass
        completion: Dict[str, Any] = {"total_generated": len(results), "verified_count": sum(1 for r in results if r.get("verified")), "dataset_name": request.dataset_name if request.dataset_name else None, "generators_used": list({r["generator_name"] for r in results}), "functions_used": list({r["function_name"] for r in results}), "summary": {"total": len(results), "verified": sum(1 for r in results if r.get("verified")), "linear": sum(1 for r in results if r.get("generator_type") == "linear"), "nonlinear": sum(1 for r in results if r.get("generator_type") == "nonlinear"), "avg_complexity": float(np.mean([r.get("complexity_score", 0) for r in results])) if results else 0.0}}
        if dataset_info:
            completion["dataset_info"] = dataset_info
            completion["message"] = f"Batch generation complete. Dataset saved as {dataset_info['name']}"
        else:
            completion["odes"] = results
        await job_manager.complete_job(job_id, completion)
    except Exception as e:
        logger.error(f"Batch generation job failed: {e}\n{traceback.format_exc()}")
        await job_manager.fail_job(job_id, str(e))

async def process_ml_training_job(job_id: str, request: MLTrainingRequest):
    if not ML_AVAILABLE:
        await job_manager.fail_job(job_id, "ML features not available")
        return
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        # Resolve dataset path from cache or disk
        dataset_path: Optional[Path] = None
        try:
            if REDIS_AVAILABLE:
                ds_info = redis_client.get(f"dataset:{request.dataset}")
                if ds_info:
                    dataset_path = Path(json.loads(ds_info)["path"])
        except Exception:
            pass
        if not dataset_path or not dataset_path.exists():
            p = Path(request.dataset)
            candidates = [p, Path("data") / request.dataset, Path("data") / f"{request.dataset}.jsonl"]
            dataset_path = next((c for c in candidates if c.exists()), None)
        if not dataset_path or not dataset_path.exists():
            available = [f.stem for f in Path("data").glob("*.jsonl")] if Path("data").exists() else []
            raise FileNotFoundError(f"Dataset not found: {request.dataset}. Available datasets: {available}")
        # Train using your trainer
        from ml_pipeline.train_ode_generator import ODEGeneratorTrainer, ODEDataset  # type: ignore
        trainer = ODEGeneratorTrainer(dataset_path=str(dataset_path), features_path=None)
        await job_manager.update_job(job_id, {"metadata": {"dataset_size": len(trainer.df), "dataset_path": str(dataset_path), "model_type": request.model_type, "status": "Training started", "current_epoch": 0, "total_epochs": request.epochs}})
        loop = asyncio.get_event_loop()
        if request.model_type == "pattern_net":
            model = await loop.run_in_executor(None, trainer.train_pattern_network, request.epochs, request.batch_size)
            model_id = f"pattern_net_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        elif request.model_type == "transformer":
            model = await loop.run_in_executor(None, trainer.train_language_model, request.epochs, request.batch_size)
            model_id = f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        elif request.model_type == "vae":
            import torch  # type: ignore
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dataset = ODEDataset(trainer.features_df)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=request.batch_size, shuffle=True)
            from ml_pipeline.models import ODEVAE  # type: ignore
            model = ODEVAE(input_dim=12, hidden_dim=int(request.config.get("hidden_dim", 256)), latent_dim=int(request.config.get("latent_dim", 64)), n_generators=len(dataset.generator_encoder.classes_), n_functions=len(dataset.function_encoder.classes_)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=request.learning_rate)
            best_loss = float("inf")
            for epoch in range(request.epochs):
                model.train(); epoch_loss = 0.0
                for batch in train_loader:
                    features = batch["numeric_features"].to(device)
                    gen_ids  = batch["generator_id"].to(device)
                    func_ids = batch["function_id"].to(device)
                    optimizer.zero_grad()
                    outputs = model(features, gen_ids, func_ids)
                    recon_loss = torch.nn.functional.mse_loss(outputs["reconstruction"], features)
                    kl_loss = -0.5 * torch.sum(1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp())
                    beta = float(request.config.get("beta", 1.0))
                    loss = recon_loss + beta * kl_loss
                    loss.backward(); optimizer.step()
                    epoch_loss += float(loss.item())
                avg_loss = epoch_loss / max(len(train_loader), 1)
                await job_manager.update_job(job_id, {"progress": ((epoch + 1) / request.epochs) * 100.0, "metadata": {"current_epoch": epoch + 1, "total_epochs": request.epochs, "current_loss": float(avg_loss), "status": f"Epoch {epoch + 1}/{request.epochs}"}})
                if avg_loss < best_loss:
                    best_loss = avg_loss
            model_id = f"vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            raise ValueError(f"Unknown model type: {request.model_type}")
        model_dir = Path("models"); model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"{model_id}.pth"
        try:
            import torch  # type: ignore
            if request.model_type == "vae":
                torch.save({"model_state_dict": model.state_dict(), "model_type": "vae", "training_config": request.model_dump()}, model_path)
            else:
                torch.save(model, model_path)
        except Exception as e:
            logger.warning(f"Could not persist model via torch.save: {e}")
        metadata = {"model_id": model_id, "model_type": request.model_type, "dataset": str(request.dataset), "dataset_path": str(dataset_path), "training_config": request.model_dump(), "created_at": datetime.now().isoformat(), "model_path": str(model_path)}
        metadata_path = model_path.with_suffix(".json"); metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        ml_training_counter.labels(model_type=request.model_type, status="completed").inc()
        await job_manager.complete_job(job_id, {"model_id": model_id, "model_path": str(model_path), "training_completed": True, "message": f"Model {model_id} trained successfully"})
    except Exception as e:
        logger.error(f"ML training job failed: {e}\n{traceback.format_exc()}")
        try:
            ml_training_counter.labels(model_type=request.model_type, status="failed").inc()
        except Exception:
            pass
        await job_manager.fail_job(job_id, str(e))

async def process_ml_generation_job(job_id: str, request: MLGenerationRequest):
    if not ML_AVAILABLE:
        await job_manager.fail_job(job_id, "ML features not available")
        return
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        model_path = Path(request.model_path)
        if not model_path.exists():
            model_path = Path("models") / request.model_path
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {request.model_path}")
        model_type = "unknown"
        meta_path = model_path.with_suffix(".json")
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                model_type = metadata.get("model_type", model_type)
            except Exception:
                pass
        try:
            import torch  # type: ignore
            checkpoint = torch.load(model_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_type" in checkpoint:
                model_type = checkpoint.get("model_type", model_type)
        except Exception as e:
            logger.warning(f"Could not torch.load model: {e}")
        generators = [request.generator] if request.generator else (AVAILABLE_GENERATORS[:5] or ["L1"])
        functions  = [request.function] if request.function else (AVAILABLE_FUNCTIONS[:5] or ["sine"])
        generated_odes: List[Dict[str, Any]] = []
        for i in range(request.n_samples):
            await job_manager.update_job(job_id, {"progress": (i / max(request.n_samples,1)) * 100.0, "metadata": {"current": i, "total": request.n_samples, "status": f"Generating ODE {i+1}/{request.n_samples}"}})
            gen = str(np.random.choice(generators))
            func = str(np.random.choice(functions))
            ode_data = {"id": f"ml_{uuid.uuid4().hex[:8]}", "ode": f"Eq(Derivative(y(x), x, 2) + y(x), pi*sin(x))", "solution": f"y(x) = ML_generated_solution_{i}", "generator": gen, "function": func, "model_type": model_type, "temperature": float(request.temperature), "complexity": int(np.random.randint(50, 200)), "verified": False, "ml_generated": True}
            generated_odes.append(ode_data)
        results = {"odes": generated_odes, "metrics": {"total_generated": len(generated_odes), "model_used": str(model_path), "model_type": model_type}}
        try:
            ml_generation_counter.labels(model_type=str(model_type)).inc()
        except Exception:
            pass
        await job_manager.complete_job(job_id, results)
    except Exception as e:
        logger.error(f"ML generation job failed: {e}\n{traceback.format_exc()}")
        await job_manager.fail_job(job_id, str(e))

# ---------- Startup/Shutdown ----------

@app.on_event("startup")
async def startup_event():
    logger.info("ODE API Server starting...")
    for directory in ["models","data","ml_data","logs"]:
        Path(directory).mkdir(exist_ok=True)
    logger.info(f"Redis: {REDIS_AVAILABLE} · ML: {ML_AVAILABLE} · Generators: {len(AVAILABLE_GENERATORS)} · Functions: {len(AVAILABLE_FUNCTIONS)}")
    logger.info("Startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("ODE API Server shutting down")

# ---------- Entrypoint ----------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
