"""
Production FastAPI server for ODE Master Generators
- Starts fast and reliably on Railway
- API routes are registered BEFORE the SPA fallback (prevents HTML intercepts)
- Optional PUBLIC_READ for /api/generators and /api/functions
"""

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
from fastapi import FastAPI, HTTPException, Depends, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# ---------------- Environment / logging ----------------
PORT = int(os.getenv("PORT", "8080"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "dev-key,railway-key").split(",") if k.strip()]
ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "development")
ENABLE_WEBSOCKET = os.getenv("ENABLE_WEBSOCKET", "true").lower() == "true"
PUBLIC_READ = os.getenv("PUBLIC_READ", "false").lower() == "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("production_server")

# Ensure project root on path when executed as module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------- Optional project imports (with fallbacks) ----------------
try:
    from pipeline.generator import ODEDatasetGenerator
    from verification.verifier import ODEVerifier
    from utils.config import ConfigManager
    from utils.features import FeatureExtractor
    from core.types import GeneratorType, VerificationMethod, ODEInstance
    from core.functions import AnalyticFunctionLibrary
    CORE_OK = True
except Exception as e:
    log.warning(f"Core imports not available: {e}")

    CORE_OK = False

    class GeneratorType(Enum):
        LINEAR = "linear"
        NONLINEAR = "nonlinear"

    class VerificationMethod(Enum):
        SUBSTITUTION = "substitution"
        NUMERIC = "numeric"
        FAILED = "failed"
        PENDING = "pending"

# ML (optional)
try:
    import torch  # noqa
    from ml_pipeline.train_ode_generator import ODEGeneratorTrainer  # noqa
    from ml_pipeline.utils import prepare_ml_dataset, load_pretrained_model, generate_novel_odes  # noqa
    ML_OK = True
except Exception as e:
    ML_OK = False
    log.warning(f"ML pipeline not available: {e}")

# ---------------- Cache (Redis with fallback) ----------------
try:
    import redis  # noqa

    if REDIS_URL.startswith(("redis://", "rediss://")):
        _redis = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=3)
        _redis.ping()
        REDIS_OK = True
    else:
        raise ValueError("Invalid Redis URL")
except Exception as e:
    log.warning(f"Redis not available ({e}); using in-memory cache")

    REDIS_OK = False

    class MemCache:
        def __init__(self):
            self.data: Dict[str, str] = {}
            self.ttl: Dict[str, float] = {}

        def get(self, k: str) -> Optional[str]:
            exp = self.ttl.get(k)
            if exp and time.time() > exp:
                self.data.pop(k, None)
                self.ttl.pop(k, None)
            return self.data.get(k)

        def setex(self, k: str, seconds: int, v: str):
            self.data[k] = v
            self.ttl[k] = time.time() + seconds

        def set(self, k: str, v: str, ex: Optional[int] = None):
            self.data[k] = v
            if ex:
                self.ttl[k] = time.time() + ex

        def incr(self, k: str) -> int:
            v = int(self.data.get(k, "0")) + 1
            self.data[k] = str(v)
            return v

    _redis = MemCache()

# ---------------- Metrics ----------------
ode_generation_counter = Counter("ode_generation_total", "Total ODEs generated", ["generator", "function"])
verification_counter = Counter("ode_verification_total", "Total verifications", ["method", "result"])
generation_time_histogram = Histogram("ode_generation_duration_seconds", "ODE generation time")
active_jobs_gauge = Gauge("active_jobs", "Active jobs")
api_request_counter = Counter("api_requests_total", "API requests", ["endpoint", "method", "status"])
api_request_duration = Histogram("api_request_duration_seconds", "API request duration", ["endpoint"])
websocket_connections = Gauge("websocket_connections", "Active WebSocket connections")
dataset_size_gauge = Gauge("dataset_size_bytes", "Datasets total size bytes")

# ---------------- Pydantic base ----------------
class APIModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

# ---------------- Models (requests/responses) ----------------
class ODEGenerationRequest(APIModel):
    generator: str
    function: str
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
    method: str = "substitution"
    timeout: int = Field(30, ge=1, le=300)

class MLTrainingRequest(APIModel):
    dataset: str
    model_type: str = Field(..., description="pattern_net or transformer")
    epochs: int = Field(50, ge=1, le=1000)
    batch_size: int = Field(32, ge=8, le=256)
    learning_rate: float = Field(0.001, ge=1e-5, le=0.1)
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

# ---------------- Jobs ----------------
class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.queues: Dict[str, deque] = defaultdict(deque)

    async def create(self, job_type: str, params: Dict[str, Any], priority: int = 5) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        data = {
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
        self.jobs[job_id] = data
        self.queues[job_type].append(job_id)
        active_jobs_gauge.inc()
        if REDIS_OK:
            _redis.setex(f"job:{job_id}", 3600, json.dumps(data))
        return job_id

    async def update(self, job_id: str, updates: Dict[str, Any]):
        if job_id not in self.jobs:
            return
        self.jobs[job_id].update(updates)
        self.jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        if REDIS_OK:
            _redis.setex(f"job:{job_id}", 3600, json.dumps(self.jobs[job_id]))

    async def complete(self, job_id: str, results: Any):
        await self.update(job_id, {"status": "completed", "progress": 100.0, "results": results,
                                   "completed_at": datetime.utcnow().isoformat()})
        active_jobs_gauge.dec()

    async def fail(self, job_id: str, error: str):
        await self.update(job_id, {"status": "failed", "error": error, "completed_at": datetime.utcnow().isoformat()})
        active_jobs_gauge.dec()

    async def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        if job_id in self.jobs:
            return self.jobs[job_id]
        if REDIS_OK:
            raw = _redis.get(f"job:{job_id}")
            if raw:
                return json.loads(raw)
        return None

    async def list(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        items = list(self.jobs.values())
        if status:
            items = [j for j in items if j["status"] == status]
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items[:limit]

    def queue_stats(self) -> Dict[str, int]:
        return {t: len(q) for t, q in self.queues.items()}

# ---------------- WebSocket ----------------
class WSManager:
    def __init__(self):
        self.conn: Dict[str, WebSocket] = {}
        self.subs: Dict[str, Set[str]] = defaultdict(set)

    async def connect(self, ws: WebSocket, cid: str):
        await ws.accept()
        self.conn[cid] = ws
        websocket_connections.inc()

    def disconnect(self, cid: str):
        if cid in self.conn:
            del self.conn[cid]
        for t in list(self.subs.keys()):
            self.subs[t].discard(cid)
        websocket_connections.dec()

    async def broadcast(self, msg: str, topic: str = "general"):
        for cid in list(self.subs.get(topic, [])):
            ws = self.conn.get(cid)
            if not ws:
                continue
            try:
                await ws.send_text(msg)
            except Exception:
                self.disconnect(cid)

    def sub(self, cid: str, topic: str):
        self.subs[topic].add(cid)

    def unsub(self, cid: str, topic: str):
        self.subs[topic].discard(cid)

# ---------------- Core service ----------------
class ODEService:
    def __init__(self):
        self.config = ConfigManager() if CORE_OK else None
        self.generator = None
        self.verifier = None
        self.feature_extractor = None
        self.function_library: Dict[str, Any] = {}
        self.working: Dict[str, Dict[str, Any]] = {"linear": {}, "nonlinear": {}}

    async def init(self):
        """Initialize quickly. Defer heavy work off the hot path."""
        if not CORE_OK:
            # Demo mode
            self.working = {"linear": {"L1": None, "L2": None, "L3": None}, "nonlinear": {"N1": None, "N2": None}}
            self.function_library = {"sine": None, "cosine": None, "exponential": None, "quadratic": None}
            log.warning("Core unavailable: running in demo mode")
            return

        try:
            # Light init
            self.generator = ODEDatasetGenerator(config=self.config)
            self.verifier = ODEVerifier(self.config.config.get("verification", {}))
            self.feature_extractor = FeatureExtractor()
            self.function_library = AnalyticFunctionLibrary.get_safe_library()

            # Heavy generator self-test runs in background so the app can respond immediately
            asyncio.create_task(self._probe_generators())
        except Exception as e:
            log.error(f"ODE service init failed: {e}")
            # Keep demo mode usable
            self.working = {"linear": {"L1": None}, "nonlinear": {"N1": None}}
            self.function_library = {"sine": None, "exponential": None}

    async def _probe_generators(self):
        try:
            self.working = self.generator.test_generators()
            log.info(f"Generators ready: {len(self.working['linear'])} linear, "
                     f"{len(self.working['nonlinear'])} nonlinear")
        except Exception as e:
            log.error(f"Generator test failed: {e}")

    def generators(self) -> List[str]:
        return list(self.working["linear"].keys()) + list(self.working["nonlinear"].keys())

    def functions(self) -> List[str]:
        return list(self.function_library.keys())

    async def generate(self, generator: str, function: str, params: Optional[Dict] = None) -> Optional[Dict]:
        if not CORE_OK or not self.generator:
            # Demo ODE
            return {
                "id": str(uuid.uuid4()),
                "generator": generator,
                "function": function,
                "ode": f"Eq(Derivative(y(x), x, 2) + y(x), {function}(x))",
                "solution": f"{function}(x)",
                "verified": True,
                "parameters": params or {},
                "complexity": 42,
            }
        try:
            gtype = "linear" if generator in self.working["linear"] else "nonlinear"
            inst = self.generator.generate_single_ode(
                generator=self.working[gtype][generator],
                gen_type=gtype,
                gen_name=generator,
                f_key=function,
                ode_id=0,
            )
            if not inst:
                return None
            return {
                "id": str(inst.id),
                "generator": inst.generator_name,
                "function": inst.function_name,
                "ode": inst.ode_symbolic,
                "solution": inst.solution_symbolic,
                "verified": inst.verified,
                "complexity": inst.complexity_score,
                "parameters": inst.parameters,
                "verification_confidence": inst.verification_confidence,
            }
        except Exception as e:
            log.error(f"generate error: {e}")
            return None

    async def verify(self, ode: str, solution: str, method: str = "substitution") -> Dict:
        if not CORE_OK or not self.verifier:
            return {"verified": False, "method": method, "confidence": 0.0, "error": "verifier unavailable"}
        try:
            ode_expr = sp.sympify(ode)
            sol_expr = sp.sympify(solution)
            ok, ver_method, conf = self.verifier.verify(ode_expr, sol_expr)
            return {"verified": ok, "method": ver_method.value, "confidence": conf}
        except Exception as e:
            return {"verified": False, "method": method, "confidence": 0.0, "error": str(e)}

# ---------------- Lifespan ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Starting on port {PORT} (env={ENVIRONMENT}) | Redis: {REDIS_OK} | ML: {ML_OK}")
    app.state.jobs = JobManager()
    app.state.ws = WSManager()
    app.state.ode = ODEService()

    # ensure dirs
    for d in ("data", "models", "ml_data", "logs"):
        Path(d).mkdir(exist_ok=True)

    # initialize quickly then probe in background
    await app.state.ode.init()

    # background processors
    asyncio.create_task(_job_loop(app))
    asyncio.create_task(_metrics_loop(app))

    yield
    log.info("Shutting down...")

# ---------------- App ----------------
app = FastAPI(
    title="ODE Master Generators API",
    description="ODE generation, verification, datasets, jobs, ML",
    version="3.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Security helpers ----------------
api_key_hdr = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_key_or_public_read(request: Request, api_key: Optional[str] = Depends(api_key_hdr)):
    # Allow GET reads if PUBLIC_READ is enabled
    if PUBLIC_READ and request.method == "GET":
        return "public"
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(403, "Invalid or missing API key")
    return api_key

# ---------------- Middleware (metrics) ----------------
@app.middleware("http")
async def metrics_mw(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    dt = time.time() - t0
    api_request_counter.labels(endpoint=request.url.path, method=request.method, status=response.status_code).inc()
    api_request_duration.labels(endpoint=request.url.path).observe(dt)
    return response

# ---------------- Health / metrics / info ----------------
@app.get("/health", response_class=PlainTextResponse)
async def health():
    # Do not block: return quickly to keep Railway happy
    return "ok"

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/api/info")
async def info():
    s = app.state
    return {
        "name": "ODE Master Generators API",
        "version": "3.1.0",
        "environment": ENVIRONMENT,
        "services": {"core": CORE_OK, "ml": ML_OK, "cache": REDIS_OK, "websocket": ENABLE_WEBSOCKET},
        "public_read": PUBLIC_READ,
        "queues": s.jobs.queue_stats(),
    }

# ---------------- API: generation & verification ----------------
@app.get("/api/generators")
async def api_generators(_: str = Depends(require_key_or_public_read)):
    s = app.state.ode
    return {"linear": list(s.working["linear"].keys()),
            "nonlinear": list(s.working["nonlinear"].keys()),
            "all": s.generators(),
            "ready": bool(s.generators())}

@app.get("/api/functions")
async def api_functions(_: str = Depends(require_key_or_public_read)):
    s = app.state.ode
    return {"functions": s.functions(), "count": len(s.functions())}

class _GenReq(ODEGenerationRequest): ...
@app.post("/api/generate")
async def api_generate(req: _GenReq, _: str = Depends(require_key_or_public_read)):
    s = app.state.ode
    if req.generator not in s.generators():
        raise HTTPException(400, f"Unknown generator: {req.generator}")
    if req.function not in s.functions():
        raise HTTPException(400, f"Unknown function: {req.function}")

    job_id = await app.state.jobs.create("generation", req.dict())
    return {"job_id": job_id, "status": "queued", "check": f"/api/jobs/{job_id}"}

@app.post("/api/batch_generate")
async def api_batch(req: BatchGenerationRequest, _: str = Depends(require_key_or_public_read)):
    s = app.state.ode
    bad_g = [g for g in req.generators if g not in s.generators()]
    bad_f = [f for f in req.functions if f not in s.functions()]
    if bad_g:
        raise HTTPException(400, f"Unknown generators: {bad_g}")
    if bad_f:
        raise HTTPException(400, f"Unknown functions: {bad_f}")
    job_id = await app.state.jobs.create("batch_generation", req.dict(), priority=10)
    total = len(req.generators) * len(req.functions) * req.samples_per_combination
    return {"job_id": job_id, "status": "queued", "total_expected": total, "check": f"/api/jobs/{job_id}"}

@app.post("/api/verify")
async def api_verify(req: ODEVerificationRequest, _: str = Depends(require_key_or_public_read)):
    res = await app.state.ode.verify(req.ode, req.solution, req.method)
    verification_counter.labels(method=res.get("method", req.method),
                                result="success" if res.get("verified") else "failed").inc()
    return res

# ---------------- API: datasets ----------------
@app.get("/api/datasets")
async def api_list_datasets(_: str = Depends(require_key_or_public_read)):
    data = []
    p = Path("data")
    if p.exists():
        for f in p.glob("*.jsonl"):
            st = f.stat()
            meta = None
            if REDIS_OK:
                raw = _redis.get(f"dataset:{f.stem}")
                if raw:
                    meta = json.loads(raw)
            data.append({
                "name": f.stem,
                "path": str(f),
                "size_bytes": st.st_size,
                "created_at": datetime.fromtimestamp(st.st_ctime).isoformat(),
                "metadata": meta,
            })
    return {"datasets": data, "count": len(data)}

@app.post("/api/datasets/create")
async def api_create_dataset(odes: List[Dict[str, Any]], name: Optional[str] = None,
                             _: str = Depends(require_key_or_public_read)):
    if not name:
        name = f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    fp = Path("data") / f"{name}.jsonl"
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w") as fh:
        for ode in odes:
            fh.write(json.dumps(ode) + "\n")

    verified = sum(1 for o in odes if o.get("verified"))
    rate = (verified / len(odes)) if odes else 0.0
    meta = {"name": name, "path": str(fp), "size": len(odes),
            "verified_count": verified, "verification_rate": rate,
            "created_at": datetime.utcnow().isoformat()}
    if REDIS_OK:
        _redis.setex(f"dataset:{name}", 86400, json.dumps(meta))

    try:
        dataset_size_gauge.inc(fp.stat().st_size)
    except Exception:
        pass

    return meta

@app.get("/api/datasets/{name}/download")
async def api_download_dataset(name: str, format: str = "jsonl", _: str = Depends(require_key_or_public_read)):
    fp = Path("data") / f"{name}.jsonl"
    if not fp.exists():
        raise HTTPException(404, "Dataset not found")
    if format == "jsonl":
        return FileResponse(fp, filename=f"{name}.jsonl")
    if format == "csv":
        df = pd.read_json(fp, lines=True)
        out = Path("/tmp") / f"{name}.csv"
        df.to_csv(out, index=False)
        return FileResponse(out, filename=f"{name}.csv")
    raise HTTPException(400, f"Unsupported format: {format}")

# ---------------- API: jobs ----------------
@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str, _: str = Depends(require_key_or_public_read)):
    job = await app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatus(
        job_id=job["id"], status=job["status"], progress=job["progress"],
        results=job.get("results"), error=job.get("error"),
        created_at=job["created_at"], updated_at=job["updated_at"],
        eta=_eta(job) if job["status"] == "running" else None,
        metadata=job.get("metadata", {}),
    )

@app.get("/api/jobs")
async def api_jobs(status: Optional[str] = None, limit: int = 100, _: str = Depends(require_key_or_public_read)):
    items = await app.state.jobs.list(status=status, limit=limit)
    return {
        "jobs": [
            JobStatus(job_id=j["id"], status=j["status"], progress=j["progress"],
                      results=None, error=j.get("error"),
                      created_at=j["created_at"], updated_at=j["updated_at"],
                      metadata=j.get("metadata", {}))
            for j in items
        ],
        "count": len(items),
        "queue_stats": app.state.jobs.queue_stats(),
    }

@app.delete("/api/jobs/{job_id}")
async def api_cancel(job_id: str, _: str = Depends(require_key_or_public_read)):
    job = await app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] in ("completed", "failed"):
        raise HTTPException(400, f"Cannot cancel {job['status']} job")
    await app.state.jobs.update(job_id, {"status": "cancelled", "error": "Cancelled by user"})
    return {"message": "Job cancelled"}

# ---------------- WebSocket ----------------
if ENABLE_WEBSOCKET:
    @app.websocket("/ws/{client_id}")
    async def ws_endpoint(ws: WebSocket, client_id: str):
        await app.state.ws.connect(ws, client_id)
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                t = msg.get("type")
                if t == "subscribe":
                    app.state.ws.sub(client_id, msg.get("topic", "general"))
                    await ws.send_text(json.dumps({"type": "subscribed"}))
                elif t == "unsubscribe":
                    app.state.ws.unsub(client_id, msg.get("topic", "general"))
                elif t == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
        except WebSocketDisconnect:
            app.state.ws.disconnect(client_id)

# ---------------- Background processors ----------------
async def _job_loop(app: FastAPI):
    while True:
        try:
            for jtype, q in list(app.state.jobs.queues.items()):
                if not q:
                    continue
                job_id = q.popleft()
                job = await app.state.jobs.get(job_id)
                if not job or job["status"] != "queued":
                    continue

                if jtype == "generation":
                    asyncio.create_task(_run_generate(app, job_id, ODEGenerationRequest(**job["params"])))
                elif jtype == "batch_generation":
                    asyncio.create_task(_run_batch(app, job_id, BatchGenerationRequest(**job["params"])))
                else:
                    log.warning(f"Unknown job type {jtype}")
            await asyncio.sleep(0.5)
        except Exception as e:
            log.error(f"job loop error: {e}")
            await asyncio.sleep(2)

async def _metrics_loop(app: FastAPI):
    while True:
        try:
            if REDIS_OK:
                active = sum(1 for j in app.state.jobs.jobs.values() if j["status"] in ("queued", "running"))
                _redis.set("metrics:active_jobs", str(active))
            await asyncio.sleep(60)
        except Exception as e:
            log.error(f"metrics loop error: {e}")

async def _run_generate(app: FastAPI, job_id: str, req: ODEGenerationRequest):
    try:
        await app.state.jobs.update(job_id, {"status": "running", "started_at": datetime.utcnow().isoformat()})
        results = []
        with generation_time_histogram.time():
            for i in range(req.count):
                await app.state.jobs.update(job_id, {"progress": i / max(1, req.count) * 100.0,
                                                     "metadata": {"current": i, "total": req.count}})
                ode = await app.state.ode.generate(req.generator, req.function, req.parameters)
                if not ode:
                    continue
                if req.verify:
                    ode["verification"] = await app.state.ode.verify(ode["ode"], ode["solution"])
                results.append(ode)
                ode_generation_counter.labels(generator=req.generator, function=req.function).inc()
        if REDIS_OK:
            _redis.incr("stats:generated_24h")
            if results:
                v = sum(1 for r in results if r.get("verified"))
                _redis.set("stats:verification_rate", str(v / len(results)))
        await app.state.jobs.complete(job_id, results)
    except Exception as e:
        log.error(f"generation job failed: {e}\n{traceback.format_exc()}")
        await app.state.jobs.fail(job_id, str(e))

async def _run_batch(app: FastAPI, job_id: str, req: BatchGenerationRequest):
    try:
        await app.state.jobs.update(job_id, {"status": "running"})
        results = []
        total = len(req.generators) * len(req.functions) * req.samples_per_combination
        cur = 0
        for g in req.generators:
            for f in req.functions:
                for _ in range(req.samples_per_combination):
                    cur += 1
                    await app.state.jobs.update(job_id, {"progress": cur / max(1, total) * 100.0,
                                                         "metadata": {"current": cur, "total": total,
                                                                      "generator": g, "function": f}})
                    params: Dict[str, Any] = {}
                    if req.parameter_ranges:
                        for k, vals in req.parameter_ranges.items():
                            if isinstance(vals, list) and vals:
                                params[k] = np.random.choice(vals)
                    ode = await app.state.ode.generate(g, f, params)
                    if not ode:
                        continue
                    if req.verify:
                        ode["verification"] = await app.state.ode.verify(ode["ode"], ode["solution"])
                    results.append(ode)

        ds_name = None
        if req.save_dataset and results:
            ds_name = req.dataset_name or f"batch_{job_id[:8]}"
            out = Path("data") / f"{ds_name}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as fh:
                for r in results:
                    fh.write(json.dumps(r) + "\n")
            if REDIS_OK:
                _redis.setex(f"dataset:{ds_name}", 86400, json.dumps({
                    "name": ds_name, "path": str(out), "size": len(results),
                    "generators": req.generators, "functions": req.functions,
                    "created_at": datetime.utcnow().isoformat()
                }))

        await app.state.jobs.complete(job_id, {
            "total_generated": len(results),
            "dataset_name": ds_name,
            "summary": {
                "verified": sum(1 for r in results if r.get("verified")),
                "generators": list({r["generator"] for r in results}),
                "functions": list({r["function"] for r in results}),
            },
        })
    except Exception as e:
        log.error(f"batch job failed: {e}")
        await app.state.jobs.fail(job_id, str(e))

# ---------------- Utils ----------------
def _eta(job: Dict[str, Any]) -> Optional[str]:
    if job.get("status") != "running" or job.get("progress", 0) <= 0:
        return None
    start = datetime.fromisoformat(job.get("started_at", job["created_at"]))
    elapsed = (datetime.utcnow() - start).total_seconds()
    prog = job["progress"] / 100.0
    remaining = elapsed * (1 - prog) / max(prog, 1e-6)
    return (datetime.utcnow() + timedelta(seconds=remaining)).isoformat()

# ---------------- GUI (mount AFTER API routes) ----------------
GUI_DIR_ENV = os.getenv("GUI_BUNDLE_DIR", "").strip()
_candidates: List[Path] = []
if GUI_DIR_ENV:
    _c = Path(GUI_DIR_ENV)
    if (_c / "index.html").exists():
        _candidates.append(_c)
repo_root = Path(__file__).resolve().parents[1]
for p in (repo_root / "ode_gui_bundle",
          repo_root / "ode_gui_bundle" / "dist",
          repo_root / "ode_gui_bundle" / "build",
          repo_root / "gui" / "gui" / "dist"):
    if (p / "index.html").exists():
        _candidates.append(p)

_GUI: Optional[Path] = next(iter(_candidates), None)
if _GUI:
    assets = _GUI / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")
    else:
        log.warning(f"GUI found at {_GUI} but no /assets dir")

    @app.get("/config.js")
    async def config_js():
        content = "window.ODE_CONFIG=" + json.dumps({
            "API_BASE": os.getenv("PUBLIC_API_BASE", ""),  # leave blank to use window.location.origin
            "API_KEY": os.getenv("PUBLIC_API_KEY", ""),    # only if intentionally public
            "WS": ENABLE_WEBSOCKET
        }) + ";"
        return Response(content, media_type="application/javascript")

    @app.get("/", response_class=HTMLResponse)
    async def spa_root():
        return FileResponse(_GUI / "index.html")

    @app.get("/{path:path}", response_class=HTMLResponse)
    async def spa_fallback(path: str):
        f = _GUI / path
        if f.exists() and f.is_file():
            return FileResponse(f)
        return FileResponse(_GUI / "index.html")
else:
    log.warning("GUI bundle not found. Set GUI_BUNDLE_DIR to the folder containing index.html.")

# ---------------- Local run ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info" if ENVIRONMENT == "production" else "debug")
