# production_server.py
"""
Production FastAPI server for ODE generation, verification, datasets, and ML hooks.

NO baked-in /api/v1 paths.
- Core routes are served at plain paths: /generate, /verify, /datasets, etc.
- Optional: set ADD_V1_ALIAS=1 to ALSO expose /api/v1/* compatibility aliases.

Other env:
  API_KEYS="test-key,dev-key"
  REDIS_URL=redis://localhost:6379
  USE_DEMO=0|1         # if your generator stack can't initialize, provide demo lists
  ADD_V1_ALIAS=0|1     # add /api/v1 aliases for all routes
"""

import os
import json
import time
import uuid
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import sympy as sp
from fastapi import FastAPI, APIRouter, HTTPException, Depends, BackgroundTasks, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict

# Observability
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Optional cache
try:
    import redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    if REDIS_URL.startswith(("redis://","rediss://")):
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    else:
        redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False
    class _Mem:
        def __init__(self): self.k = {}
        def set(self, a,b): self.k[a]=b
        def setex(self,a,ttl,b): self.k[a]=b
        def get(self,a): return self.k.get(a)
        def incr(self,a): v=int(self.k.get(a,"0")); v+=1; self.k[a]=str(v); return v
        def keys(self,p="*"): return list(self.k.keys())
    redis_client = _Mem()

# Optional ML (soft import)
ML_AVAILABLE = True
try:
    import torch  # noqa
except Exception:
    ML_AVAILABLE = False

# Import your project parts (make soft/fallbacks if needed)
try:
    # Plug your real implementations here
    # from pipeline.generator import ODEDatasetGenerator
    # from verification.verifier import ODEVerifier
    # from core.functions import AnalyticFunctionLibrary
    class DummyVerifier:
        def verify(self, ode, sol, method="substitution"):
            res = sp.simplify(sp.Eq(ode.lhs.subs({sp.Function("y")(sp.Symbol("x")): sp.sympify(sol)}), ode.rhs))
            return bool(res), method, 1.0 if res else 0.0
    class DummyGenerator:
        def test_generators(self):
            return {"linear": {"L1": object(), "L2": object()}, "nonlinear": {"N1": object(), "N2": object()}}
        def generate_single_ode(self, **kwargs):
            # This should return an object with these attributes in your real code
            class Obj:
                id = uuid.uuid4()
                ode_symbolic = "Eq(Derivative(y(x), x, 2) + y(x), pi*sin(x))"
                solution_symbolic = "pi*sin(x)"
                verified = True
                complexity_score = 120
                generator_name = kwargs.get("gen_name","L1")
                function_name = kwargs.get("f_key","sine")
                parameters = kwargs.get("params", {})
                verification_confidence = 1.0
                verification_method = "substitution"
                initial_conditions = {}
                operation_count = 0; atom_count = 0; symbol_count = 0
                has_pantograph = False
            return Obj()
    class DummyFuncLib(dict):
        def __init__(self): super().__init__({"sine":None,"cosine":None,"exponential":None})
    ode_generator = DummyGenerator()
    ode_verifier = DummyVerifier()
    function_lib = DummyFuncLib()
except Exception:
    ode_generator = None
    ode_verifier = None
    function_lib = {}

USE_DEMO = os.getenv("USE_DEMO","0") in {"1","true","True"}
ADD_V1_ALIAS = os.getenv("ADD_V1_ALIAS","0") in {"1","true","True"}

# Security
VALID_API_KEYS = [k.strip() for k in os.getenv("API_KEYS","test-key,dev-key").split(",") if k.strip()]
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)):
    if not api_key: raise HTTPException(status_code=403, detail="API key required. Add 'X-API-Key' header.")
    if api_key not in VALID_API_KEYS: raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# FastAPI app
app = FastAPI(title="ODE Generation API", version="3.0.0", description="Prefix-agnostic ODE API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

# Metrics
ode_generation_counter = Counter("ode_generation_total", "Total ODEs generated", ["generator","function"])
verification_counter  = Counter("ode_verification_total", "Total verifications", ["method","result"])
generation_time_hist  = Histogram("ode_generation_duration_seconds", "ODE generation time")
active_jobs_gauge     = Gauge("active_jobs", "Number of active jobs")
api_req_counter       = Counter("api_requests_total", "Total API requests", ["endpoint","method","status"])
api_req_duration      = Histogram("api_request_duration_seconds", "API request duration", ["endpoint"])

# ---- Jobs ----
class JobStore:
    def __init__(self): self.jobs: Dict[str, Dict[str,Any]] = {}
    async def create(self, kind: str, params: Dict[str,Any]) -> str:
        jid = str(uuid.uuid4()); now = datetime.utcnow().isoformat()
        data = {"id": jid, "type": kind, "params": params, "status":"pending","progress":0.0,"created_at":now,"updated_at":now}
        self.jobs[jid]=data; active_jobs_gauge.inc(); return jid
    async def update(self, jid: str, updates: Dict[str,Any]): 
        if jid in self.jobs: self.jobs[jid].update(updates); self.jobs[jid]["updated_at"]=datetime.utcnow().isoformat()
    async def complete(self, jid: str, results: Any):
        await self.update(jid, {"status":"completed","progress":100.0,"results":results}); active_jobs_gauge.dec()
    async def fail(self, jid: str, err: str):
        await self.update(jid, {"status":"failed","error":err}); active_jobs_gauge.dec()
    async def get(self, jid: str) -> Optional[Dict[str,Any]]: return self.jobs.get(jid)
jobs = JobStore()

# ---- Models ----
class ODEGenerationRequest(BaseModel):
    generator: str
    function: str
    parameters: Optional[Dict[str, float]] = None
    count: int = Field(1, ge=1, le=100)
    verify: bool = True

class BatchGenerationRequest(BaseModel):
    generators: List[str]
    functions: List[str]
    samples_per_combination: int = Field(5, ge=1, le=50)
    parameters: Optional[Dict[str, List[float]]] = None
    verify: bool = True
    dataset_name: Optional[str] = None

class ODEVerificationRequest(BaseModel):
    ode: str
    solution: str
    method: str = "substitution"

class DatasetCreationRequest(BaseModel):
    odes: List[Dict[str, Any]]
    dataset_name: Optional[str] = None

class MLTrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset: str
    model_type: str
    epochs: int = Field(50, ge=1, le=1000)
    batch_size: int = Field(32, ge=8, le=256)
    learning_rate: float = Field(0.001, ge=0.00001, le=0.1)
    early_stopping: bool = True
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MLGenerationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_path: str
    n_samples: int = Field(10, ge=1, le=1000)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    generator: Optional[str] = None
    function: Optional[str] = None
    complexity_range: Optional[List[int]] = None

class JobStatus(BaseModel):
    job_id: str; status: str; progress: float
    results: Optional[Any] = None; error: Optional[str] = None
    created_at: str; updated_at: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

# ---- Generators init ----
WORKING_GENERATORS = {"linear": {}, "nonlinear": {}}
AVAILABLE_GENS: List[str] = []
AVAILABLE_FUNCS: List[str] = []

try:
    WORKING_GENERATORS = (ode_generator.test_generators() if ode_generator else {"linear": {}, "nonlinear": {}})
    AVAILABLE_GENS = list(WORKING_GENERATORS.get("linear", {}).keys()) + list(WORKING_GENERATORS.get("nonlinear", {}).keys())
    AVAILABLE_FUNCS = list(function_lib.keys()) if function_lib else ["sine","cosine","exponential"]
except Exception:
    pass

if USE_DEMO and (not AVAILABLE_GENS or not AVAILABLE_FUNCS):
    AVAILABLE_GENS = ["L1","L2","L3","L4","N1","N2","N3","N4","N5","N6","N7"]
    AVAILABLE_FUNCS = ["sine","cosine","tangent_safe","exponential","exp_scaled","quadratic","cubic","sinh","cosh","tanh","log_safe"]

# ---- Middleware for metrics ----
@app.middleware("http")
async def _metrics(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    dt = time.time() - t0
    try:
        api_req_counter.labels(endpoint=request.url.path, method=request.method, status=resp.status_code).inc()
        api_req_duration.labels(endpoint=request.url.path).observe(dt)
    except Exception:
        pass
    return resp

# ---- Discovery & health (no auth) ----
@app.get("/")
async def root():
    # describe unprefixed routes; GUI will probe this
    return {
        "name": "ODE Generation API",
        "version": "3.0.0",
        "description": "Prefix-agnostic ODE API",
        "ml_enabled": ML_AVAILABLE,
        "redis_enabled": REDIS_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "api": {
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
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "generators": len(AVAILABLE_GENS), "functions": len(AVAILABLE_FUNCS), "ml_enabled": ML_AVAILABLE}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

# ---- Protected router (no prefix baked) ----
router = APIRouter(dependencies=[Depends(verify_api_key)])

@router.get("/generators")
async def list_generators():
    return {"linear": list(WORKING_GENERATORS.get("linear", {}).keys()), "nonlinear": list(WORKING_GENERATORS.get("nonlinear", {}).keys()), "all": AVAILABLE_GENS, "total": len(AVAILABLE_GENS)}

@router.get("/functions")
async def list_functions():
    cats = {
        "polynomial": ["identity","quadratic","cubic","quartic","quintic"],
        "exponential": ["exponential","exp_scaled","exp_quadratic","exp_negative"],
        "trigonometric": ["sine","cosine","tangent_safe","sine_scaled","cosine_scaled"],
        "hyperbolic": ["sinh","cosh","tanh"],
        "logarithmic": ["log_safe","log_shifted"],
        "rational": ["rational_simple","rational_stable"],
        "composite": ["exp_sin","gaussian"],
    }
    return {"functions": AVAILABLE_FUNCS, "categories": cats, "count": len(AVAILABLE_FUNCS)}

@router.post("/generate")
async def generate_odes(request: ODEGenerationRequest, bg: BackgroundTasks):
    if request.generator not in AVAILABLE_GENS: raise HTTPException(400, f"Unknown generator: {request.generator}")
    if request.function  not in AVAILABLE_FUNCS: raise HTTPException(400, f"Unknown function: {request.function}")
    jid = await jobs.create("generation", request.model_dump())
    bg.add_task(_job_generate, jid, request)
    return {"job_id": jid, "status": "Job created", "check_status_url": f"/jobs/{jid}"}

@router.post("/batch_generate")
async def batch_generate_odes(request: BatchGenerationRequest, bg: BackgroundTasks):
    bad_g = [g for g in request.generators if g not in AVAILABLE_GENS]
    bad_f = [f for f in request.functions if f not in AVAILABLE_FUNCS]
    if bad_g: raise HTTPException(400, f"Unknown generators: {bad_g}")
    if bad_f: raise HTTPException(400, f"Unknown functions: {bad_f}")
    jid = await jobs.create("batch_generation", request.model_dump())
    bg.add_task(_job_batch_generate, jid, request)
    return {"job_id": jid, "status": "Batch job created", "total_expected": len(request.generators)*len(request.functions)*request.samples_per_combination, "check_status_url": f"/jobs/{jid}"}

@router.post("/verify")
async def verify_ode(request: ODEVerificationRequest):
    try:
        ode_expr = sp.sympify(request.ode)
        sol_expr = sp.sympify(request.solution)
        verified, method_used, conf = ode_verifier.verify(ode_expr, sol_expr, method=request.method) if ode_verifier else (False, "none", 0.0)
        verification_counter.labels(method=str(method_used), result="success" if verified else "failed").inc()
        return {"verified": bool(verified), "confidence": float(conf), "method": str(method_used), "details": {"ode": str(ode_expr), "solution": str(sol_expr)}}
    except Exception as e:
        raise HTTPException(400, str(e))

@router.post("/datasets/create")
async def create_dataset(request: DatasetCreationRequest = Body(...)):
    name = request.dataset_name or f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    Path("data").mkdir(exist_ok=True)
    path = Path("data")/f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in request.odes: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    info = {"name": name, "path": str(path), "size": len(request.odes), "created_at": datetime.utcnow().isoformat()}
    try: redis_client.setex(f"dataset:{name}", 86400, json.dumps(info))
    except Exception: pass
    return {"dataset_name": name, "path": str(path), "size": len(request.odes), "message": "Dataset created successfully"}

@router.get("/datasets")
async def list_datasets():
    out: List[Dict[str,Any]] = []
    # redis
    try:
        for k in getattr(redis_client,"keys",lambda *_:[])("dataset:*"):
            v = redis_client.get(k)
            if v: out.append(json.loads(v))
    except Exception: pass
    # fs
    p = Path("data")
    if p.exists():
        for fp in p.glob("*.jsonl"):
            if not any(d.get("path")==str(fp) for d in out):
                stat = fp.stat()
                try:
                    with open(fp,"r",encoding="utf-8") as f: n = sum(1 for line in f if line.strip())
                except Exception:
                    n = 0
                out.append({"name": fp.stem, "path": str(fp), "size": n, "file_size_bytes": stat.st_size, "created_at": datetime.utcfromtimestamp(stat.st_ctime).isoformat()})
    return {"datasets": out, "count": len(out)}

@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    j = await jobs.get(job_id)
    if not j: raise HTTPException(404, "Job not found")
    return JobStatus(job_id=j["id"], status=j["status"], progress=j["progress"], results=j.get("results"), error=j.get("error"), created_at=j["created_at"], updated_at=j["updated_at"], metadata=j.get("metadata", {}))

@router.post("/ml/train")
async def ml_train(request: MLTrainingRequest, bg: BackgroundTasks):
    if not ML_AVAILABLE: raise HTTPException(503, "ML not available")
    jid = await jobs.create("ml_training", request.model_dump())
    bg.add_task(_job_ml_train, jid, request)
    return {"job_id": jid, "status": "Training job created", "check_status_url": f"/jobs/{jid}"}

@router.post("/ml/generate")
async def ml_generate(request: MLGenerationRequest, bg: BackgroundTasks):
    if not ML_AVAILABLE: raise HTTPException(503, "ML not available")
    jid = await jobs.create("ml_generation", request.model_dump())
    bg.add_task(_job_ml_generate, jid, request)
    return {"job_id": jid, "status": "ML generation job created", "check_status_url": f"/jobs/{jid}"}

@router.get("/models")
async def list_models():
    models_dir = Path("models"); models: List[Dict[str,Any]] = []
    if models_dir.exists():
        for p in models_dir.glob("*.pth"):
            meta = (p.with_suffix(".json").read_text(encoding="utf-8") if p.with_suffix(".json").exists() else None)
            models.append({"path": str(p), "name": p.stem, "size": p.stat().st_size, "created": datetime.utcfromtimestamp(p.stat().st_ctime).isoformat(), "metadata": (json.loads(meta) if meta else {})})
    return {"models": models, "count": len(models), "ml_enabled": ML_AVAILABLE}

@router.get("/stats")
async def stats():
    total_generated = int(redis_client.get("metric:total_generated_24h") or 0)
    verification_success_rate = float(redis_client.get("metric:verification_success_rate") or 0.0)
    return {"status": "operational", "total_generated_24h": total_generated, "verification_success_rate": verification_success_rate, "active_jobs": sum(1 for j in jobs.jobs.values() if j["status"] in ("pending","running")), "available_generators": len(AVAILABLE_GENS), "available_functions": len(AVAILABLE_FUNCS), "ml_enabled": ML_AVAILABLE, "redis_enabled": REDIS_AVAILABLE}

# Mount router (no prefix)
app.include_router(router)

# Optional compatibility aliases at /api/v1/*
if ADD_V1_ALIAS:
    app.include_router(router, prefix="/api/v1")

# ---- Background job processors ----
async def _job_generate(job_id: str, req: ODEGenerationRequest):
    try:
        await jobs.update(job_id, {"status":"running"})
        results: List[Dict[str,Any]] = []
        all_gens = {**WORKING_GENERATORS.get("linear",{}), **WORKING_GENERATORS.get("nonlinear",{})}
        if req.generator not in all_gens: raise ValueError(f"Generator {req.generator} not available")
        base_params = req.parameters or {}
        with generation_time_hist.time():
            for i in range(req.count):
                await jobs.update(job_id, {"progress": (i/req.count)*100.0, "metadata": {"current": i, "total": req.count}})
                t0 = time.time()
                inst = ode_generator.generate_single_ode(gen_type=("linear" if req.generator in WORKING_GENERATORS.get("linear",{}) else "nonlinear"), gen_name=req.generator, f_key=req.function, ode_id=i, params=base_params, verify=req.verify) if ode_generator else None
                dt = time.time() - t0
                if inst:
                    ode_generation_counter.labels(generator=req.generator, function=req.function).inc()
                    results.append({
                        "id": str(getattr(inst,"id",uuid.uuid4())),
                        "ode": getattr(inst,"ode_symbolic",""),
                        "solution": getattr(inst,"solution_symbolic",None),
                        "verified": bool(getattr(inst,"verified",False)),
                        "complexity": int(getattr(inst,"complexity_score",0)),
                        "generator": getattr(inst,"generator_name",req.generator),
                        "function": getattr(inst,"function_name",req.function),
                        "parameters": dict(getattr(inst,"parameters", base_params)),
                        "timestamp": datetime.utcnow().isoformat(),
                        "properties": {"verification_confidence": float(getattr(inst,"verification_confidence",0.0)), "verification_method": str(getattr(inst,"verification_method","unknown")), "generation_time_ms": dt*1000.0},
                    })
        if results:
            ver = sum(1 for r in results if r.get("verified"))
            try:
                redis_client.set("metric:verification_success_rate", ver/len(results))
                redis_client.incr("metric:total_generated_24h")
            except Exception: pass
        await jobs.complete(job_id, results)
    except Exception as e:
        await jobs.fail(job_id, f"Generation job failed: {e}\n{traceback.format_exc()}")

async def _job_batch_generate(job_id: str, req: BatchGenerationRequest):
    try:
        await jobs.update(job_id, {"status":"running"})
        results: List[Dict[str,Any]] = []
        total = len(req.generators)*len(req.functions)*req.samples_per_combination
        current = 0
        pr = req.parameters or {"alpha":[0,1],"beta":[1,2],"M":[0],"q":[2,3],"v":[2,3],"a":[2]}
        all_gens = {**WORKING_GENERATORS.get("linear",{}), **WORKING_GENERATORS.get("nonlinear",{})}
        for g in req.generators:
            if g not in all_gens: continue
            for f in req.functions:
                for s in range(req.samples_per_combination):
                    current += 1
                    await jobs.update(job_id, {"progress": (current/max(total,1))*100.0, "metadata": {"current": current, "total": total, "current_generator": g, "current_function": f}})
                    params = {}
                    for k, vals in pr.items():
                        params[k] = (np.random.choice(vals) if isinstance(vals, list) and vals else vals)
                    try:
                        inst = ode_generator.generate_single_ode(gen_type=("linear" if g in WORKING_GENERATORS.get("linear",{}) else "nonlinear"), gen_name=g, f_key=f, ode_id=len(results), params=params, verify=req.verify) if ode_generator else None
                    except TypeError:
                        inst = None
                    if inst:
                        results.append({"id": len(results), "generator_type": ("linear" if g in WORKING_GENERATORS.get("linear",{}) else "nonlinear"), "generator_name": g, "function_name": f, "ode_symbolic": getattr(inst,"ode_symbolic",""), "solution_symbolic": getattr(inst,"solution_symbolic",None), "parameters": getattr(inst,"parameters", params), "complexity_score": int(getattr(inst,"complexity_score",0)), "verified": bool(getattr(inst,"verified",False)), "generation_time": 0.0, "timestamp": datetime.utcnow().isoformat()})
                        ode_generation_counter.labels(generator=g, function=f).inc()
        out: Dict[str,Any] = {"total_generated": len(results), "verified_count": sum(1 for r in results if r.get("verified")), "generators_used": list({r["generator_name"] for r in results}), "functions_used": list({r["function_name"] for r in results}), "summary": {"total": len(results), "verified": sum(1 for r in results if r.get("verified")), "linear": sum(1 for r in results if r.get("generator_type")=="linear"), "nonlinear": sum(1 for r in results if r.get("generator_type")=="nonlinear"), "avg_complexity": float(np.mean([r.get("complexity_score",0) for r in results])) if results else 0.0}}
        if req.dataset_name and results:
            Path("data").mkdir(exist_ok=True)
            path = Path("data")/f"{req.dataset_name}.jsonl"
            with open(path,"w",encoding="utf-8") as f:
                for r in results: f.write(json.dumps(r, ensure_ascii=False)+"\n")
            ds = {"name": req.dataset_name, "path": str(path), "size": len(results), "created_at": datetime.utcnow().isoformat(), "generators": out["generators_used"], "functions": out["functions_used"]}
            try: redis_client.setex(f"dataset:{req.dataset_name}", 86400, json.dumps(ds))
            except Exception: pass
            out["dataset_info"] = ds; out["message"] = f"Batch generation complete. Dataset saved as {ds['name']}"
        else:
            out["odes"] = results
        await jobs.complete(job_id, out)
    except Exception as e:
        await jobs.fail(job_id, f"Batch generation job failed: {e}\n{traceback.format_exc()}")

async def _job_ml_train(job_id: str, req: MLTrainingRequest):
    try:
        await jobs.update(job_id, {"status":"running", "metadata": {"status":"training"}})
        # Placeholder; implement your real training here
        time.sleep(2)
        out = {"model_id": f"{req.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}", "training_completed": True}
        await jobs.complete(job_id, out)
    except Exception as e:
        await jobs.fail(job_id, f"ML training failed: {e}")

async def _job_ml_generate(job_id: str, req: MLGenerationRequest):
    try:
        await jobs.update(job_id, {"status":"running"})
        odes = []
        gens = [req.generator] if req.generator else (AVAILABLE_GENS[:3] or ["L1"])
        funcs = [req.function]  if req.function  else (AVAILABLE_FUNCS[:3] or ["sine"])
        for i in range(req.n_samples):
            await jobs.update(job_id, {"progress": (i/max(req.n_samples,1))*100.0, "metadata": {"current": i, "total": req.n_samples}})
            gen = str(np.random.choice(gens)); fun = str(np.random.choice(funcs))
            odes.append({"id": f"ml_{uuid.uuid4().hex[:8]}", "ode": f"Eq(Derivative(y(x), x, 2) + y(x), pi*sin(x))", "solution": "pi*sin(x)", "generator": gen, "function": fun, "model_type": "placeholder", "temperature": float(req.temperature), "complexity": int(np.random.randint(50,200)), "verified": False, "ml_generated": True})
        await jobs.complete(job_id, {"odes": odes, "metrics": {"total_generated": len(odes)}})
    except Exception as e:
        await jobs.fail(job_id, f"ML generation failed: {e}")

# ---- Startup / run ----
@app.on_event("startup")
async def _startup():
    for d in ["models","data","logs"]: Path(d).mkdir(exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
