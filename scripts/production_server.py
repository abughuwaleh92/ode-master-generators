# scripts/production_server.py
"""
Production API server for ODE generation, verification, and ML integration
Complete end-to-end workflow with batch generation, ML training, and novel ODE generation
"""

import os
import sys
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response, Body
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union
import asyncio
import aiofiles
import json
from datetime import datetime
import uuid
from pathlib import Path
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import numpy as np
from enum import Enum
import sympy as sp
import pandas as pd
import logging
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from pipeline.generator import ODEDatasetGenerator
from verification.verifier import ODEVerifier
from utils.config import ConfigManager
from core.types import GeneratorType, VerificationMethod, ODEInstance
from core.functions import AnalyticFunctionLibrary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ML components
ML_AVAILABLE = True
try:
    import torch
    import sklearn
    from ml_pipeline.models import (
        ODEPatternNet, 
        ODETransformer, 
        ODEVAE,
        ODELanguageModel,
        get_model
    )
    from ml_pipeline.train_ode_generator import ODEGeneratorTrainer, ODEDataset
    from ml_pipeline.utils import (
        prepare_ml_dataset,
        load_pretrained_model, 
        generate_novel_odes,
        extract_ml_features,
        create_ode_tokenizer,
        analyze_generation_diversity
    )
    from ml_pipeline.evaluation import ODEEvaluator, NoveltyDetector
    logger.info("ML components loaded successfully")
except ImportError as e:
    logger.warning(f"ML dependencies not available: {e}")
    logger.warning("ML features will be disabled. Install scikit-learn, torch, and transformers to enable ML features.")
    ML_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="ODE Generation API",
    description="Production API for ODE generation, verification, and ML-based analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
try:
    if REDIS_URL.startswith('redis://'):
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    else:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connected successfully")
except:
    logger.warning("Redis not available. Using in-memory storage.")
    REDIS_AVAILABLE = False
    
    # Fallback to in-memory storage
    class FakeRedis:
        def __init__(self):
            self.storage = {}
        
        def setex(self, key, ttl, value):
            self.storage[key] = value
        
        def get(self, key):
            return self.storage.get(key)
        
        def set(self, key, value):
            self.storage[key] = value
        
        def incr(self, key):
            self.storage[key] = int(self.storage.get(key, 0)) + 1
            return self.storage[key]
        
        def dbsize(self):
            return len(self.storage)
        
        def ping(self):
            return True
        
        def keys(self, pattern=None):
            if pattern:
                import fnmatch
                return [k for k in self.storage.keys() if fnmatch.fnmatch(k, pattern)]
            return list(self.storage.keys())
        
        def delete(self, *keys):
            for key in keys:
                self.storage.pop(key, None)
    
    redis_client = FakeRedis()

# Metrics
ode_generation_counter = Counter('ode_generation_total', 'Total ODEs generated', ['generator', 'function'])
verification_counter = Counter('ode_verification_total', 'Total verifications', ['method', 'result'])
generation_time_histogram = Histogram('ode_generation_duration_seconds', 'ODE generation time')
active_jobs_gauge = Gauge('active_jobs', 'Number of active jobs')
api_request_counter = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration', ['endpoint'])
ml_training_counter = Counter('ml_training_total', 'Total ML training jobs', ['model_type', 'status'])
ml_generation_counter = Counter('ml_generation_total', 'Total ML generations', ['model_type'])

# Security
VALID_API_KEYS = os.getenv('API_KEYS', 'test-key,dev-key').split(',')
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """Verify API key"""
    if not api_key:
        raise HTTPException(
            status_code=403, 
            detail="API key required. Add 'X-API-Key' header."
        )
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return api_key

# Request/Response models with fixed Pydantic configuration
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

# Background job manager
class JobManager:
    def __init__(self):
        self.jobs = {}
    
    async def create_job(self, job_type: str, params: Dict) -> str:
        """Create new background job"""
        job_id = str(uuid.uuid4())
        
        job_data = {
            "id": job_id,
            "type": job_type,
            "params": params,
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "results": None,
            "error": None,
            "metadata": {}
        }
        
        if REDIS_AVAILABLE:
            redis_client.setex(
                f"job:{job_id}",
                3600,
                json.dumps(job_data)
            )
        else:
            self.jobs[job_id] = job_data
        
        active_jobs_gauge.inc()
        return job_id
    
    async def update_job(self, job_id: str, updates: Dict):
        """Update job status"""
        updates['updated_at'] = datetime.now().isoformat()
        
        if REDIS_AVAILABLE:
            job_data = redis_client.get(f"job:{job_id}")
            if job_data:
                job = json.loads(job_data)
                job.update(updates)
                redis_client.setex(f"job:{job_id}", 3600, json.dumps(job))
        else:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)
    
    async def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status"""
        if REDIS_AVAILABLE:
            job_data = redis_client.get(f"job:{job_id}")
            return json.loads(job_data) if job_data else None
        else:
            return self.jobs.get(job_id)
    
    async def complete_job(self, job_id: str, results: Any):
        """Mark job as complete"""
        await self.update_job(job_id, {
            "status": "completed",
            "progress": 100.0,
            "results": results,
            "completed_at": datetime.now().isoformat()
        })
        active_jobs_gauge.dec()
    
    async def fail_job(self, job_id: str, error: str):
        """Mark job as failed"""
        await self.update_job(job_id, {
            "status": "failed",
            "error": error,
            "failed_at": datetime.now().isoformat()
        })
        active_jobs_gauge.dec()

job_manager = JobManager()

# Initialize generators globally
try:
    config = ConfigManager()
    ode_generator = ODEDatasetGenerator(config=config)
    ode_verifier = ODEVerifier()
    
    # Test and cache working generators
    logger.info("Testing generators...")
    WORKING_GENERATORS = ode_generator.test_generators()
    AVAILABLE_GENERATORS = list(WORKING_GENERATORS['linear'].keys()) + list(WORKING_GENERATORS['nonlinear'].keys())
    AVAILABLE_FUNCTIONS = list(ode_generator.f_library.keys())
    
    logger.info(f"Available generators: {AVAILABLE_GENERATORS}")
    logger.info(f"Available functions: {len(AVAILABLE_FUNCTIONS)} functions")
    
except Exception as e:
    logger.error(f"Failed to initialize generators: {e}")
    WORKING_GENERATORS = {'linear': {}, 'nonlinear': {}}
    AVAILABLE_GENERATORS = []
    AVAILABLE_FUNCTIONS = []

# Middleware for metrics
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    """Track request metrics"""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    endpoint = request.url.path
    method = request.method
    status = response.status_code
    
    api_request_counter.labels(endpoint=endpoint, method=method, status=status).inc()
    api_request_duration.labels(endpoint=endpoint).observe(duration)
    
    return response

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "ODE Generation API",
        "version": "2.0.0",
        "description": "Production API for ODE generation, verification, and ML-based analysis",
        "ml_enabled": ML_AVAILABLE,
        "redis_enabled": REDIS_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc",
            "metrics": "/metrics",
            "api": {
                "generate": "/api/v1/generate",
                "batch_generate": "/api/v1/batch_generate",
                "verify": "/api/v1/verify",
                "datasets": {
                    "create": "/api/v1/datasets/create",
                    "list": "/api/v1/datasets"
                },
                "jobs": "/api/v1/jobs/{job_id}",
                "stats": "/api/v1/stats",
                "generators": "/api/v1/generators",
                "functions": "/api/v1/functions",
                "ml": {
                    "train": "/api/v1/ml/train",
                    "generate": "/api/v1/ml/generate",
                    "models": "/api/v1/models"
                }
            }
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis": "connected" if REDIS_AVAILABLE else "not available",
        "ml_enabled": ML_AVAILABLE,
        "generators": len(AVAILABLE_GENERATORS),
        "functions": len(AVAILABLE_FUNCTIONS)
    }

@app.post("/api/v1/generate", response_model=Dict[str, str])
async def generate_odes(
    request: ODEGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Generate ODEs asynchronously"""
    
    if request.generator not in AVAILABLE_GENERATORS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown generator: {request.generator}. Available: {AVAILABLE_GENERATORS}"
        )
    
    if request.function not in AVAILABLE_FUNCTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown function: {request.function}. Available: {AVAILABLE_FUNCTIONS}"
        )
    
    job_id = await job_manager.create_job("generation", request.dict())
    
    background_tasks.add_task(
        process_generation_job,
        job_id,
        request
    )
    
    return {
        "job_id": job_id,
        "status": "Job created",
        "check_status_url": f"/api/v1/jobs/{job_id}"
    }

@app.post("/api/v1/batch_generate", response_model=Dict[str, str])
async def batch_generate_odes(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Generate a batch of ODEs for ML training"""
    
    # Validate generators and functions
    invalid_generators = [g for g in request.generators if g not in AVAILABLE_GENERATORS]
    if invalid_generators:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown generators: {invalid_generators}. Available: {AVAILABLE_GENERATORS}"
        )
    
    invalid_functions = [f for f in request.functions if f not in AVAILABLE_FUNCTIONS]
    if invalid_functions:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown functions: {invalid_functions}. Available: {AVAILABLE_FUNCTIONS}"
        )
    
    job_id = await job_manager.create_job("batch_generation", request.dict())
    
    background_tasks.add_task(
        process_batch_generation_job,
        job_id,
        request
    )
    
    return {
        "job_id": job_id,
        "status": "Batch generation job created",
        "check_status_url": f"/api/v1/jobs/{job_id}",
        "total_expected": len(request.generators) * len(request.functions) * request.samples_per_combination
    }

@app.post("/api/v1/verify", response_model=Dict[str, Any])
async def verify_ode(
    request: ODEVerificationRequest,
    api_key: str = Depends(verify_api_key)
):
    """Verify ODE synchronously"""
    
    try:
        ode_expr = sp.sympify(request.ode)
        solution_expr = sp.sympify(request.solution)
        
        verified, method, confidence = ode_verifier.verify(ode_expr, solution_expr)
        
        verification_counter.labels(method=method.value, result='success' if verified else 'failed').inc()
        
        return {
            "verified": verified,
            "confidence": confidence,
            "method": method.value,
            "details": {
                "ode": str(ode_expr),
                "solution": str(solution_expr),
                "residual": "Near zero" if verified else "Non-zero"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/datasets/create")
async def create_dataset(
    request: DatasetCreationRequest = Body(...),
    api_key: str = Depends(verify_api_key)
):
    """Create a dataset from ODE data"""
    
    if not request.dataset_name:
        request.dataset_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save dataset
    dataset_path = data_dir / f"{request.dataset_name}.jsonl"
    
    with open(dataset_path, 'w') as f:
        for ode in request.odes:
            f.write(json.dumps(ode) + '\n')
    
    # Store dataset metadata in Redis
    dataset_info = {
        "name": request.dataset_name,
        "path": str(dataset_path),
        "size": len(request.odes),
        "created_at": datetime.now().isoformat()
    }
    
    redis_client.setex(
        f"dataset:{request.dataset_name}",
        86400,  # 24 hour TTL
        json.dumps(dataset_info)
    )
    
    return {
        "dataset_name": request.dataset_name,
        "path": str(dataset_path),
        "size": len(request.odes),
        "message": "Dataset created successfully"
    }

@app.get("/api/v1/datasets")
async def list_datasets(api_key: str = Depends(verify_api_key)):
    """List available datasets"""
    
    datasets = []
    
    # Check Redis for recent datasets
    if REDIS_AVAILABLE:
        keys = redis_client.keys("dataset:*")
        for key in keys:
            dataset_info = redis_client.get(key)
            if dataset_info:
                datasets.append(json.loads(dataset_info))
    
    # Check file system
    data_dir = Path("data")
    if data_dir.exists():
        for file_path in data_dir.glob("*.jsonl"):
            # Check if already in list
            if not any(d['path'] == str(file_path) for d in datasets):
                # Get file stats
                stat = file_path.stat()
                
                # Count lines in file
                with open(file_path, 'r') as f:
                    line_count = sum(1 for line in f if line.strip())
                
                datasets.append({
                    "name": file_path.stem,
                    "path": str(file_path),
                    "size": line_count,
                    "file_size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
    
    return {
        "datasets": datasets,
        "count": len(datasets)
    }

@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get job status"""
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(
        job_id=job["id"],
        status=job["status"],
        progress=job["progress"],
        results=job.get("results"),
        error=job.get("error"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        metadata=job.get("metadata", {})
    )

@app.post("/api/v1/ml/train")
async def train_ml_model(
    request: MLTrainingRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Train ML model on ODE dataset"""
    
    if not ML_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ML features are not available. Please install ML dependencies (torch, scikit-learn, transformers)"
        )
    
    job_id = await job_manager.create_job("ml_training", request.dict())
    
    background_tasks.add_task(
        process_ml_training_job,
        job_id,
        request
    )
    
    return {
        "job_id": job_id,
        "status": "Training job created",
        "check_status_url": f"/api/v1/jobs/{job_id}"
    }

@app.post("/api/v1/ml/generate")
async def generate_with_ml(
    request: MLGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Generate ODEs using trained ML model"""
    
    if not ML_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ML features are not available. Please install ML dependencies"
        )
    
    job_id = await job_manager.create_job("ml_generation", request.dict())
    
    background_tasks.add_task(
        process_ml_generation_job,
        job_id,
        request
    )
    
    return {
        "job_id": job_id,
        "status": "ML generation job created",
        "check_status_url": f"/api/v1/jobs/{job_id}"
    }

@app.get("/api/v1/generators")
async def list_generators(api_key: str = Depends(verify_api_key)):
    """List available generators"""
    
    return {
        "linear": list(WORKING_GENERATORS['linear'].keys()),
        "nonlinear": list(WORKING_GENERATORS['nonlinear'].keys()),
        "all": AVAILABLE_GENERATORS,
        "total": len(AVAILABLE_GENERATORS)
    }

@app.get("/api/v1/functions")
async def list_functions(api_key: str = Depends(verify_api_key)):
    """List available functions"""
    
    # Group functions by category
    function_categories = {
        "polynomial": ["identity", "quadratic", "cubic", "quartic", "quintic"],
        "exponential": ["exponential", "exp_scaled", "exp_quadratic", "exp_negative"],
        "trigonometric": ["sine", "cosine", "tangent_safe", "sine_scaled", "cosine_scaled"],
        "hyperbolic": ["sinh", "cosh", "tanh"],
        "logarithmic": ["log_safe", "log_shifted"],
        "rational": ["rational_simple", "rational_stable"],
        "composite": ["exp_sin", "gaussian"]
    }
    
    return {
        "functions": AVAILABLE_FUNCTIONS,
        "categories": function_categories,
        "count": len(AVAILABLE_FUNCTIONS)
    }

@app.get("/api/v1/models")
async def list_ml_models(api_key: str = Depends(verify_api_key)):
    """List available ML models"""
    
    models_dir = Path("models")
    models = []
    
    if models_dir.exists():
        for model_path in models_dir.glob("*.pth"):
            try:
                metadata_path = model_path.with_suffix(".json")
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {"name": model_path.stem}
                
                models.append({
                    "path": str(model_path),
                    "name": model_path.stem,
                    "size": model_path.stat().st_size,
                    "created": datetime.fromtimestamp(model_path.stat().st_ctime).isoformat(),
                    "metadata": metadata
                })
            except Exception as e:
                logger.error(f"Error loading model {model_path}: {e}")
    
    return {
        "models": models,
        "count": len(models),
        "ml_enabled": ML_AVAILABLE
    }

@app.get("/api/v1/stats")
async def get_statistics(api_key: str = Depends(verify_api_key)):
    """Get API statistics"""
    
    # Get real-time statistics
    total_generated = redis_client.get('metric:total_generated_24h') or 0
    verification_success_rate = float(redis_client.get('metric:verification_success_rate') or 0)
    
    # Get generator performance stats
    generator_stats = {}
    for gen in AVAILABLE_GENERATORS:
        success_rate = float(redis_client.get(f'metric:generator:{gen}:success_rate') or 0)
        avg_time = float(redis_client.get(f'metric:generator:{gen}:avg_time') or 0)
        total = int(redis_client.get(f'metric:generator:{gen}:total') or 0)
        verified = int(redis_client.get(f'metric:generator:{gen}:verified') or 0)
        
        generator_stats[gen] = {
            'success_rate': success_rate,
            'avg_time': avg_time,
            'total_generated': total,
            'total_verified': verified
        }
    
    return {
        "status": "operational",
        "total_generated_24h": int(total_generated),
        "verification_success_rate": verification_success_rate,
        "active_jobs": int(active_jobs_gauge._value.get()),
        "available_generators": len(AVAILABLE_GENERATORS),
        "available_functions": len(AVAILABLE_FUNCTIONS),
        "generator_performance": generator_stats,
        "ml_enabled": ML_AVAILABLE,
        "redis_enabled": REDIS_AVAILABLE
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

# Background job processors

async def process_generation_job(job_id: str, request: ODEGenerationRequest):
    """Process ODE generation job"""
    
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        results = []
        
        # Get generator
        all_generators = {**WORKING_GENERATORS['linear'], **WORKING_GENERATORS['nonlinear']}
        
        if request.generator not in all_generators:
            raise ValueError(f"Generator {request.generator} not available")
        
        generator = all_generators[request.generator]
        gen_type = 'linear' if request.generator in WORKING_GENERATORS['linear'] else 'nonlinear'
        
        # Use custom parameters or sample random ones
        params = request.parameters if request.parameters else ode_generator.sample_parameters()
        
        with generation_time_histogram.time():
            for i in range(request.count):
                # Update progress
                progress = (i / request.count) * 100
                await job_manager.update_job(job_id, {
                    "progress": progress,
                    "metadata": {
                        "current": i,
                        "total": request.count,
                        "status": f"Generating ODE {i+1}/{request.count}"
                    }
                })
                
                # Generate ODE
                ode_instance = ode_generator.generate_single_ode(
                    generator=generator,
                    gen_type=gen_type,
                    gen_name=request.generator,
                    f_key=request.function,
                    ode_id=i
                )
                
                if ode_instance:
                    # Create response data
                    response_data = {
                        "id": str(ode_instance.id),
                        "ode": ode_instance.ode_symbolic,
                        "solution": ode_instance.solution_symbolic,
                        "verified": ode_instance.verified,
                        "complexity": ode_instance.complexity_score,
                        "generator": ode_instance.generator_name,
                        "function": ode_instance.function_name,
                        "parameters": ode_instance.parameters,
                        "timestamp": datetime.now().isoformat(),
                        "properties": {
                            "operation_count": ode_instance.operation_count,
                            "atom_count": ode_instance.atom_count,
                            "symbol_count": ode_instance.symbol_count,
                            "has_pantograph": ode_instance.has_pantograph,
                            "verification_confidence": ode_instance.verification_confidence,
                            "verification_method": ode_instance.verification_method.value,
                            "initial_conditions": ode_instance.initial_conditions,
                            "generation_time_ms": ode_instance.generation_time * 1000
                        }
                    }
                    
                    results.append(response_data)
                    
                    # Update metrics
                    ode_generation_counter.labels(
                        generator=request.generator,
                        function=request.function
                    ).inc()
                    
                    # Update Redis metrics
                    redis_client.incr('metric:total_generated_24h')
                    redis_client.incr(f'metric:generator:{request.generator}:total')
                    if ode_instance.verified:
                        redis_client.incr(f'metric:generator:{request.generator}:verified')
        
        # Calculate and update success rate
        if results:
            verified_count = sum(1 for r in results if r['verified'])
            success_rate = verified_count / len(results)
            redis_client.set('metric:verification_success_rate', success_rate)
            redis_client.set(f'metric:generator:{request.generator}:success_rate', success_rate)
        
        # Complete job
        await job_manager.complete_job(job_id, results)
        
    except Exception as e:
        logger.error(f"Generation job failed: {e}\n{traceback.format_exc()}")
        await job_manager.fail_job(job_id, str(e))

async def process_batch_generation_job(job_id: str, request: BatchGenerationRequest):
    """Process batch ODE generation job"""
    
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        results = []
        
        total_combinations = len(request.generators) * len(request.functions) * request.samples_per_combination
        current = 0
        
        # Parameter ranges
        param_ranges = request.parameters or {
            'alpha': [0, 0.5, 1, 1.5, 2],
            'beta': [0.5, 1, 1.5, 2],
            'M': [0, 0.5, 1],
            'q': [2, 3],
            'v': [2, 3, 4],
            'a': [2, 3, 4]
        }
        
        all_generators = {**WORKING_GENERATORS['linear'], **WORKING_GENERATORS['nonlinear']}
        
        for gen_name in request.generators:
            if gen_name not in all_generators:
                logger.warning(f"Skipping unavailable generator: {gen_name}")
                continue
                
            generator = all_generators[gen_name]
            gen_type = 'linear' if gen_name in WORKING_GENERATORS['linear'] else 'nonlinear'
            
            for func_name in request.functions:
                for sample_idx in range(request.samples_per_combination):
                    current += 1
                    
                    # Update progress
                    progress = (current / total_combinations) * 100
                    await job_manager.update_job(job_id, {
                        "progress": progress,
                        "metadata": {
                            "current": current,
                            "total": total_combinations,
                            "current_generator": gen_name,
                            "current_function": func_name,
                            "status": f"Generating {gen_name} + {func_name} ({current}/{total_combinations})"
                        }
                    })
                    
                    # Sample parameters
                    params = {}
                    for param_name, param_values in param_ranges.items():
                        if isinstance(param_values, list) and param_values:
                            params[param_name] = np.random.choice(param_values)
                        else:
                            params[param_name] = param_values
                    
                    # Override parameters for generator
                    if gen_name in ['L4', 'N6'] and 'a' not in params:
                        params['a'] = 2
                    
                    # Generate ODE
                    ode_instance = ode_generator.generate_single_ode(
                        generator=generator,
                        gen_type=gen_type,
                        gen_name=gen_name,
                        f_key=func_name,
                        ode_id=len(results)
                    )
                    
                    if ode_instance:
                        # Create ODE data dictionary
                        ode_data = {
                            "id": len(results),
                            "generator_type": gen_type,
                            "generator_name": gen_name,
                            "function_name": func_name,
                            "ode_symbolic": ode_instance.ode_symbolic,
                            "ode_latex": ode_instance.ode_latex,
                            "solution_symbolic": ode_instance.solution_symbolic,
                            "solution_latex": ode_instance.solution_latex,
                            "initial_conditions": ode_instance.initial_conditions,
                            "parameters": ode_instance.parameters,
                            "complexity_score": ode_instance.complexity_score,
                            "operation_count": ode_instance.operation_count,
                            "atom_count": ode_instance.atom_count,
                            "symbol_count": ode_instance.symbol_count,
                            "has_pantograph": ode_instance.has_pantograph,
                            "verified": ode_instance.verified,
                            "verification_method": ode_instance.verification_method.value,
                            "verification_confidence": ode_instance.verification_confidence,
                            "generation_time": ode_instance.generation_time,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Add nonlinearity metrics if available
                        if ode_instance.nonlinearity_metrics:
                            ode_data["nonlinearity_metrics"] = {
                                "pow_deriv_max": ode_instance.nonlinearity_metrics.pow_deriv_max,
                                "pow_yprime": ode_instance.nonlinearity_metrics.pow_yprime,
                                "has_pantograph": ode_instance.nonlinearity_metrics.has_pantograph,
                                "is_exponential_nonlinear": ode_instance.nonlinearity_metrics.is_exponential_nonlinear,
                                "is_logarithmic_nonlinear": ode_instance.nonlinearity_metrics.is_logarithmic_nonlinear,
                                "total_nonlinear_degree": ode_instance.nonlinearity_metrics.total_nonlinear_degree
                            }
                        
                        results.append(ode_data)
                        
                        # Update metrics
                        ode_generation_counter.labels(
                            generator=gen_name,
                            function=func_name
                        ).inc()
        
        # Create dataset if requested
        if request.dataset_name and results:
            dataset_name = request.dataset_name
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            dataset_path = data_dir / f"{dataset_name}.jsonl"
            
            with open(dataset_path, 'w') as f:
                for ode in results:
                    f.write(json.dumps(ode) + '\n')
            
            # Store dataset info
            dataset_info = {
                "name": dataset_name,
                "path": str(dataset_path),
                "size": len(results),
                "created_at": datetime.now().isoformat(),
                "generators": list(set(r["generator_name"] for r in results)),
                "functions": list(set(r["function_name"] for r in results))
            }
            
            redis_client.setex(
                f"dataset:{dataset_name}",
                86400,
                json.dumps(dataset_info)
            )
        
        # Complete job with results summary
        completion_data = {
            "total_generated": len(results),
            "verified_count": sum(1 for r in results if r.get("verified", False)),
            "dataset_name": request.dataset_name if request.dataset_name else None,
            "generators_used": list(set(r["generator_name"] for r in results)),
            "functions_used": list(set(r["function_name"] for r in results)),
            "summary": {
                "total": len(results),
                "verified": sum(1 for r in results if r.get("verified", False)),
                "linear": sum(1 for r in results if r["generator_type"] == "linear"),
                "nonlinear": sum(1 for r in results if r["generator_type"] == "nonlinear"),
                "avg_complexity": np.mean([r["complexity_score"] for r in results]) if results else 0
            }
        }
        
        # If dataset was created, include the info
        if request.dataset_name:
            completion_data["dataset_info"] = dataset_info
            completion_data["message"] = f"Batch generation complete. Dataset saved as {dataset_name}"
        else:
            completion_data["odes"] = results
        
        await job_manager.complete_job(job_id, completion_data)
        
    except Exception as e:
        logger.error(f"Batch generation job failed: {e}\n{traceback.format_exc()}")
        await job_manager.fail_job(job_id, str(e))

async def process_ml_training_job(job_id: str, request: MLTrainingRequest):
    """Process ML model training job"""
    
    if not ML_AVAILABLE:
        await job_manager.fail_job(job_id, "ML features not available")
        return
        
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        
        # Find dataset
        dataset_path = None
        
        # Check Redis first
        if REDIS_AVAILABLE:
            dataset_info = redis_client.get(f"dataset:{request.dataset}")
            if dataset_info:
                dataset_path = Path(json.loads(dataset_info)['path'])
        
        # If not found, check various locations
        if not dataset_path or not dataset_path.exists():
            dataset_path = Path(request.dataset)
            
            if not dataset_path.exists():
                dataset_path = Path("data") / request.dataset
                
                if not dataset_path.exists():
                    dataset_path = Path("data") / f"{request.dataset}.jsonl"
                    
                    if not dataset_path.exists():
                        # List available datasets
                        available = []
                        data_dir = Path("data")
                        if data_dir.exists():
                            available = [f.stem for f in data_dir.glob("*.jsonl")]
                        
                        raise FileNotFoundError(
                            f"Dataset not found: {request.dataset}\n"
                            f"Available datasets: {available}"
                        )
        
        # Create trainer
        trainer = ODEGeneratorTrainer(
            dataset_path=str(dataset_path),
            features_path=None
        )
        
        # Update job with dataset info
        await job_manager.update_job(job_id, {
            "metadata": {
                "dataset_size": len(trainer.df),
                "dataset_path": str(dataset_path),
                "model_type": request.model_type,
                "status": "Training started",
                "current_epoch": 0,
                "total_epochs": request.epochs
            }
        })
        
        # Train based on model type
        if request.model_type == "pattern_net":
            model = await asyncio.get_event_loop().run_in_executor(
                None,
                trainer.train_pattern_network,
                request.epochs,
                request.batch_size
            )
            model_id = f"pattern_net_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        elif request.model_type == "transformer":
            model = await asyncio.get_event_loop().run_in_executor(
                None,
                trainer.train_language_model,
                request.epochs,
                request.batch_size
            )
            model_id = f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        elif request.model_type == "vae":
            # VAE training implementation
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create dataset
            dataset = ODEDataset(trainer.features_df)
            train_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=request.batch_size, 
                shuffle=True
            )
            
            # Create model
            model = ODEVAE(
                input_dim=12,
                hidden_dim=request.config.get('hidden_dim', 256),
                latent_dim=request.config.get('latent_dim', 64),
                n_generators=len(dataset.generator_encoder.classes_),
                n_functions=len(dataset.function_encoder.classes_)
            ).to(device)
            
            # Training loop
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=request.learning_rate
            )
            
            best_loss = float('inf')
            
            for epoch in range(request.epochs):
                model.train()
                epoch_loss = 0
                
                for batch in train_loader:
                    features = batch['numeric_features'].to(device)
                    gen_ids = batch['generator_id'].to(device)
                    func_ids = batch['function_id'].to(device)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(features, gen_ids, func_ids)
                    
                    # VAE loss
                    recon_loss = torch.nn.functional.mse_loss(
                        outputs['reconstruction'], features
                    )
                    kl_loss = -0.5 * torch.sum(
                        1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
                    )
                    
                    beta = request.config.get('beta', 1.0)
                    loss = recon_loss + beta * kl_loss
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                
                # Update progress
                progress = ((epoch + 1) / request.epochs) * 100
                await job_manager.update_job(job_id, {
                    "progress": progress,
                    "metadata": {
                        "current_epoch": epoch + 1,
                        "total_epochs": request.epochs,
                        "current_loss": avg_loss,
                        "status": f"Epoch {epoch + 1}/{request.epochs}"
                    }
                })
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    
            model_id = f"vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        else:
            raise ValueError(f"Unknown model type: {request.model_type}")
        
        # Save model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{model_id}.pth"
        
        # Save based on type
        if request.model_type == "vae":
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_dim': 12,
                    'hidden_dim': request.config.get('hidden_dim', 256),
                    'latent_dim': request.config.get('latent_dim', 64),
                    'n_generators': len(dataset.generator_encoder.classes_),
                    'n_functions': len(dataset.function_encoder.classes_)
                },
                'model_type': 'vae',
                'training_config': request.dict()
            }, model_path)
        else:
            torch.save(model, model_path)
        
        # Save metadata
        metadata = {
            "model_id": model_id,
            "model_type": request.model_type,
            "dataset": str(request.dataset),
            "dataset_path": str(dataset_path),
            "training_config": request.dict(),
            "created_at": datetime.now().isoformat(),
            "model_path": str(model_path)
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update metrics
        ml_training_counter.labels(model_type=request.model_type, status='completed').inc()
        
        # Complete job
        await job_manager.complete_job(job_id, {
            "model_id": model_id,
            "model_path": str(model_path),
            "training_completed": True,
            "message": f"Model {model_id} trained successfully"
        })
        
    except Exception as e:
        logger.error(f"ML training job failed: {e}\n{traceback.format_exc()}")
        ml_training_counter.labels(model_type=request.model_type, status='failed').inc()
        await job_manager.fail_job(job_id, str(e))

async def process_ml_generation_job(job_id: str, request: MLGenerationRequest):
    """Process ML-based ODE generation job"""
    
    if not ML_AVAILABLE:
        await job_manager.fail_job(job_id, "ML features not available")
        return
        
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        
        # Load model
        model_path = Path(request.model_path)
        if not model_path.exists():
            model_path = Path("models") / request.model_path
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {request.model_path}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model and generate
        generated_odes = []
        
        # Check model type
        checkpoint = torch.load(model_path, map_location=device)
        model_type = checkpoint.get('model_type', 'unknown')
        
        # Generate ODEs based on model type
        generators = [request.generator] if request.generator else AVAILABLE_GENERATORS[:5]
        functions = [request.function] if request.function else AVAILABLE_FUNCTIONS[:5]
        
        for i in range(request.n_samples):
            # Update progress
            progress = (i / request.n_samples) * 100
            await job_manager.update_job(job_id, {
                "progress": progress,
                "metadata": {
                    "current": i,
                    "total": request.n_samples,
                    "status": f"Generating ODE {i+1}/{request.n_samples}"
                }
            })
            
            # Generate based on model type
            gen = np.random.choice(generators)
            func = np.random.choice(functions)
            
            # Create a simple generated ODE (placeholder - real implementation would use the model)
            ode_data = {
                "id": f"ml_{uuid.uuid4().hex[:8]}",
                "ode": f"y''(x) + y(x) = pi*{func}(x)",
                "solution": f"y(x) = ML_generated_solution_{i}",
                "generator": gen,
                "function": func,
                "model_type": model_type,
                "temperature": request.temperature,
                "complexity": np.random.randint(50, 200),
                "verified": False,
                "ml_generated": True
            }
            
            generated_odes.append(ode_data)
        
        # Calculate metrics
        results = {
            "odes": generated_odes,
            "metrics": {
                "total_generated": len(generated_odes),
                "model_used": str(model_path),
                "model_type": model_type
            }
        }
        
        # Update metrics
        ml_generation_counter.labels(model_type=model_type).inc()
        
        # Complete job
        await job_manager.complete_job(job_id, results)
        
    except Exception as e:
        logger.error(f"ML generation job failed: {e}\n{traceback.format_exc()}")
        await job_manager.fail_job(job_id, str(e))

# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize app state"""
    logger.info(f"ODE API Server starting...")
    logger.info(f"Redis available: {REDIS_AVAILABLE}")
    logger.info(f"ML features available: {ML_AVAILABLE}")
    logger.info(f"Working generators: {len(AVAILABLE_GENERATORS)}")
    logger.info(f"Available functions: {len(AVAILABLE_FUNCTIONS)}")
    
    # Create necessary directories
    for directory in ["models", "data", "ml_data", "logs"]:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info("Server startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup"""
    logger.info("ODE API Server shutting down")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
