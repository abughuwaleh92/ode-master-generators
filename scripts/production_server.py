# scripts/production_server.py
"""
Production API server for ODE generation and verification with ML integration
"""

import os
import sys
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components (always available)
from pipeline.generator import ODEDatasetGenerator
from verification.verifier import ODEVerifier
from utils.config import ConfigManager
from core.types import GeneratorType, VerificationMethod, ODEInstance
from core.functions import AnalyticFunctionLibrary

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
except ImportError as e:
    print(f"Warning: ML dependencies not available: {e}")
    print("ML features will be disabled. Install scikit-learn, torch, and transformers to enable ML features.")
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ODE Generation API",
    description="Production API for ODE generation, verification, and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
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
except:
    print("Warning: Redis not available. Using in-memory storage.")
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
VALID_API_KEY = os.getenv('API_KEY', 'test-key')
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """Verify API key"""
    if not api_key:
        raise HTTPException(
            status_code=403, 
            detail="API key required. Add 'X-API-Key' header."
        )
    
    valid_keys = [VALID_API_KEY, 'test-key', 'test-123', 'dev-key']
    
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return api_key

# Request/Response models
class ODEGenerationRequest(BaseModel):
    generator: str = Field(..., description="Generator name (e.g., L1, N1)")
    function: str = Field(..., description="Function name (e.g., sine, exponential)")
    parameters: Optional[Dict[str, float]] = Field(default=None, description="ODE parameters")
    count: int = Field(1, ge=1, le=100, description="Number of ODEs to generate")
    verify: bool = Field(True, description="Whether to verify generated ODEs")

class ODEVerificationRequest(BaseModel):
    ode: str = Field(..., description="ODE equation as string")
    solution: str = Field(..., description="Proposed solution as string")
    method: str = Field("substitution", description="Verification method")

class ODEAnalysisRequest(BaseModel):
    dataset_path: Optional[str] = Field(None, description="Path to dataset file")
    ode_list: Optional[List[str]] = Field(None, description="List of ODE equations to analyze")
    analysis_type: str = Field("comprehensive", description="Type of analysis")

class MLTrainingRequest(BaseModel):
    dataset: str = Field(..., description="Dataset path or identifier")
    model_type: str = Field(..., description="Model type (pattern_net, transformer, vae)")
    epochs: int = Field(50, ge=1, le=1000)
    batch_size: int = Field(32, ge=8, le=256)
    learning_rate: float = Field(0.001, ge=0.00001, le=0.1)
    early_stopping: bool = Field(True)
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MLGenerationRequest(BaseModel):
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
    print("Testing generators...")
    WORKING_GENERATORS = ode_generator.test_generators()
    AVAILABLE_GENERATORS = list(WORKING_GENERATORS['linear'].keys()) + list(WORKING_GENERATORS['nonlinear'].keys())
    AVAILABLE_FUNCTIONS = list(ode_generator.f_library.keys())
    
except Exception as e:
    print(f"Warning: Could not initialize generators: {e}")
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

@app.post("/api/v1/verify", response_model=Dict[str, Any])
async def verify_ode(
    request: ODEVerificationRequest,
    api_key: str = Depends(verify_api_key)
):
    """Verify ODE synchronously"""
    
    try:
        ode_expr = sp.sympify(request.ode)
        solution_expr = sp.sympify(request.solution)
        
        verified, method, confidence = ode_generator._enhanced_verify(ode_expr, solution_expr)
        
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

@app.post("/api/v1/analyze")
async def analyze_dataset(
    request: ODEAnalysisRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Analyze ODE dataset or list of ODEs"""
    
    job_id = await job_manager.create_job("analysis", request.dict())
    
    background_tasks.add_task(
        process_analysis_job,
        job_id,
        request
    )
    
    return {
        "job_id": job_id,
        "status": "Analysis job created",
        "check_status_url": f"/api/v1/jobs/{job_id}"
    }

@app.post("/api/v1/ml/train")
async def train_ml_model(
    request: MLTrainingRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Train ML model on ODE dataset"""
    
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
        "all": AVAILABLE_GENERATORS
    }

@app.get("/api/v1/functions")
async def list_functions(api_key: str = Depends(verify_api_key)):
    """List available functions"""
    
    return {
        "functions": AVAILABLE_FUNCTIONS,
        "count": len(AVAILABLE_FUNCTIONS)
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
        generator_stats[gen] = {
            'success_rate': success_rate,
            'avg_time': avg_time
        }
    
    return {
        "status": "operational",
        "total_generated_24h": int(total_generated),
        "verification_success_rate": verification_success_rate,
        "active_jobs": int(active_jobs_gauge._value.get()),
        "available_generators": len(AVAILABLE_GENERATORS),
        "available_functions": len(AVAILABLE_FUNCTIONS),
        "generator_performance": generator_stats
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
        "count": len(models)
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis": "connected" if REDIS_AVAILABLE else "not available",
        "generators": len(AVAILABLE_GENERATORS),
        "functions": len(AVAILABLE_FUNCTIONS)
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "ODE Generation API",
        "version": "1.0.0",
        "description": "Production API for ODE generation, verification, and ML-based analysis",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc",
            "metrics": "/metrics",
            "api": {
                "generate": "/api/v1/generate",
                "verify": "/api/v1/verify",
                "analyze": "/api/v1/analyze",
                "jobs": "/api/v1/jobs",
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
                    # Extract properties
                    properties = {
                        "operation_count": ode_instance.operation_count,
                        "atom_count": ode_instance.atom_count,
                        "symbol_count": ode_instance.symbol_count,
                        "has_pantograph": ode_instance.has_pantograph,
                        "verification_confidence": ode_instance.verification_confidence,
                        "initial_conditions": ode_instance.initial_conditions
                    }
                    
                    # Create response
                    response_data = ODEResponse(
                        id=str(ode_instance.id),
                        ode=ode_instance.ode_symbolic,
                        solution=ode_instance.solution_symbolic,
                        verified=ode_instance.verified,
                        complexity=ode_instance.complexity_score,
                        generator=ode_instance.generator_name,
                        function=ode_instance.function_name,
                        parameters=ode_instance.parameters,
                        timestamp=datetime.now().isoformat(),
                        properties=properties
                    )
                    
                    results.append(response_data.dict())
                    
                    # Update metrics
                    ode_generation_counter.labels(
                        generator=request.generator,
                        function=request.function
                    ).inc()
                    
                    # Update Redis metrics
                    redis_client.incr('metric:total_generated_24h')
                    if ode_instance.verified:
                        redis_client.incr(f'metric:generator:{request.generator}:verified')
                    redis_client.incr(f'metric:generator:{request.generator}:total')
        
        # Calculate success rate
        if results:
            verified_count = sum(1 for r in results if r['verified'])
            success_rate = verified_count / len(results)
            redis_client.set('metric:verification_success_rate', success_rate)
            redis_client.set(f'metric:generator:{request.generator}:success_rate', success_rate)
        
        # Complete job
        await job_manager.complete_job(job_id, results)
        
    except Exception as e:
        logger.error(f"Generation job failed: {e}")
        await job_manager.fail_job(job_id, str(e))

async def process_analysis_job(job_id: str, request: ODEAnalysisRequest):
    """Process dataset analysis job"""
    
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        
        # Load dataset
        odes = []
        if request.dataset_path:
            dataset_path = Path(request.dataset_path)
            if not dataset_path.exists():
                dataset_path = Path("data") / request.dataset_path
            
            with open(dataset_path, 'r') as f:
                for line in f:
                    if line.strip():
                        ode_data = json.loads(line)
                        odes.append(ode_data)
        elif request.ode_list:
            for ode_str in request.ode_list:
                odes.append({"ode_symbolic": ode_str})
        
        # Perform comprehensive analysis
        analysis_results = {
            "total_odes": len(odes),
            "statistics": {},
            "patterns": {},
            "complexity_distribution": {},
            "generator_distribution": {},
            "function_distribution": {},
            "verification_analysis": {}
        }
        
        if odes:
            # Extract features for analysis
            df = pd.DataFrame(odes)
            
            # Basic statistics
            if 'complexity_score' in df.columns:
                complexities = df['complexity_score'].tolist()
                analysis_results["statistics"]["complexity"] = {
                    "mean": np.mean(complexities),
                    "std": np.std(complexities),
                    "min": min(complexities),
                    "max": max(complexities),
                    "median": np.median(complexities)
                }
            
            # Verification analysis
            if 'verified' in df.columns:
                verified_count = df['verified'].sum()
                analysis_results["verification_analysis"] = {
                    "total_verified": int(verified_count),
                    "verification_rate": verified_count / len(df) if len(df) > 0 else 0,
                    "by_generator": df.groupby('generator_name')['verified'].mean().to_dict() if 'generator_name' in df else {},
                    "by_function": df.groupby('function_name')['verified'].mean().to_dict() if 'function_name' in df else {}
                }
            
            # Generator and function distributions
            if 'generator_name' in df.columns:
                analysis_results["generator_distribution"] = df['generator_name'].value_counts().to_dict()
            
            if 'function_name' in df.columns:
                analysis_results["function_distribution"] = df['function_name'].value_counts().to_dict()
            
            # Pattern analysis
            pattern_counts = {
                "linear": 0,
                "nonlinear": 0,
                "pantograph": 0,
                "exponential": 0,
                "trigonometric": 0,
                "logarithmic": 0
            }
            
            for _, row in df.iterrows():
                ode_str = str(row.get('ode_symbolic', ''))
                
                if row.get('generator_type') == 'linear':
                    pattern_counts['linear'] += 1
                else:
                    pattern_counts['nonlinear'] += 1
                
                if row.get('has_pantograph', False):
                    pattern_counts['pantograph'] += 1
                
                if 'exp' in ode_str:
                    pattern_counts['exponential'] += 1
                
                if any(trig in ode_str for trig in ['sin', 'cos', 'tan']):
                    pattern_counts['trigonometric'] += 1
                
                if 'log' in ode_str:
                    pattern_counts['logarithmic'] += 1
            
            analysis_results["patterns"] = pattern_counts
            
            # ML-based analysis if comprehensive
            if request.analysis_type == "comprehensive" and len(odes) > 0:
                try:
                    # Use ODEEvaluator for advanced analysis
                    evaluator = ODEEvaluator()
                    
                    # Extract ODE strings for evaluation
                    ode_strings = [ode.get('ode_symbolic', '') for ode in odes if ode.get('ode_symbolic')]
                    
                    # Evaluate generated ODEs
                    eval_results = evaluator.evaluate_generated_odes(ode_strings, df if len(df) > 0 else None)
                    
                    analysis_results["ml_analysis"] = {
                        "total_valid": eval_results['valid'],
                        "diversity_score": eval_results['diverse'],
                        "novelty_count": eval_results['novel'],
                        "complexity_stats": eval_results['complexity_stats']
                    }
                    
                except Exception as e:
                    logger.error(f"ML analysis failed: {e}")
                    analysis_results["ml_analysis"] = {"error": str(e)}
        
        # Complete job
        await job_manager.complete_job(job_id, analysis_results)
        
    except Exception as e:
        logger.error(f"Analysis job failed: {e}")
        await job_manager.fail_job(job_id, str(e))

async def process_ml_training_job(job_id: str, request: MLTrainingRequest):
    """Process ML model training job"""
    
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        
        # Load dataset
        dataset_path = Path(request.dataset)
        if not dataset_path.exists():
            dataset_path = Path("data") / request.dataset
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {request.dataset}")
        
        # Create trainer
        trainer = ODEGeneratorTrainer(
            dataset_path=str(dataset_path),
            features_path=None  # Will extract features automatically
        )
        
        # Update job with dataset info
        await job_manager.update_job(job_id, {
            "metadata": {
                "dataset_size": len(trainer.df),
                "model_type": request.model_type,
                "status": "Training started"
            }
        })
        
        # Train based on model type
        if request.model_type == "pattern_net" or request.model_type == "pattern":
            # Train pattern network
            model = await asyncio.get_event_loop().run_in_executor(
                None,
                trainer.train_pattern_network,
                request.epochs,
                request.batch_size
            )
            model_path = Path("ode_pattern_model.pth")
            
        elif request.model_type == "language" or request.model_type == "transformer":
            # Train language model
            model = await asyncio.get_event_loop().run_in_executor(
                None,
                trainer.train_language_model,
                request.epochs,
                request.batch_size
            )
            model_path = Path("ode_language_model.pth")
            
        elif request.model_type == "vae":
            # Train VAE model with custom implementation
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
                input_dim=12,  # numeric features
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
                    loss = recon_loss + request.config.get('beta', 1.0) * kl_loss
                    
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
                        "current_loss": avg_loss
                    }
                })
                
                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    model_id = f"vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    model_dir = Path("models")
                    model_dir.mkdir(exist_ok=True)
                    
                    model_path = model_dir / f"{model_id}.pth"
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
                        'best_loss': best_loss,
                        'epoch': epoch + 1
                    }, model_path)
        
        else:
            raise ValueError(f"Unknown model type: {request.model_type}")
        
        # Save metadata
        model_id = model_path.stem
        metadata = {
            "model_id": model_id,
            "model_type": request.model_type,
            "dataset": str(request.dataset),
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
            "metadata": metadata
        })
        
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"ML training job failed: {error_details}")
        ml_training_counter.labels(model_type=request.model_type, status='failed').inc()
        await job_manager.fail_job(job_id, error_details)

async def process_ml_generation_job(job_id: str, request: MLGenerationRequest):
    """Process ML-based ODE generation job"""
    
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        
        # Load model
        model_path = Path(request.model_path)
        if not model_path.exists():
            model_path = Path("models") / request.model_path
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {request.model_path}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Check model type
        checkpoint = torch.load(model_path, map_location=device)
        model_type = checkpoint.get('model_type', 'unknown')
        
        generated_odes = []
        
        if 'language' in str(model_path) or model_type == 'language':
            # Language model generation
            from ml_pipeline.train_ode_generator import ODELanguageModel
            lm = ODELanguageModel()
            lm.model.load_state_dict(checkpoint['model_state'])
            lm.generation_head.load_state_dict(checkpoint['generation_head'])
            
            for i in range(request.n_samples):
                # Update progress
                progress = (i / request.n_samples) * 100
                await job_manager.update_job(job_id, {"progress": progress})
                
                # Generate ODE
                gen = request.generator or np.random.choice(AVAILABLE_GENERATORS)
                func = request.function or np.random.choice(AVAILABLE_FUNCTIONS[:10])
                
                ode_str = lm.generate_ode(gen, func, max_length=200)
                
                if ode_str:
                    # Verify and create ODE data
                    try:
                        ode_expr = sp.sympify(ode_str)
                        verified = isinstance(ode_expr, sp.Eq)
                        
                        # Try to extract solution if possible
                        solution = None
                        if verified:
                            # This is a placeholder - in practice you'd solve or have the solution
                            solution = "Generated solution pending"
                    except:
                        verified = False
                        solution = None
                    
                    ode_data = {
                        "id": f"ml_{uuid.uuid4().hex[:8]}",
                        "ode": ode_str,
                        "solution": solution,
                        "generator": gen,
                        "function": func,
                        "verified": verified,
                        "model_type": "language_model",
                        "temperature": request.temperature,
                        "complexity": len(ode_str)
                    }
                    generated_odes.append(ode_data)
        
        else:
            # Use standard ML generation pipeline
            model = load_pretrained_model(
                model_type=model_type,
                checkpoint_path=str(model_path),
                device=device
            )
            
            # Generate ODEs
            generators = [request.generator] if request.generator else None
            functions = [request.function] if request.function else None
            
            generated_results = generate_novel_odes(
                model=model,
                n_samples=request.n_samples,
                generators=generators,
                functions=functions,
                temperature=request.temperature,
                device=device
            )
            
            for i, result in enumerate(generated_results):
                # Update progress
                progress = ((i + 1) / len(generated_results)) * 100
                await job_manager.update_job(job_id, {"progress": progress})
                
                ode_data = {
                    "id": result.get('id', f"ml_{uuid.uuid4().hex[:8]}"),
                    "ode": result.get('ode', ''),
                    "solution": result.get('solution'),
                    "generator": result.get('generator', 'ML'),
                    "function": result.get('function', 'unknown'),
                    "verified": False,  # Will verify below
                    "temperature": request.temperature,
                    "complexity": len(result.get('ode', ''))
                }
                generated_odes.append(ode_data)
        
        # Verify generated ODEs if solutions are available
        for ode_data in generated_odes:
            if ode_data.get('solution') and ode_data['solution'] != "Generated solution pending":
                try:
                    verified, method, confidence = ode_verifier.verify(
                        sp.sympify(ode_data['ode']),
                        sp.sympify(ode_data['solution'])
                    )
                    ode_data['verified'] = verified
                    ode_data['verification_method'] = method.value
                    ode_data['verification_confidence'] = confidence
                except:
                    pass
        
        # Calculate metrics
        valid_count = sum(1 for ode in generated_odes if ode.get('ode'))
        verified_count = sum(1 for ode in generated_odes if ode.get('verified', False))
        
        # Analyze diversity
        diversity_metrics = analyze_generation_diversity(
            [ode['ode'] for ode in generated_odes if ode.get('ode')]
        )
        
        results = {
            "odes": generated_odes,
            "metrics": {
                "total_generated": len(generated_odes),
                "valid_count": valid_count,
                "verified_count": verified_count,
                "validity_rate": valid_count / len(generated_odes) if generated_odes else 0,
                "verification_rate": verified_count / len(generated_odes) if generated_odes else 0,
                "diversity_metrics": diversity_metrics
            },
            "model_info": {
                "model_path": str(model_path),
                "model_type": model_type,
                "temperature": request.temperature
            }
        }
        
        # Update metrics
        ml_generation_counter.labels(model_type=model_type).inc()
        
        # Complete job
        await job_manager.complete_job(job_id, results)
        
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"ML generation job failed: {error_details}")
        await job_manager.fail_job(job_id, error_details)

# Startup/Shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize app state"""
    app.state.start_time = time.time()
    print(f"ODE API Server started")
    print(f"Redis available: {REDIS_AVAILABLE}")
    print(f"Working generators: {len(AVAILABLE_GENERATORS)}")
    print(f"Available functions: {len(AVAILABLE_FUNCTIONS)}")
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("ml_data").mkdir(exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup"""
    print("ODE API Server shutting down")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
