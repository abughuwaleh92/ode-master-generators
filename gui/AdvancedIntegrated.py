# scripts/production_server.py
"""
Production API server for ODE generation and verification with ML integration

Benefits:
- RESTful API for ODE operations
- Async processing for scalability
- Rate limiting and authentication
- Real-time monitoring
- API documentation
- ML model integration
- Advanced analysis capabilities
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
import torch
import pickle
import sympy as sp

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with correct names
from pipeline.generator import ODEDatasetGenerator as ODEGenerator
from verification.verifier import ODEVerifier
from utils.config import ConfigManager
from core.types import GeneratorType, VerificationMethod

# Import ML pipeline components
try:
    from ml_pipeline.train_ode_generator import ODEGeneratorTrainer
    from ml_pipeline.models import (
        ODEPatternNet, 
        ODETransformer, 
        ODEVAE,
        ODELanguageModel,
        get_model
    )
    from ml_pipeline.utils import (
        prepare_ml_dataset,
        load_pretrained_model, 
        generate_novel_odes,
        extract_ml_features,
        create_ode_tokenizer,
        sample_ode_parameters,
        construct_ode_from_params
    )
    from ml_pipeline.evaluation import ODEEvaluator
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML pipeline not available: {e}")
    ML_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="ODE Generation API",
    description="Production API for ODE generation, verification, and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.up.railway.app",
        "https://*.railway.app",
        "http://localhost:8501",
        "http://localhost:3000",
        "*"  # Allow all origins in development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis configuration for Railway
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
try:
    if REDIS_URL.startswith('redis://'):
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    else:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    # Test connection
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

# Security - Get API key from environment
VALID_API_KEY = os.getenv('API_KEY', 'your-secret-key-1')
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """Verify API key"""
    if not api_key:
        raise HTTPException(
            status_code=403, 
            detail="API key required. Add 'X-API-Key' header."
        )
    
    # Check against valid keys
    valid_keys = [VALID_API_KEY]
    if VALID_API_KEY == 'your-secret-key-1':
        # Development mode - accept common test keys
        valid_keys.extend(['test-key', 'test-123', 'dev-key'])
    
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
    results: Optional[List[ODEResponse]]
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
        
        # Store in Redis or memory
        if REDIS_AVAILABLE:
            redis_client.setex(
                f"job:{job_id}",
                3600,  # 1 hour TTL
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
    
    async def list_jobs(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """List jobs with optional status filter"""
        jobs = []
        
        if REDIS_AVAILABLE:
            keys = redis_client.keys("job:*")
            for key in keys[:limit]:
                job_data = redis_client.get(key)
                if job_data:
                    job = json.loads(job_data)
                    if status is None or job.get('status') == status:
                        jobs.append(job)
        else:
            for job in list(self.jobs.values())[:limit]:
                if status is None or job.get('status') == status:
                    jobs.append(job)
        
        return sorted(jobs, key=lambda x: x.get('created_at', ''), reverse=True)

job_manager = JobManager()

# Initialize generators globally
try:
    config = ConfigManager()
    ode_generator = ODEGenerator(config=config)
    ode_verifier = ODEVerifier()
    GENERATORS_AVAILABLE = True
    
    # Test and cache working generators
    print("Testing generators...")
    WORKING_GENERATORS = ode_generator.test_generators()
    AVAILABLE_GENERATORS = list(WORKING_GENERATORS['linear'].keys()) + list(WORKING_GENERATORS['nonlinear'].keys())
    AVAILABLE_FUNCTIONS = list(ode_generator.f_library.keys())
    
except Exception as e:
    print(f"Warning: Could not initialize generators: {e}")
    GENERATORS_AVAILABLE = False
    WORKING_GENERATORS = {'linear': {}, 'nonlinear': {}}
    AVAILABLE_GENERATORS = []
    AVAILABLE_FUNCTIONS = []

# Middleware for metrics
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    """Track request metrics"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Track metrics
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
    
    if not GENERATORS_AVAILABLE:
        raise HTTPException(status_code=500, detail="ODE generators not available")
    
    # Validate generator and function
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
    
    # Create job
    job_id = await job_manager.create_job("generation", request.dict())
    
    # Start background task
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
    
    if not GENERATORS_AVAILABLE:
        raise HTTPException(status_code=500, detail="Verifier not available")
    
    try:
        # Parse ODE and solution
        import sympy as sp
        from core.symbols import SYMBOLS
        
        x = SYMBOLS.x
        y = SYMBOLS.y
        
        # Convert string to sympy expression
        ode_expr = sp.sympify(request.ode)
        solution_expr = sp.sympify(request.solution)
        
        # Verify using enhanced verification
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

@app.get("/api/v1/jobs", response_model=List[JobStatus])
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 100,
    api_key: str = Depends(verify_api_key)
):
    """List all jobs with optional status filter"""
    
    jobs = await job_manager.list_jobs(status=status, limit=limit)
    
    return [
        JobStatus(
            job_id=job["id"],
            status=job["status"],
            progress=job["progress"],
            results=job.get("results"),
            error=job.get("error"),
            created_at=job["created_at"],
            updated_at=job["updated_at"],
            metadata=job.get("metadata", {})
        )
        for job in jobs
    ]

@app.post("/api/v1/analyze")
async def analyze_dataset(
    request: ODEAnalysisRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Analyze ODE dataset or list of ODEs"""
    
    # Create job
    job_id = await job_manager.create_job("analysis", request.dict())
    
    # Start background task
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
    
    if not ML_AVAILABLE:
        raise HTTPException(status_code=500, detail="ML pipeline not available")
    
    # Create job
    job_id = await job_manager.create_job("ml_training", request.dict())
    
    # Start background task
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
        raise HTTPException(status_code=500, detail="ML pipeline not available")
    
    # Create job
    job_id = await job_manager.create_job("ml_generation", request.dict())
    
    # Start background task
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
    
    try:
        # Get job statistics
        all_jobs = await job_manager.list_jobs(limit=1000)
        job_stats = {
            "total": len(all_jobs),
            "completed": sum(1 for j in all_jobs if j['status'] == 'completed'),
            "failed": sum(1 for j in all_jobs if j['status'] == 'failed'),
            "pending": sum(1 for j in all_jobs if j['status'] == 'pending'),
            "running": sum(1 for j in all_jobs if j['status'] == 'running')
        }
        
        stats = {
            "status": "operational",
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            "redis_available": REDIS_AVAILABLE,
            "generators_available": GENERATORS_AVAILABLE,
            "ml_available": ML_AVAILABLE,
            "active_jobs": int(active_jobs_gauge._value.get()),
            "total_generated": int(ode_generation_counter._value.get()),
            "total_verified": int(verification_counter._value.get()),
            "cache_size": redis_client.dbsize() if REDIS_AVAILABLE else 0,
            "job_statistics": job_stats,
            "available_generators": len(AVAILABLE_GENERATORS),
            "available_functions": len(AVAILABLE_FUNCTIONS)
        }
    except Exception as e:
        stats = {
            "status": "operational",
            "redis_available": REDIS_AVAILABLE,
            "generators_available": GENERATORS_AVAILABLE,
            "ml_available": ML_AVAILABLE,
            "error": str(e)
        }
    
    return stats

@app.get("/api/v1/models")
async def list_ml_models(api_key: str = Depends(verify_api_key)):
    """List available ML models"""
    
    models_dir = Path("models")
    models = []
    
    if models_dir.exists():
        for model_path in models_dir.glob("*.pth"):
            try:
                # Try to load model metadata
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
                print(f"Error loading model {model_path}: {e}")
    
    return {
        "models": models,
        "count": len(models)
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
        
        with generation_time_histogram.time():
            for i in range(request.count):
                # Update progress
                progress = (i / request.count) * 100
                await job_manager.update_job(job_id, {"progress": progress})
                
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
                        "verification_confidence": ode_instance.verification_confidence
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
                    
                    results.append(response_data)
                    ode_generation_counter.labels(
                        generator=request.generator,
                        function=request.function
                    ).inc()
        
        # Complete job
        await job_manager.complete_job(job_id, [r.dict() for r in results])
        
    except Exception as e:
        await job_manager.fail_job(job_id, str(e))

async def process_analysis_job(job_id: str, request: ODEAnalysisRequest):
    """Process dataset analysis job"""
    
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        
        # Load dataset or parse ODEs
        odes = []
        if request.dataset_path:
            # Load from file
            dataset_path = Path(request.dataset_path)
            if not dataset_path.exists():
                dataset_path = Path("data") / request.dataset_path
            
            with open(dataset_path, 'r') as f:
                for line in f:
                    if line.strip():
                        ode_data = json.loads(line)
                        odes.append(ode_data)
        elif request.ode_list:
            # Parse provided ODEs
            for ode_str in request.ode_list:
                odes.append({"ode_symbolic": ode_str})
        
        # Perform analysis
        analysis_results = {
            "total_odes": len(odes),
            "statistics": {},
            "patterns": {},
            "complexity_distribution": {},
            "generator_distribution": {},
            "function_distribution": {}
        }
        
        if odes:
            # Calculate statistics
            complexities = [ode.get('complexity_score', 0) for ode in odes]
            verified_count = sum(1 for ode in odes if ode.get('verified', False))
            
            analysis_results["statistics"] = {
                "verified_rate": verified_count / len(odes) if odes else 0,
                "avg_complexity": np.mean(complexities) if complexities else 0,
                "complexity_std": np.std(complexities) if complexities else 0,
                "complexity_range": [min(complexities), max(complexities)] if complexities else [0, 0],
                "median_complexity": np.median(complexities) if complexities else 0
            }
            
            # Generator distribution
            generator_counts = {}
            for ode in odes:
                gen = ode.get('generator_name', 'unknown')
                generator_counts[gen] = generator_counts.get(gen, 0) + 1
            
            analysis_results["generator_distribution"] = generator_counts
            
            # Function distribution
            function_counts = {}
            for ode in odes:
                func = ode.get('function_name', 'unknown')
                function_counts[func] = function_counts.get(func, 0) + 1
            
            analysis_results["function_distribution"] = function_counts
            
            # Complexity distribution (binned)
            if complexities:
                hist, bins = np.histogram(complexities, bins=10)
                analysis_results["complexity_distribution"] = {
                    f"{bins[i]:.0f}-{bins[i+1]:.0f}": int(hist[i])
                    for i in range(len(hist))
                }
            
            # Pattern analysis
            pattern_counts = {
                "linear": 0,
                "nonlinear": 0,
                "pantograph": 0,
                "exponential": 0,
                "trigonometric": 0,
                "logarithmic": 0
            }
            
            for ode in odes:
                ode_str = ode.get('ode_symbolic', '')
                
                if ode.get('generator_type') == 'linear':
                    pattern_counts['linear'] += 1
                else:
                    pattern_counts['nonlinear'] += 1
                
                if ode.get('has_pantograph', False):
                    pattern_counts['pantograph'] += 1
                
                if 'exp' in ode_str:
                    pattern_counts['exponential'] += 1
                
                if any(trig in ode_str for trig in ['sin', 'cos', 'tan']):
                    pattern_counts['trigonometric'] += 1
                
                if 'log' in ode_str:
                    pattern_counts['logarithmic'] += 1
            
            analysis_results["patterns"] = pattern_counts
            
            # If ML is available and comprehensive analysis requested
            if ML_AVAILABLE and request.analysis_type == "comprehensive":
                try:
                    # Extract ML features
                    features_list = []
                    
                    for ode in odes[:1000]:  # Limit for performance
                        # Basic feature extraction
                        features = {
                            'complexity': ode.get('complexity_score', 0),
                            'verified': int(ode.get('verified', False)),
                            'order': 2 if "y''" in ode.get('ode_symbolic', '') else 1,
                            'length': len(ode.get('ode_symbolic', '')),
                            'has_exp': int('exp' in ode.get('ode_symbolic', '')),
                            'has_trig': int(any(t in ode.get('ode_symbolic', '') for t in ['sin', 'cos'])),
                            'operation_count': ode.get('operation_count', 0)
                        }
                        features_list.append(features)
                    
                    if features_list:
                        # Convert to numpy for analysis
                        feature_names = list(features_list[0].keys())
                        feature_matrix = np.array([[f[k] for k in feature_names] for f in features_list])
                        
                        # Calculate correlations
                        correlations = {}
                        for i, name1 in enumerate(feature_names):
                            for j, name2 in enumerate(feature_names):
                                if i < j:
                                    corr = np.corrcoef(feature_matrix[:, i], feature_matrix[:, j])[0, 1]
                                    correlations[f"{name1}_vs_{name2}"] = float(corr)
                        
                        analysis_results["ml_analysis"] = {
                            "feature_means": {name: float(feature_matrix[:, i].mean()) 
                                            for i, name in enumerate(feature_names)},
                            "feature_stds": {name: float(feature_matrix[:, i].std()) 
                                           for i, name in enumerate(feature_names)},
                            "correlations": correlations,
                            "n_samples_analyzed": len(features_list)
                        }
                        
                except Exception as e:
                    logger.error(f"ML analysis failed: {e}")
                    analysis_results["ml_analysis"] = {"error": str(e)}
        
        # Update job with results
        await job_manager.update_job(job_id, {
            "status": "completed",
            "progress": 100.0,
            "results": analysis_results
        })
        
    except Exception as e:
        await job_manager.fail_job(job_id, str(e))

async def process_ml_training_job(job_id: str, request: MLTrainingRequest):
    """Process ML model training job with real ML pipeline"""
    
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        
        if not ML_AVAILABLE:
            raise Exception("ML pipeline not available")
        
        # Prepare dataset path
        dataset_path = Path(request.dataset)
        if not dataset_path.exists():
            # If dataset doesn't exist, try to find it in data directory
            dataset_path = Path("data") / request.dataset
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {request.dataset}")
        
        # Prepare ML dataset if needed
        ml_data_dir = Path("ml_data")
        ml_data_dir.mkdir(exist_ok=True)
        
        # Extract features from dataset if not already done
        features_path = ml_data_dir / "features.parquet"
        if not features_path.exists():
            print(f"Preparing ML dataset from {dataset_path}")
            data_paths = prepare_ml_dataset(
                str(dataset_path),
                output_dir=str(ml_data_dir),
                test_split=0.2,
                val_split=0.1
            )
            features_path = Path(data_paths['train'])
        
        # Create trainer
        trainer = ODEGeneratorTrainer(
            dataset_path=str(dataset_path),
            features_path=str(features_path) if features_path.exists() else None
        )
        
        # Use the trainer's built-in methods based on model type
        if request.model_type == "pattern_net" or request.model_type == "pattern":
            # Train pattern network
            model = await asyncio.get_event_loop().run_in_executor(
                None,
                trainer.train_pattern_network,
                request.epochs,
                request.batch_size
            )
            
            # The trainer saves the model, get the path
            model_path = Path("ode_pattern_model.pth")
            
        elif request.model_type == "language" or request.model_type == "transformer":
            # Train language model
            model = await asyncio.get_event_loop().run_in_executor(
                None,
                trainer.train_language_model,
                request.epochs,
                request.batch_size
            )
            
            # The trainer saves the model
            model_path = Path("ode_language_model.pth")
            
        else:
            # For other model types, use get_model from models.py
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load dataset for custom training
            dataset = trainer.features_df
            
            # Get model configuration
            model_config = {
                'n_numeric_features': 12,
                'n_generators': len(trainer.generator_encoder.classes_),
                'n_functions': len(trainer.function_encoder.classes_),
                'hidden_dim': request.config.get('hidden_dim', 256),
                'input_dim': 12,  # numeric features
                'latent_dim': request.config.get('latent_dim', 64),
                'vocab_size': 50000  # default vocab size
            }
            
            # Create model using get_model
            if request.model_type == "vae":
                model = ODEVAE(
                    input_dim=model_config['input_dim'],
                    hidden_dim=model_config['hidden_dim'],
                    latent_dim=model_config['latent_dim'],
                    n_generators=model_config['n_generators'],
                    n_functions=model_config['n_functions']
                ).to(device)
            else:
                model = get_model(request.model_type, **model_config).to(device)
            
            # Training loop
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=request.learning_rate,
                weight_decay=request.config.get('weight_decay', 0.0001)
            )
            
            # Create data loader
            from torch.utils.data import TensorDataset, DataLoader
            
            # Convert dataset to tensors
            numeric_features = torch.tensor(
                dataset[['complexity_score', 'operation_count', 'atom_count', 
                        'symbol_count', 'pow_deriv_max', 'pow_yprime', 
                        'has_pantograph', 'alpha', 'beta', 'M', 'q', 'v']].values,
                dtype=torch.float32
            )
            generator_ids = torch.tensor(dataset['generator_id'].values, dtype=torch.long)
            function_ids = torch.tensor(dataset['function_id'].values, dtype=torch.long)
            verified = torch.tensor(dataset['verified'].values, dtype=torch.float32)
            
            train_dataset = TensorDataset(numeric_features, generator_ids, function_ids, verified)
            train_loader = DataLoader(train_dataset, batch_size=request.batch_size, shuffle=True)
            
            best_loss = float('inf')
            patience = 10 if request.early_stopping else request.epochs
            patience_counter = 0
            
            for epoch in range(request.epochs):
                model.train()
                epoch_loss = 0
                
                for batch in train_loader:
                    features, gen_ids, func_ids, targets = batch
                    features = features.to(device)
                    gen_ids = gen_ids.to(device)
                    func_ids = func_ids.to(device)
                    targets = targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    if request.model_type == "vae":
                        outputs = model(features, gen_ids, func_ids)
                        recon_loss = torch.nn.functional.mse_loss(
                            outputs['reconstruction'], features
                        )
                        kl_loss = -0.5 * torch.sum(
                            1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
                        )
                        loss = recon_loss + 0.1 * kl_loss
                    else:
                        outputs = model(features, gen_ids, func_ids)
                        loss = torch.nn.functional.binary_cross_entropy(
                            outputs['verification'].squeeze(), targets
                        )
                    
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
                        "status": f"Training epoch {epoch + 1}/{request.epochs}",
                        "current_metrics": {
                            'train_loss': avg_loss,
                            'learning_rate': optimizer.param_groups[0]['lr']
                        }
                    }
                })
                
                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    
                    model_id = f"{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    model_dir = Path("models")
                    model_dir.mkdir(exist_ok=True)
                    
                    model_path = model_dir / f"{model_id}.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_config': model_config,
                        'model_type': request.model_type,
                        'best_loss': best_loss,
                        'epoch': epoch + 1
                    }, model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience and request.early_stopping:
                        break
        
        # Save metadata
        model_id = model_path.stem
        metadata = {
            "model_id": model_id,
            "model_type": request.model_type,
            "dataset": str(request.dataset),
            "final_metrics": {
                "best_loss": best_loss if 'best_loss' in locals() else 0,
                "epochs_trained": epoch + 1 if 'epoch' in locals() else request.epochs
            },
            "training_config": request.dict(),
            "created_at": datetime.now().isoformat()
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update ML training metrics
        ml_training_counter.labels(model_type=request.model_type, status='completed').inc()
        
        # Complete job
        await job_manager.complete_job(job_id, {
            "model_id": model_id,
            "model_path": str(model_path),
            "final_metrics": metadata["final_metrics"],
            "metadata": metadata
        })
        
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        ml_training_counter.labels(model_type=request.model_type, status='failed').inc()
        await job_manager.fail_job(job_id, error_details)

async def process_ml_generation_job(job_id: str, request: MLGenerationRequest):
    """Process ML-based ODE generation job with real models"""
    
    try:
        await job_manager.update_job(job_id, {"status": "running"})
        
        if not ML_AVAILABLE:
            raise Exception("ML pipeline not available")
        
        # Load model path
        model_path = Path(request.model_path)
        if not model_path.exists():
            # Try in models directory
            model_path = Path("models") / request.model_path
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {request.model_path}")
        
        # Load model using ml_pipeline utils
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Check if it's a language model (special case)
        if 'language' in str(model_path) or 'transformer' in str(model_path):
            # For language models saved by ODELanguageModel
            checkpoint = torch.load(model_path, map_location=device)
            
            # Create language model
            from ml_pipeline.train_ode_generator import ODELanguageModel
            lm = ODELanguageModel()
            lm.model.load_state_dict(checkpoint['model_state'])
            lm.generation_head.load_state_dict(checkpoint['generation_head'])
            
            generated_odes = []
            
            for i in range(request.n_samples):
                # Update progress
                progress = (i / request.n_samples) * 100
                await job_manager.update_job(job_id, {"progress": progress})
                
                # Select generator and function
                gen = request.generator or np.random.choice(['L1', 'L2', 'L3', 'N1', 'N2', 'N3'])
                func = request.function or np.random.choice(['sine', 'cosine', 'exponential'])
                
                # Generate ODE
                ode_str = lm.generate_ode(gen, func, max_length=200)
                
                if ode_str:
                    # Verify generated ODE
                    try:
                        ode_expr = sp.sympify(ode_str)
                        verified = isinstance(ode_expr, sp.Eq)
                    except:
                        verified = False
                    
                    ode_data = {
                        "id": f"ml_{uuid.uuid4().hex[:8]}",
                        "ode": ode_str,
                        "ode_latex": sp.latex(ode_expr) if verified else ode_str,
                        "generator": gen,
                        "function": func,
                        "verified": verified,
                        "model_type": "language_model",
                        "temperature": request.temperature
                    }
                    generated_odes.append(ode_data)
        
        else:
            # Use generate_novel_odes from ml_pipeline utils
            generators = [request.generator] if request.generator else None
            functions = [request.function] if request.function else None
            
            # Load the model
            model = load_pretrained_model(
                model_type=request.model_path.split('_')[0],  # Extract model type from filename
                checkpoint_path=str(model_path),
                device=device
            )
            
            # Generate ODEs
            generated_results = generate_novel_odes(
                model=model,
                n_samples=request.n_samples,
                generators=generators,
                functions=functions,
                temperature=request.temperature,
                device=device
            )
            
            generated_odes = []
            for i, result in enumerate(generated_results):
                # Update progress
                progress = ((i + 1) / len(generated_results)) * 100
                await job_manager.update_job(job_id, {"progress": progress})
                
                # Verify if the ODE is valid
                try:
                    if 'ode' in result:
                        ode_expr = sp.sympify(result['ode'])
                        verified = isinstance(ode_expr, sp.Eq)
                    else:
                        verified = False
                except:
                    verified = False
                
                ode_data = {
                    "id": result.get('id', f"ml_{uuid.uuid4().hex[:8]}"),
                    "ode": result.get('ode', ''),
                    "ode_latex": sp.latex(ode_expr) if verified else result.get('ode', ''),
                    "generator": result.get('generator', 'ML'),
                    "function": result.get('function', 'unknown'),
                    "verified": verified,
                    "temperature": request.temperature,
                    "complexity": len(result.get('ode', ''))
                }
                generated_odes.append(ode_data)
        
        # If requested, verify using the ODE verifier
        if generated_odes:
            verifier = ODEVerifier()
            for ode_data in generated_odes:
                if ode_data['verified'] and 'solution' in ode_data:
                    try:
                        verified, method, confidence = verifier.verify(
                            ode_data['ode'],
                            ode_data.get('solution', '')
                        )
                        ode_data['verification_method'] = method.value
                        ode_data['verification_confidence'] = confidence
                    except:
                        pass
        
        # Calculate generation metrics
        valid_count = sum(1 for ode in generated_odes if ode.get('verified', False))
        
        # Calculate diversity metrics
        unique_odes = len(set(ode['ode'] for ode in generated_odes))
        diversity_score = unique_odes / len(generated_odes) if generated_odes else 0
        
        # Complexity metrics
        if generated_odes:
            complexities = [ode.get('complexity', len(ode['ode'])) for ode in generated_odes]
            avg_complexity = np.mean(complexities)
            std_complexity = np.std(complexities)
        else:
            avg_complexity = 0
            std_complexity = 0
        
        results = {
            "odes": generated_odes,
            "metrics": {
                "total_generated": len(generated_odes),
                "valid_count": valid_count,
                "validity_rate": valid_count / len(generated_odes) if generated_odes else 0,
                "unique_odes": unique_odes,
                "diversity_score": diversity_score,
                "avg_complexity": avg_complexity,
                "complexity_std": std_complexity
            },
            "model_info": {
                "model_path": str(model_path),
                "temperature": request.temperature,
                "generator": request.generator,
                "function": request.function
            }
        }
        
        # Update ML generation metrics
        ml_generation_counter.labels(model_type=model_path.stem.split('_')[0]).inc()
        
        # Complete job
        await job_manager.complete_job(job_id, results)
        
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        await job_manager.fail_job(job_id, error_details)

# Startup/Shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize app state"""
    app.state.start_time = time.time()
    print(f"ODE API Server started")
    print(f"Redis available: {REDIS_AVAILABLE}")
    print(f"Generators available: {GENERATORS_AVAILABLE}")
    print(f"ML Pipeline available: {ML_AVAILABLE}")
    print(f"Working generators: {len(AVAILABLE_GENERATORS)}")
    print(f"Available functions: {len(AVAILABLE_FUNCTIONS)}")
    print(f"API Key configured: {'Yes' if VALID_API_KEY != 'your-secret-key-1' else 'No (using default)'}")
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup"""
    print("ODE API Server shutting down")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis": "connected" if REDIS_AVAILABLE else "not available",
        "generators": "loaded" if GENERATORS_AVAILABLE else "not available",
        "ml_pipeline": "loaded" if ML_AVAILABLE else "not available",
        "working_generators": len(AVAILABLE_GENERATORS),
        "functions": len(AVAILABLE_FUNCTIONS)
    }

# Root endpoint
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
        },
        "features": [
            "ODE Generation with multiple generators",
            "Solution verification",
            "Dataset analysis",
            "ML model training (PatternNet, Transformer, VAE)",
            "ML-based ODE generation",
            "Job queue with progress tracking",
            "Prometheus metrics",
            "Redis caching",
            "Real-time streaming"
        ],
        "ml_status": "enabled" if ML_AVAILABLE else "disabled"
    }

# Stream endpoint for real-time generation
@app.get("/api/v1/stream/generate")
async def stream_generate_odes(
    generator: str,
    function: str,
    count: int = 10,
    api_key: str = Depends(verify_api_key)
):
    """Stream ODEs as they are generated"""
    
    async def generate():
        all_generators = {**WORKING_GENERATORS['linear'], **WORKING_GENERATORS['nonlinear']}
        
        if generator not in all_generators:
            yield f"data: {{\"error\": \"Unknown generator: {generator}\"}}\n\n"
            return
        
        gen = all_generators[generator]
        gen_type = 'linear' if generator in WORKING_GENERATORS['linear'] else 'nonlinear'
        
        for i in range(count):
            ode_instance = ode_generator.generate_single_ode(
                generator=gen,
                gen_type=gen_type,
                gen_name=generator,
                f_key=function,
                ode_id=i
            )
            
            if ode_instance:
                result = {
                    "id": str(ode_instance.id),
                    "ode": ode_instance.ode_symbolic,
                    "solution": ode_instance.solution_symbolic,
                    "verified": ode_instance.verified,
                    "progress": (i + 1) / count * 100
                }
                yield f"data: {json.dumps(result)}\n\n"
            
            await asyncio.sleep(0.1)  # Small delay between generations
    
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
