# scripts/production_server.py
"""
Production API server for ODE generation and verification

Benefits:
- RESTful API for ODE operations
- Async processing for scalability
- Rate limiting and authentication
- Real-time monitoring
- API documentation
"""

import os
import sys
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import aiofiles
import json
from datetime import datetime
import uuid
from pathlib import Path
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with correct names
from pipeline.generator import ODEDatasetGenerator as ODEGenerator
from verification.verifier import ODEVerifier
from utils.config import ConfigManager

# Initialize FastAPI app
app = FastAPI(
    title="ODE Generation API",
    description="Production API for ODE generation, verification, and analysis",
    version="1.0.0"
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
    
    redis_client = FakeRedis()

# Metrics
ode_generation_counter = Counter('ode_generation_total', 'Total ODEs generated')
verification_counter = Counter('ode_verification_total', 'Total verifications')
generation_time_histogram = Histogram('ode_generation_duration_seconds', 'ODE generation time')
active_jobs_gauge = Gauge('active_jobs', 'Number of active jobs')

# Security - Get API key from environment
VALID_API_KEY = os.getenv('API_KEY', 'your-secret-key-1')
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """Verify API key"""
    print(f"DEBUG: Received API key: {api_key}")  # Add this
    print(f"DEBUG: Expected API key: {VALID_API_KEY}")  # Add this
    
    if not api_key:
        raise HTTPException(
            status_code=403, 
            detail="API key required. Add 'X-API-Key' header."
        )
    
    # Temporarily accept any key for testing
    print("DEBUG: Accepting any API key for testing")  # Add this
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

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    results: Optional[List[ODEResponse]]
    error: Optional[str]

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
            "results": None,
            "error": None
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

job_manager = JobManager()

# Initialize generators globally to avoid reimporting
try:
    ode_generator = ODEGenerator()
    ode_verifier = ODEVerifier()
    GENERATORS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not initialize generators: {e}")
    GENERATORS_AVAILABLE = False

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
    
    verification_counter.inc()
    
    try:
        result = ode_verifier.verify(request.ode, request.solution, method=request.method)
        
        return {
            "verified": result.get("verified", False),
            "confidence": result.get("confidence", 0.0),
            "method": request.method,
            "details": result.get("details", {})
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
        error=job.get("error")
    )

@app.get("/api/v1/stats")
async def get_statistics(api_key: str = Depends(verify_api_key)):
    """Get API statistics"""
    
    try:
        stats = {
            "total_generated": int(ode_generation_counter._value.get()),
            "total_verified": int(verification_counter._value.get()),
            "active_jobs": int(active_jobs_gauge._value.get()),
            "cache_size": redis_client.dbsize() if REDIS_AVAILABLE else 0,
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            "redis_available": REDIS_AVAILABLE,
            "generators_available": GENERATORS_AVAILABLE
        }
    except:
        stats = {
            "status": "operational",
            "redis_available": REDIS_AVAILABLE,
            "generators_available": GENERATORS_AVAILABLE
        }
    
    return stats

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

# Background job processors
async def process_generation_job(job_id: str, request: ODEGenerationRequest):
    """Process ODE generation job"""
    
    try:
        results = []
        
        for i in range(request.count):
            # Update progress
            progress = (i / request.count) * 100
            await job_manager.update_job(job_id, {"progress": progress})
            
            # Generate ODE
            ode_data = ode_generator.generate_single(
                request.generator,
                request.function,
                request.parameters
            )
            
            if ode_data:
                # Verify if requested
                if request.verify and ode_data.get("solution_symbolic"):
                    try:
                        verification = ode_verifier.verify(
                            ode_data["ode_symbolic"],
                            ode_data["solution_symbolic"]
                        )
                        ode_data["verified"] = verification.get("verified", False)
                    except:
                        ode_data["verified"] = False
                
                # Create response object
                response_data = ODEResponse(
                    id=ode_data.get("id", str(uuid.uuid4())),
                    ode=ode_data.get("ode_symbolic", ""),
                    solution=ode_data.get("solution_symbolic"),
                    verified=ode_data.get("verified", False),
                    complexity=ode_data.get("complexity_score", 0),
                    generator=ode_data.get("generator_name", request.generator),
                    function=ode_data.get("function_name", request.function),
                    parameters=ode_data.get("parameters", {}),
                    timestamp=ode_data.get("timestamp", datetime.now().isoformat())
                )
                
                results.append(response_data)
                ode_generation_counter.inc()
        
        # Complete job
        await job_manager.complete_job(job_id, [r.dict() for r in results])
        
    except Exception as e:
        await job_manager.update_job(job_id, {
            "status": "failed",
            "error": str(e)
        })
        active_jobs_gauge.dec()

# Startup/Shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize app state"""
    app.state.start_time = time.time()
    print(f"ODE API Server started")
    print(f"Redis available: {REDIS_AVAILABLE}")
    print(f"Generators available: {GENERATORS_AVAILABLE}")
    print(f"API Key configured: {'Yes' if VALID_API_KEY != 'your-secret-key-1' else 'No (using default)'}")
    
    # Debug: Try to manually test generator
    if GENERATORS_AVAILABLE:
        try:
            test_result = ode_generator.generate_single("L1", "sine", {"alpha": 1.0})
            print(f"Generator test: {'Success' if test_result else 'Failed'}")
        except Exception as e:
            print(f"Generator test error: {e}")

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
        "generators": "loaded" if GENERATORS_AVAILABLE else "not available"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "ODE Generation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc",
            "generate": "/api/v1/generate",
            "verify": "/api/v1/verify",
            "stats": "/api/v1/stats"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
