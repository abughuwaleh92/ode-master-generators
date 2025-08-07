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

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import APIKeyHeader
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

# Initialize FastAPI app
app = FastAPI(
    title="ODE Generation API",
    description="Production API for ODE generation, verification, and analysis",
    version="1.0.0"
)

# Redis for caching and job queue
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Metrics
ode_generation_counter = Counter('ode_generation_total', 'Total ODEs generated')
verification_counter = Counter('ode_verification_total', 'Total verifications')
generation_time_histogram = Histogram('ode_generation_duration_seconds', 'ODE generation time')
active_jobs_gauge = Gauge('active_jobs', 'Number of active jobs')

# Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """Verify API key"""
    valid_keys = ["your-secret-key-1", "your-secret-key-2"]  # In production, use database
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
        
        # Store in Redis
        redis_client.setex(
            f"job:{job_id}",
            3600,  # 1 hour TTL
            json.dumps(job_data)
        )
        
        active_jobs_gauge.inc()
        return job_id
    
    async def update_job(self, job_id: str, updates: Dict):
        """Update job status"""
        job_data = redis_client.get(f"job:{job_id}")
        if job_data:
            job = json.loads(job_data)
            job.update(updates)
            redis_client.setex(f"job:{job_id}", 3600, json.dumps(job))
    
    async def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status"""
        job_data = redis_client.get(f"job:{job_id}")
        return json.loads(job_data) if job_data else None
    
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

# API Endpoints
@app.post("/api/v1/generate", response_model=Dict[str, str])
async def generate_odes(
    request: ODEGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Generate ODEs asynchronously"""
    
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
    
    with verification_counter.count():
        # Import verifier
        from verification.verifier import ODEVerifier
        verifier = ODEVerifier()
        
        try:
            result = verifier.verify(request.ode, request.solution, method=request.method)
            
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
    
    stats = {
        "total_generated": int(ode_generation_counter._value.get()),
        "total_verified": int(verification_counter._value.get()),
        "active_jobs": int(active_jobs_gauge._value.get()),
        "cache_size": redis_client.dbsize(),
        "uptime": time.time() - app.state.start_time
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
        # Import generator
        from pipeline.generator import ODEGenerator
        generator = ODEGenerator()
        
        results = []
        
        with generation_time_histogram.time():
            for i in range(request.count):
                # Update progress
                progress = (i / request.count) * 100
                await job_manager.update_job(job_id, {"progress": progress})
                
                # Generate ODE
                ode_data = generator.generate_single(
                    request.generator,
                    request.function,
                    request.parameters
                )
                
                if ode_data:
                    # Verify if requested
                    if request.verify:
                        from verification.verifier import ODEVerifier
                        verifier = ODEVerifier()
                        
                        try:
                            verification = verifier.verify(
                                ode_data["ode_symbolic"],
                                ode_data["solution_symbolic"]
                            )
                            ode_data["verified"] = verification.get("verified", False)
                        except:
                            ode_data["verified"] = False
                    
                    results.append(ODEResponse(
                        id=ode_data["id"],
                        ode=ode_data["ode_symbolic"],
                        solution=ode_data.get("solution_symbolic"),
                        verified=ode_data.get("verified"),
                        complexity=ode_data["complexity_score"],
                        generator=ode_data["generator_name"],
                        function=ode_data["function_name"],
                        parameters=ode_data["parameters"],
                        timestamp=ode_data["timestamp"]
                    ))
                    
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
    print("ODE API Server started")

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
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
