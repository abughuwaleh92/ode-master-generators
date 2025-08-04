import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uuid
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with correct name
from pipeline.generator import ODEDatasetGenerator as ODEGenerator
from verification.verifier import ODEVerifier

# Initialize FastAPI app
app = FastAPI(
    title="ODE Generation API",
    description="Simple API for ODE generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your GUI URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment
VALID_API_KEY = os.getenv('API_KEY', 'test-key-123')

# Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """Verify API key"""
    if api_key != VALID_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Request/Response models
class ODEGenerationRequest(BaseModel):
    generator: str = Field(..., description="Generator name (e.g., L1, N1)")
    function: str = Field(..., description="Function name (e.g., sine, exponential)")
    parameters: Optional[Dict[str, float]] = Field(default=None, description="ODE parameters")
    count: int = Field(1, ge=1, le=10, description="Number of ODEs to generate")
    verify: bool = Field(True, description="Whether to verify generated ODEs")

# Initialize generators
generator = ODEGenerator()
verifier = ODEVerifier()

# API Endpoints
@app.post("/api/v1/generate")
async def generate_odes(
    request: ODEGenerationRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate ODEs synchronously (simplified version)"""
    
    try:
        results = []
        
        for i in range(request.count):
            # Generate ODE
            ode_data = generator.generate_single(
                request.generator,
                request.function,
                request.parameters
            )
            
            if ode_data:
                # Simple response format
                result = {
                    "id": str(uuid.uuid4()),
                    "ode": ode_data.get("ode_symbolic", ""),
                    "solution": ode_data.get("solution_symbolic", ""),
                    "complexity": ode_data.get("complexity_score", 0),
                    "generator": request.generator,
                    "function": request.function,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Verify if requested
                if request.verify and result["solution"]:
                    try:
                        verification = verifier.verify(result["ode"], result["solution"])
                        result["verified"] = verification.get("verified", False)
                    except:
                        result["verified"] = False
                
                results.append(result)
        
        return {
            "status": "success",
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stats")
async def get_statistics(api_key: str = Depends(verify_api_key)):
    """Get simple statistics"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ODE Generation API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
