from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
import json
import asyncio
from datetime import datetime
from pathlib import Path

from interface.api.models import (
    GenerationConfig, GenerationResponse, ODEResponse,
    StatusResponse, AnalysisResponse
)
from utils.config import ConfigManager
from pipeline.generator import ODEDatasetGenerator
from utils.features import FeatureExtractor

app = FastAPI(
    title="ODE Master Generators API",
    description="API for generating and analyzing ordinary differential equations",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active tasks
active_tasks = {}

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "ODE Master Generators API",
        "version": "2.0.0",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "generators": "/api/generators",
            "functions": "/api/functions",
            "generate": "/api/generate",
            "status": "/api/status/{task_id}",
            "results": "/api/results/{task_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/generators")
async def list_generators():
    """List available generators"""
    return {
        "linear": ["L1", "L2", "L3", "L4"],
        "nonlinear": ["N1", "N2", "N3"],
        "descriptions": {
            "L1": "y''(x) + y(x) = RHS",
            "L2": "y''(x) + y'(x) = RHS",
            "L3": "y(x) + y'(x) = RHS",
            "L4": "y''(x) + y(x/a) - y(x) = RHS (Pantograph)",
            "N1": "(y''(x))^q + y(x) = RHS",
            "N2": "(y''(x))^q + (y'(x))^v = RHS",
            "N3": "y(x) + (y'(x))^v = RHS"
        }
    }

@app.get("/api/functions")
async def list_functions():
    """List available analytic functions"""
    from core.functions import AnalyticFunctionLibrary
    
    functions = list(AnalyticFunctionLibrary.get_safe_library().keys())
    
    return {
        "functions": functions,
        "categories": {
            "polynomial": [f for f in functions if any(x in f for x in ["identity", "quadratic", "cubic", "quartic", "quintic"])],
            "exponential": [f for f in functions if "exp" in f],
            "trigonometric": [f for f in functions if any(x in f for x in ["sin", "cos", "tan"])],
            "hyperbolic": [f for f in functions if any(x in f for x in ["sinh", "cosh", "tanh"])],
            "logarithmic": [f for f in functions if "log" in f],
            "rational": [f for f in functions if "rational" in f],
            "composite": [f for f in functions if any(x in f for x in ["gaussian", "bessel", "erf", "sigmoid"])]
        }
    }

@app.post("/api/generate", response_model=GenerationResponse)
async def generate_odes(
    config: GenerationConfig,
    background_tasks: BackgroundTasks
):
    """Start ODE generation task"""
    task_id = str(uuid.uuid4())
    
    # Initialize task
    active_tasks[task_id] = {
        "id": task_id,
        "status": "pending",
        "start_time": datetime.now(),
        "config": config.dict(),
        "progress": 0,
        "total": 0,
        "generated": 0,
        "verified": 0,
        "errors": []
    }
    
    # Start generation in background
    background_tasks.add_task(
        run_generation_task,
        task_id,
        config
    )
    
    return GenerationResponse(
        task_id=task_id,
        status="started",
        message="Generation task started successfully"
    )

async def run_generation_task(task_id: str, config: GenerationConfig):
    """Run generation task asynchronously"""
    try:
        # Update status
        active_tasks[task_id]["status"] = "running"
        
        # Create config manager
        config_mgr = ConfigManager()
        
        # Override with API config
        if config.samples_per_combo:
            config_mgr.config['generation']['samples_per_combo'] = config.samples_per_combo
        
        if config.parameter_ranges:
            config_mgr.config['generation']['parameter_ranges'].update(
                config.parameter_ranges.dict()
            )
        
        # Create generator
        generator = ODEDatasetGenerator(config_mgr)
        
        # Custom progress tracking
        def update_progress(current, total):
            active_tasks[task_id].update({
                "progress": current,
                "total": total,
                "generated": len(generator.dataset),
                "verified": sum(1 for ode in generator.dataset if ode.verified)
            })
        
        # Monkey patch progress method
        original_log_progress = generator._log_progress
        generator._log_progress = lambda c, t: (original_log_progress(c, t), update_progress(c, t))
        
        # Filter generators if specified
        if config.generators:
            if config.generators.linear:
                generator.linear_generators = {
                    k: v for k, v in generator.linear_generators.items()
                    if k in config.generators.linear
                }
            if config.generators.nonlinear:
                generator.nonlinear_generators = {
                    k: v for k, v in generator.nonlinear_generators.items()
                    if k in config.generators.nonlinear
                }
        
        # Generate dataset
        dataset = generator.generate_dataset(config.samples_per_combo)
        
        # Save results
        result_file = Path(f"/tmp/ode_results_{task_id}.json")
        with open(result_file, 'w') as f:
            json.dump([ode.to_dict() for ode in dataset], f, default=str)
        
        # Extract features if requested
        if config.extract_features:
            extractor = FeatureExtractor()
            features_df = extractor.extract_features(dataset)
            features_file = Path(f"/tmp/ode_features_{task_id}.parquet")
            features_df.to_parquet(features_file)
            
            active_tasks[task_id]["features_file"] = str(features_file)
        
        # Update task
        active_tasks[task_id].update({
            "status": "completed",
            "end_time": datetime.now(),
            "result_file": str(result_file),
            "total_generated": len(dataset),
            "verification_rate": 100 * sum(1 for ode in dataset if ode.verified) / len(dataset)
        })
        
    except Exception as e:
        active_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now()
        })

@app.get("/api/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """Get generation task status"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    
    return StatusResponse(
        task_id=task_id,
        status=task["status"],
progress=task["progress"],
       total=task["total"],
       generated=task["generated"],
       verified=task["verified"],
       start_time=task["start_time"],
       end_time=task.get("end_time"),
       error=task.get("error")
   )

@app.get("/api/results/{task_id}")
async def get_results(
   task_id: str,
   offset: int = Query(0, ge=0),
   limit: int = Query(100, ge=1, le=1000)
):
   """Get generation results"""
   if task_id not in active_tasks:
       raise HTTPException(status_code=404, detail="Task not found")
   
   task = active_tasks[task_id]
   
   if task["status"] != "completed":
       raise HTTPException(
           status_code=400, 
           detail=f"Task is {task['status']}, not completed"
       )
   
   # Load results
   result_file = task.get("result_file")
   if not result_file or not Path(result_file).exists():
       raise HTTPException(status_code=404, detail="Results file not found")
   
   with open(result_file, 'r') as f:
       all_results = json.load(f)
   
   # Apply pagination
   paginated_results = all_results[offset:offset + limit]
   
   return {
       "task_id": task_id,
       "total": len(all_results),
       "offset": offset,
       "limit": limit,
       "results": paginated_results,
       "has_more": offset + limit < len(all_results)
   }

@app.get("/api/results/{task_id}/download/{format}")
async def download_results(task_id: str, format: str):
   """Download results in various formats"""
   if task_id not in active_tasks:
       raise HTTPException(status_code=404, detail="Task not found")
   
   task = active_tasks[task_id]
   
   if task["status"] != "completed":
       raise HTTPException(
           status_code=400, 
           detail=f"Task is {task['status']}, not completed"
       )
   
   if format == "json":
       result_file = task.get("result_file")
       if not result_file or not Path(result_file).exists():
           raise HTTPException(status_code=404, detail="Results file not found")
       
       return FileResponse(
           result_file,
           media_type="application/json",
           filename=f"ode_results_{task_id}.json"
       )
   
   elif format == "jsonl":
       # Convert to JSONL
       result_file = task.get("result_file")
       with open(result_file, 'r') as f:
           results = json.load(f)
       
       def generate():
           for result in results:
               yield json.dumps(result) + '\n'
       
       return StreamingResponse(
           generate(),
           media_type="application/x-ndjson",
           headers={
               "Content-Disposition": f"attachment; filename=ode_results_{task_id}.jsonl"
           }
       )
   
   elif format == "parquet":
       features_file = task.get("features_file")
       if not features_file or not Path(features_file).exists():
           raise HTTPException(
               status_code=404, 
               detail="Features file not found. Enable extract_features in generation."
           )
       
       return FileResponse(
           features_file,
           media_type="application/octet-stream",
           filename=f"ode_features_{task_id}.parquet"
       )
   
   else:
       raise HTTPException(
           status_code=400,
           detail=f"Invalid format: {format}. Use json, jsonl, or parquet."
       )

@app.get("/api/results/{task_id}/analysis", response_model=AnalysisResponse)
async def analyze_results(task_id: str):
   """Get analysis of generation results"""
   if task_id not in active_tasks:
       raise HTTPException(status_code=404, detail="Task not found")
   
   task = active_tasks[task_id]
   
   if task["status"] != "completed":
       raise HTTPException(
           status_code=400, 
           detail=f"Task is {task['status']}, not completed"
       )
   
   # Load results
   result_file = task.get("result_file")
   with open(result_file, 'r') as f:
       results = json.load(f)
   
   # Compute statistics
   from collections import Counter
   import numpy as np
   
   generator_counts = Counter(r['generator_name'] for r in results)
   function_counts = Counter(r['function_name'] for r in results)
   verification_methods = Counter(r['verification_method'] for r in results)
   
   complexities = [r['complexity_score'] for r in results]
   operation_counts = [r['operation_count'] for r in results]
   
   # Generator performance
   generator_performance = {}
   for gen in generator_counts:
       gen_results = [r for r in results if r['generator_name'] == gen]
       verified = sum(1 for r in gen_results if r['verified'])
       generator_performance[gen] = {
           'total': len(gen_results),
           'verified': verified,
           'verification_rate': 100 * verified / len(gen_results)
       }
   
   return AnalysisResponse(
       task_id=task_id,
       total_generated=len(results),
       total_verified=sum(1 for r in results if r['verified']),
       verification_rate=100 * sum(1 for r in results if r['verified']) / len(results),
       complexity_stats={
           'mean': np.mean(complexities),
           'std': np.std(complexities),
           'min': min(complexities),
           'max': max(complexities),
           'quartiles': np.percentile(complexities, [25, 50, 75]).tolist()
       },
       operation_count_stats={
           'mean': np.mean(operation_counts),
           'std': np.std(operation_counts),
           'min': min(operation_counts),
           'max': max(operation_counts)
       },
       generator_performance=generator_performance,
       function_distribution=dict(function_counts),
       verification_methods=dict(verification_methods),
       pantograph_count=sum(1 for r in results if r.get('has_pantograph')),
       linear_count=sum(1 for r in results if r['generator_type'] == 'linear'),
       nonlinear_count=sum(1 for r in results if r['generator_type'] == 'nonlinear')
   )

@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
   """Delete a task and its results"""
   if task_id not in active_tasks:
       raise HTTPException(status_code=404, detail="Task not found")
   
   task = active_tasks[task_id]
   
   # Delete result files
   if 'result_file' in task:
       result_file = Path(task['result_file'])
       if result_file.exists():
           result_file.unlink()
   
   if 'features_file' in task:
       features_file = Path(task['features_file'])
       if features_file.exists():
           features_file.unlink()
   
   # Remove from active tasks
   del active_tasks[task_id]
   
   return {"message": "Task deleted successfully"}

@app.get("/api/tasks")
async def list_tasks(
   status: Optional[str] = None,
   limit: int = Query(50, ge=1, le=100)
):
   """List all tasks"""
   tasks = list(active_tasks.values())
   
   # Filter by status if specified
   if status:
       tasks = [t for t in tasks if t['status'] == status]
   
   # Sort by start time (newest first)
   tasks.sort(key=lambda t: t['start_time'], reverse=True)
   
   # Apply limit
   tasks = tasks[:limit]
   
   # Return summary
   return {
       'total': len(active_tasks),
       'filtered': len(tasks),
       'tasks': [
           {
               'id': t['id'],
               'status': t['status'],
               'start_time': t['start_time'],
               'end_time': t.get('end_time'),
               'generated': t.get('generated', 0),
               'verified': t.get('verified', 0)
           }
           for t in tasks
       ]
   }

# Cleanup old tasks periodically
async def cleanup_old_tasks():
   """Remove completed tasks older than 24 hours"""
   while True:
       await asyncio.sleep(3600)  # Check every hour
       
       now = datetime.now()
       to_remove = []
       
       for task_id, task in active_tasks.items():
           if task['status'] in ['completed', 'failed']:
               if 'end_time' in task:
                   age = now - task['end_time']
                   if age.total_seconds() > 86400:  # 24 hours
                       to_remove.append(task_id)
       
       for task_id in to_remove:
           # Delete files
           task = active_tasks[task_id]
           if 'result_file' in task:
               try:
                   Path(task['result_file']).unlink()
               except:
                   pass
           if 'features_file' in task:
               try:
                   Path(task['features_file']).unlink()
               except:
                   pass
           
           del active_tasks[task_id]

@app.on_event("startup")
async def startup_event():
   """Start background tasks"""
   asyncio.create_task(cleanup_old_tasks())

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)