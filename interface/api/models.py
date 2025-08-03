from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class GeneratorType(str, Enum):
    LINEAR = "linear"
    NONLINEAR = "nonlinear"

class ParameterRanges(BaseModel):
    alpha: Optional[List[float]] = Field(default=[0, 0.5, 1, 1.5, 2])
    beta: Optional[List[float]] = Field(default=[0.5, 1, 1.5, 2])
    M: Optional[List[float]] = Field(default=[0, 0.5, 1])
    q: Optional[List[int]] = Field(default=[2, 3])
    v: Optional[List[int]] = Field(default=[2, 3, 4])
    a: Optional[List[int]] = Field(default=[2, 3, 4])

class GeneratorSelection(BaseModel):
    linear: Optional[List[str]] = Field(default=["L1", "L2", "L3", "L4"])
    nonlinear: Optional[List[str]] = Field(default=["N1", "N2", "N3"])

class GenerationConfig(BaseModel):
    samples_per_combo: int = Field(default=5, ge=1, le=100)
    generators: Optional[GeneratorSelection] = None
    functions: Optional[List[str]] = None
    parameter_ranges: Optional[ParameterRanges] = None
    extract_features: bool = Field(default=False)
    parallel: bool = Field(default=False)
    n_workers: Optional[int] = None

class GenerationResponse(BaseModel):
    task_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    total: int
    generated: int
    verified: int
    start_time: datetime
    end_time: Optional[datetime] = None
    error: Optional[str] = None

class ODEResponse(BaseModel):
    id: int
    generator_type: str
    generator_name: str
    function_name: str
    ode_symbolic: str
    ode_latex: str
    solution_symbolic: str
    solution_latex: str
    initial_conditions: Dict[str, str]
    parameters: Dict[str, float]
    complexity_score: int
    verified: bool
    verification_method: str
    has_pantograph: bool

class AnalysisResponse(BaseModel):
    task_id: str
    total_generated: int
    total_verified: int
    verification_rate: float
    complexity_stats: Dict[str, float]
    operation_count_stats: Dict[str, float]
    generator_performance: Dict[str, Dict[str, Any]]
    function_distribution: Dict[str, int]
    verification_methods: Dict[str, int]
    pantograph_count: int
    linear_count: int
    nonlinear_count: int