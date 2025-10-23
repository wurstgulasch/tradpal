from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class OptimizeBotRequest(BaseModel):
    """Request model for bot configuration optimization."""
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    population_size: Optional[int] = 100
    generations: Optional[int] = 30
    use_walk_forward: Optional[bool] = True

class OptimizeBotResponse(BaseModel):
    """Response model for bot configuration optimization."""
    success: bool
    optimization_id: str
    best_fitness: float
    best_config: Dict[str, Any]
    total_evaluations: int
    duration_seconds: float
    top_configurations: List[Dict[str, Any]]
    message: str