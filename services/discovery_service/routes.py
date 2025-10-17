from fastapi import APIRouter, HTTPException
from .api import DiscoveryAPI
from .models import OptimizeBotRequest

router = APIRouter()
api = DiscoveryAPI()

@router.post("/optimize/bot", response_model=dict)
async def optimize_bot_configuration(request: OptimizeBotRequest):
    """Optimize complete bot configuration including indicators and trading parameters."""
    return await api.optimize_bot_configuration(request)

@router.get("/optimize/bot/{optimization_id}/status")
async def get_bot_optimization_status(optimization_id: str):
    """Get status of bot optimization process."""
    return await api.get_bot_optimization_status(optimization_id)

@router.get("/optimize/bot/{optimization_id}/results")
async def get_bot_optimization_results(optimization_id: str):
    """Get complete results of bot optimization."""
    return await api.get_bot_optimization_results(optimization_id)