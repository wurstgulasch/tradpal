"""
TradPal Backtesting Service Orchestrator
Unified orchestrator for backtesting, ML training, optimization, and walk-forward analysis
"""

import logging
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime
import pandas as pd

from .backtesting.service import BacktestingService
from .ml_training.service import MLTrainingService
from .optimization.service import OptimizationService
from .walk_forward.service import WalkForwardService

logger = logging.getLogger(__name__)


class BacktestingServiceOrchestrator:
    """Unified orchestrator for all backtesting-related services"""

    def __init__(self, event_system=None):
        self.event_system = event_system

        # Initialize service components
        self.backtesting_service = BacktestingService(event_system=event_system)
        self.ml_training_service = MLTrainingService(event_system=event_system)
        self.optimization_service = OptimizationService(event_system=event_system)
        self.walk_forward_service = WalkForwardService(
            event_system=event_system,
            backtesting_service=self.backtesting_service,
            optimization_service=self.optimization_service
        )

        self.is_initialized = False

    async def initialize(self):
        """Initialize all backtesting service components"""
        logger.info("Initializing Backtesting Service Orchestrator...")

        try:
            # Initialize all services concurrently
            init_tasks = [
                self.backtesting_service.initialize(),
                self.ml_training_service.initialize(),
                self.optimization_service.initialize(),
                self.walk_forward_service.initialize()
            ]

            await asyncio.gather(*init_tasks)

            self.is_initialized = True
            logger.info("Backtesting Service Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Backtesting Service Orchestrator: {e}")
            raise

    async def shutdown(self):
        """Shutdown all backtesting service components"""
        logger.info("Shutting down Backtesting Service Orchestrator...")

        try:
            # Shutdown all services concurrently
            shutdown_tasks = [
                self.backtesting_service.shutdown(),
                self.ml_training_service.shutdown(),
                self.optimization_service.shutdown(),
                self.walk_forward_service.shutdown()
            ]

            await asyncio.gather(*shutdown_tasks)

            self.is_initialized = False
            logger.info("Backtesting Service Orchestrator shut down successfully")

        except Exception as e:
            logger.error(f"Error during Backtesting Service Orchestrator shutdown: {e}")

    async def run_complete_backtesting_workflow(self, strategy_config: Dict[str, Any],
                                              data: pd.DataFrame,
                                              workflow_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run complete backtesting workflow including optimization and walk-forward analysis"""
        if not self.is_initialized:
            raise RuntimeError("Backtesting Service Orchestrator not initialized")

        workflow_config = workflow_config or {}
        enable_ml = workflow_config.get("enable_ml", False)
        enable_optimization = workflow_config.get("enable_optimization", True)
        enable_walk_forward = workflow_config.get("enable_walk_forward", True)

        logger.info(f"Starting complete backtesting workflow for {strategy_config.get('name', 'unknown')}")
        logger.info(f"ML: {enable_ml}, Optimization: {enable_optimization}, Walk-Forward: {enable_walk_forward}")

        workflow_results = {
            "strategy": strategy_config.get("name", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "phases": {},
            "final_recommendation": {},
            "success": False
        }

        try:
            # Phase 1: Initial backtest
            logger.info("Phase 1: Running initial backtest")
            initial_backtest = await self.backtesting_service.run_backtest(strategy_config, data)
            workflow_results["phases"]["initial_backtest"] = initial_backtest

            # Phase 2: ML training (if enabled)
            if enable_ml:
                logger.info("Phase 2: Training ML model")
                ml_training = await self.ml_training_service.train_model(
                    strategy_config, data, workflow_config.get("ml_config", {})
                )
                workflow_results["phases"]["ml_training"] = ml_training

            # Phase 3: Parameter optimization (if enabled)
            optimized_params = strategy_config
            if enable_optimization:
                logger.info("Phase 3: Optimizing strategy parameters")
                param_ranges = workflow_config.get("param_ranges", {})
                if param_ranges:
                    optimization = await self.optimization_service.optimize_strategy(
                        strategy_config.get("name", "unknown"), param_ranges, data
                    )
                    workflow_results["phases"]["optimization"] = optimization
                    optimized_params = {**strategy_config, **optimization["best_params"]}
                else:
                    logger.warning("No parameter ranges provided for optimization")

            # Phase 4: Walk-forward analysis (if enabled)
            if enable_walk_forward and enable_optimization:
                logger.info("Phase 4: Running walk-forward analysis")
                param_ranges = workflow_config.get("param_ranges", {})
                if param_ranges:
                    wf_config = workflow_config.get("walk_forward_config", {})
                    walk_forward = await self.walk_forward_service.run_walk_forward_analysis(
                        strategy_config.get("name", "unknown"), param_ranges, data, wf_config
                    )
                    workflow_results["phases"]["walk_forward"] = walk_forward

                    # Analyze stability
                    stability = await self.walk_forward_service.analyze_walk_forward_stability(walk_forward)
                    workflow_results["phases"]["stability_analysis"] = stability

                    # Generate report
                    report = await self.walk_forward_service.generate_walk_forward_report(
                        walk_forward, stability
                    )
                    workflow_results["phases"]["walk_forward_report"] = report
                else:
                    logger.warning("No parameter ranges provided for walk-forward analysis")

            # Generate final recommendation
            workflow_results["final_recommendation"] = self._generate_workflow_recommendation(workflow_results)
            workflow_results["success"] = True

            logger.info("Complete backtesting workflow finished successfully")
            return workflow_results

        except Exception as e:
            logger.error(f"Error in backtesting workflow: {e}")
            workflow_results["error"] = str(e)
            return workflow_results

    async def run_quick_backtest(self, strategy_config: Dict[str, Any],
                               data: pd.DataFrame) -> Dict[str, Any]:
        """Run quick backtest without optimization or ML"""
        if not self.is_initialized:
            raise RuntimeError("Backtesting Service Orchestrator not initialized")

        logger.info(f"Running quick backtest for {strategy_config.get('name', 'unknown')}")
        return await self.backtesting_service.run_backtest(strategy_config, data)

    async def optimize_strategy(self, strategy_name: str, param_ranges: Dict[str, List],
                              data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        if not self.is_initialized:
            raise RuntimeError("Backtesting Service Orchestrator not initialized")

        logger.info(f"Optimizing strategy: {strategy_name}")
        return await self.optimization_service.optimize_strategy(strategy_name, param_ranges, data)

    async def train_ml_model(self, strategy_config: Dict[str, Any],
                           data: pd.DataFrame, ml_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train ML model for strategy"""
        if not self.is_initialized:
            raise RuntimeError("Backtesting Service Orchestrator not initialized")

        logger.info(f"Training ML model for {strategy_config.get('name', 'unknown')}")
        return await self.ml_training_service.train_model(strategy_config, data, ml_config or {})

    async def run_walk_forward_analysis(self, strategy_name: str, param_ranges: Dict[str, List],
                                      data: pd.DataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        if not self.is_initialized:
            raise RuntimeError("Backtesting Service Orchestrator not initialized")

        logger.info(f"Running walk-forward analysis for {strategy_name}")
        return await self.walk_forward_service.run_walk_forward_analysis(strategy_name, param_ranges, data, config)

    def _generate_workflow_recommendation(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final recommendation based on workflow results"""
        phases = workflow_results.get("phases", {})

        recommendation = {
            "overall_rating": "unknown",
            "confidence_level": "low",
            "recommendations": [],
            "risk_assessment": "unknown"
        }

        # Analyze initial backtest
        if "initial_backtest" in phases:
            initial_metrics = phases["initial_backtest"].get("metrics", {})
            sharpe = initial_metrics.get("sharpe_ratio", 0)
            if sharpe > 1.0:
                recommendation["recommendations"].append("Initial backtest shows good risk-adjusted returns")
            elif sharpe < 0:
                recommendation["recommendations"].append("Initial backtest shows negative risk-adjusted returns")

        # Analyze optimization results
        if "optimization" in phases:
            opt_result = phases["optimization"]
            if opt_result.get("success", False):
                recommendation["recommendations"].append("Parameter optimization completed successfully")
            else:
                recommendation["recommendations"].append("Parameter optimization failed")

        # Analyze walk-forward stability
        if "stability_analysis" in phases:
            stability = phases["stability_analysis"]
            rating = stability.get("stability_rating", "unknown")
            recommendation["overall_rating"] = rating

            if rating == "excellent":
                recommendation["confidence_level"] = "high"
                recommendation["risk_assessment"] = "low"
                recommendation["recommendations"].append("Strategy shows excellent stability - ready for live trading")
            elif rating == "good":
                recommendation["confidence_level"] = "medium"
                recommendation["risk_assessment"] = "medium"
                recommendation["recommendations"].append("Strategy shows good stability with minor optimization needed")
            elif rating == "moderate":
                recommendation["confidence_level"] = "low"
                recommendation["risk_assessment"] = "high"
                recommendation["recommendations"].append("Strategy shows moderate stability - significant improvements recommended")
            else:
                recommendation["confidence_level"] = "low"
                recommendation["risk_assessment"] = "very_high"
                recommendation["recommendations"].append("Strategy shows poor stability - not recommended for live trading")

        # Analyze ML training
        if "ml_training" in phases:
            ml_result = phases["ml_training"]
            if ml_result.get("success", False):
                recommendation["recommendations"].append("ML model training completed successfully")
            else:
                recommendation["recommendations"].append("ML model training failed")

        return recommendation

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all orchestrated services"""
        return {
            "orchestrator_initialized": self.is_initialized,
            "backtesting_service": self.backtesting_service.is_initialized,
            "ml_training_service": self.ml_training_service.is_initialized,
            "optimization_service": self.optimization_service.is_initialized,
            "walk_forward_service": self.walk_forward_service.is_initialized,
            "timestamp": datetime.now().isoformat()
        }

    def get_default_workflow_config(self) -> Dict[str, Any]:
        """Get default workflow configuration"""
        return {
            "enable_ml": False,
            "enable_optimization": True,
            "enable_walk_forward": True,
            "ml_config": {
                "model_type": "random_forest",
                "train_split": 0.7,
                "validation_split": 0.2
            },
            "walk_forward_config": {
                "in_sample_window": 252,
                "out_sample_window": 21,
                "step_size": 21
            }
        }


# Simplified model classes for API compatibility
class BacktestingWorkflowRequest:
    """Backtesting workflow request model"""
    def __init__(self, strategy: Dict[str, Any], data: pd.DataFrame, config: Dict[str, Any] = None):
        self.strategy = strategy
        self.data = data
        self.config = config or {}

class BacktestingWorkflowResponse:
    """Backtesting workflow response model"""
    def __init__(self, success: bool, results: Dict[str, Any] = None, error: str = None):
        self.success = success
        self.results = results or {}
        self.error = error