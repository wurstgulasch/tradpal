"""
TradPal Backtesting Service Orchestrator
Unified orchestrator using consolidated BacktestingService
"""

import logging
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime
import pandas as pd

from services.trading_service.backtesting_service.service import BacktestingService

logger = logging.getLogger(__name__)


class BacktestingServiceOrchestrator:
    """Unified orchestrator using consolidated BacktestingService"""

    def __init__(self, event_system=None):
        self.event_system = event_system

        # Initialize consolidated service
        self.backtesting_service = BacktestingService(event_system=event_system)

        self.is_initialized = False

    async def initialize(self):
        """Initialize the consolidated backtesting service"""
        logger.info("Initializing Backtesting Service Orchestrator...")

        try:
            await self.backtesting_service.initialize()
            self.is_initialized = True
            logger.info("Backtesting Service Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Backtesting Service Orchestrator: {e}")
            raise

    async def shutdown(self):
        """Shutdown the consolidated backtesting service"""
        logger.info("Shutting down Backtesting Service Orchestrator...")

        try:
            await self.backtesting_service.shutdown()
            self.is_initialized = False
            logger.info("Backtesting Service Orchestrator shut down successfully")

        except Exception as e:
            logger.error(f"Error during Backtesting Service Orchestrator shutdown: {e}")

    async def run_complete_backtesting_workflow(self, strategy_config: Dict[str, Any],
                                              data: pd.DataFrame,
                                              workflow_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run complete backtesting workflow using consolidated service"""
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
                ml_config = workflow_config.get("ml_config", {})
                ml_training = await self.backtesting_service.train_model(
                    symbol=strategy_config.get("symbol", "BTC/USDT"),
                    timeframe=strategy_config.get("timeframe", "1d"),
                    start_date=str(data.index.min()),
                    end_date=str(data.index.max()),
                    model_type=ml_config.get("model_type", "random_forest"),
                    use_optuna=ml_config.get("optimize_hyperparams", False)
                )
                workflow_results["phases"]["ml_training"] = ml_training

            # Phase 3: Parameter optimization (if enabled)
            optimized_params = strategy_config
            if enable_optimization:
                logger.info("Phase 3: Optimizing strategy parameters")
                param_ranges = workflow_config.get("param_ranges", {})
                if param_ranges:
                    optimization = await self.backtesting_service.optimize_strategy(
                        strategy_config.get("name", "unknown"), param_ranges
                    )
                    workflow_results["phases"]["optimization"] = optimization
                    optimized_params = {**strategy_config, **optimization["best_params"]}
                else:
                    logger.warning("No parameter ranges provided for optimization")

            # Phase 4: Walk-forward analysis (if enabled)
            if enable_walk_forward:
                logger.info("Phase 4: Running walk-forward analysis")
                symbol = strategy_config.get("symbol", "BTC/USDT")
                timeframe = strategy_config.get("timeframe", "1d")
                start_date = str(data.index.min())
                end_date = str(data.index.max())

                walk_forward = await self.backtesting_service.run_walk_forward_optimization(
                    symbol, timeframe, start_date, end_date, "sharpe_ratio"
                )
                workflow_results["phases"]["walk_forward"] = walk_forward

                # Analyze stability from walk-forward results
                stability = self._analyze_walk_forward_stability(walk_forward)
                workflow_results["phases"]["stability_analysis"] = stability

                # Generate report
                report = self._generate_walk_forward_report(walk_forward, stability)
                workflow_results["phases"]["walk_forward_report"] = report

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
        return await self.backtesting_service.optimize_strategy(strategy_name, param_ranges)

    async def train_ml_model(self, strategy_config: Dict[str, Any],
                           data: pd.DataFrame, ml_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train ML model for strategy"""
        if not self.is_initialized:
            raise RuntimeError("Backtesting Service Orchestrator not initialized")

        logger.info(f"Training ML model for {strategy_config.get('name', 'unknown')}")

        ml_config = ml_config or {}
        model_name = f"{strategy_config.get('name', 'unknown')}_ml"
        symbol = strategy_config.get("symbol", "BTC/USDT")
        timeframe = strategy_config.get("timeframe", "1d")
        start_date = str(data.index.min())
        end_date = str(data.index.max())

        return await self.backtesting_service.train_model(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            model_type=ml_config.get("model_type", "random_forest"),
            use_optuna=ml_config.get("optimize_hyperparams", False)
        )

    async def run_walk_forward_analysis(self, strategy_name: str, param_ranges: Dict[str, List],
                                      data: pd.DataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        if not self.is_initialized:
            raise RuntimeError("Backtesting Service Orchestrator not initialized")

        logger.info(f"Running walk-forward analysis for {strategy_name}")

        # Extract parameters from data
        symbol = "BTC/USDT"  # Default, could be extracted from data or config
        timeframe = "1d"     # Default, could be extracted from data or config
        start_date = str(data.index.min())
        end_date = str(data.index.max())

        return await self.backtesting_service.run_walk_forward_optimization(
            symbol, timeframe, start_date, end_date, "sharpe_ratio"
        )

    def _analyze_walk_forward_stability(self, walk_forward_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze walk-forward stability from results"""
        analysis = walk_forward_results.get("analysis", {})

        stability_rating = "unknown"
        positive_ratio = analysis.get("positive_ratio", 0)
        consistency_score = analysis.get("consistency_score", 0)
        overfitting_ratio = analysis.get("overfitting_ratio", 1)

        if positive_ratio >= 0.7 and consistency_score >= 0.8 and overfitting_ratio <= 0.3:
            stability_rating = "excellent"
        elif positive_ratio >= 0.6 and consistency_score >= 0.6 and overfitting_ratio <= 0.5:
            stability_rating = "good"
        elif positive_ratio >= 0.5 and consistency_score >= 0.4:
            stability_rating = "moderate"
        else:
            stability_rating = "poor"

        return {
            "stability_rating": stability_rating,
            "positive_ratio": positive_ratio,
            "consistency_score": consistency_score,
            "overfitting_ratio": overfitting_ratio,
            "information_coefficient": analysis.get("information_coefficient", 0),
            "performance_decay": analysis.get("performance_decay", 0)
        }

    def _generate_walk_forward_report(self, walk_forward_results: Dict[str, Any],
                                    stability_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate walk-forward analysis report"""
        analysis = walk_forward_results.get("analysis", {})

        report = {
            "summary": {
                "total_windows": walk_forward_results.get("total_windows", 0),
                "average_oos_performance": analysis.get("average_oos_performance", 0),
                "stability_rating": stability_analysis.get("stability_rating", "unknown")
            },
            "key_metrics": {
                "positive_ratio": stability_analysis.get("positive_ratio", 0),
                "consistency_score": stability_analysis.get("consistency_score", 0),
                "overfitting_ratio": stability_analysis.get("overfitting_ratio", 1),
                "information_coefficient": stability_analysis.get("information_coefficient", 0)
            },
            "recommendations": []
        }

        # Generate recommendations based on stability
        rating = stability_analysis.get("stability_rating", "unknown")
        if rating == "excellent":
            report["recommendations"].append("Strategy shows excellent stability - suitable for live trading")
            report["recommendations"].append("Consider implementing with confidence")
        elif rating == "good":
            report["recommendations"].append("Strategy shows good stability with minor concerns")
            report["recommendations"].append("Monitor performance closely in live trading")
        elif rating == "moderate":
            report["recommendations"].append("Strategy shows moderate stability")
            report["recommendations"].append("Significant improvements recommended before live trading")
        else:
            report["recommendations"].append("Strategy shows poor stability")
            report["recommendations"].append("Not recommended for live trading without major revisions")

        return report

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
        """Get status of the consolidated backtesting service"""
        return {
            "orchestrator_initialized": self.is_initialized,
            "backtesting_service": self.backtesting_service.is_initialized,
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