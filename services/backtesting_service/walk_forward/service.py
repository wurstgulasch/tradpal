"""
TradPal Walk-Forward Service - Walk-Forward Analysis
Simplified implementation for unified service consolidation
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WalkForwardService:
    """Simplified walk-forward analysis service for core functionality"""

    def __init__(self, event_system=None, backtesting_service=None, optimization_service=None):
        self.event_system = event_system
        self.backtesting_service = backtesting_service
        self.optimization_service = optimization_service
        self.is_initialized = False

    async def initialize(self):
        """Initialize the walk-forward service"""
        logger.info("Initializing Walk-Forward Service...")
        # TODO: Initialize actual walk-forward components
        self.is_initialized = True
        logger.info("Walk-Forward Service initialized")

    async def shutdown(self):
        """Shutdown the walk-forward service"""
        logger.info("Walk-Forward Service shut down")
        self.is_initialized = False

    async def run_walk_forward_analysis(self, strategy_name: str, param_ranges: Dict[str, List],
                                      data: pd.DataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run complete walk-forward analysis"""
        if not self.is_initialized:
            raise RuntimeError("Walk-forward service not initialized")

        config = config or {}
        in_sample_window = config.get("in_sample_window", 252)  # ~1 year
        out_sample_window = config.get("out_sample_window", 21)  # ~1 month
        step_size = config.get("step_size", out_sample_window)

        logger.info(f"Running walk-forward analysis for {strategy_name}")
        logger.info(f"IS Window: {in_sample_window}, OOS Window: {out_sample_window}, Step: {step_size}")

        results = []
        data_length = len(data)

        for start_idx in range(0, data_length - in_sample_window - out_sample_window, step_size):
            is_end_idx = start_idx + in_sample_window
            oos_end_idx = min(is_end_idx + out_sample_window, data_length)

            is_data = data.iloc[start_idx:is_end_idx]
            oos_data = data.iloc[is_end_idx:oos_end_idx]

            # Optimize on in-sample data
            optimization_result = await self.optimization_service.optimize_strategy(
                strategy_name, param_ranges, is_data
            )

            # Test on out-of-sample data
            backtest_result = await self.backtesting_service.run_backtest(
                {"name": strategy_name, **optimization_result["best_params"]}, oos_data
            )

            window_result = {
                "window_number": len(results) + 1,
                "is_start": data.index[start_idx].isoformat(),
                "is_end": data.index[is_end_idx-1].isoformat(),
                "oos_start": data.index[is_end_idx].isoformat(),
                "oos_end": data.index[oos_end_idx-1].isoformat(),
                "optimized_params": optimization_result["best_params"],
                "is_score": optimization_result["best_score"],
                "oos_performance": backtest_result["metrics"],
                "oos_score": backtest_result["metrics"].get("sharpe_ratio", 0)
            }

            results.append(window_result)

        # Calculate overall statistics
        oos_scores = [r["oos_score"] for r in results]
        is_scores = [r["is_score"] for r in results]

        analysis = {
            "strategy": strategy_name,
            "total_windows": len(results),
            "in_sample_window": in_sample_window,
            "out_sample_window": out_sample_window,
            "step_size": step_size,
            "average_oos_score": float(np.mean(oos_scores)),
            "average_is_score": float(np.mean(is_scores)),
            "oos_score_std": float(np.std(oos_scores)),
            "best_oos_window": max(results, key=lambda x: x["oos_score"]),
            "worst_oos_window": min(results, key=lambda x: x["oos_score"]),
            "all_windows": results,
            "success": True
        }

        return analysis

    async def analyze_walk_forward_stability(self, wf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the stability of walk-forward optimization results"""
        if not self.is_initialized:
            raise RuntimeError("Walk-forward service not initialized")

        logger.info("Analyzing walk-forward stability")

        windows = wf_results["all_windows"]
        oos_scores = [w["oos_score"] for w in windows]

        # Calculate stability metrics
        score_mean = np.mean(oos_scores)
        score_std = np.std(oos_scores)
        score_cv = score_std / score_mean if score_mean != 0 else 0  # Coefficient of variation

        # Calculate consistency (percentage of windows with positive scores)
        positive_scores = sum(1 for s in oos_scores if s > 0)
        consistency = positive_scores / len(oos_scores)

        # Calculate degradation (IS vs OOS performance drop)
        is_scores = [w["is_score"] for w in windows]
        avg_degradation = np.mean(is_scores) - np.mean(oos_scores)

        return {
            "total_windows": len(windows),
            "mean_oos_score": float(score_mean),
            "std_oos_score": float(score_std),
            "coefficient_of_variation": float(score_cv),
            "consistency_ratio": float(consistency),
            "average_degradation": float(avg_degradation),
            "stability_rating": self._rate_stability(score_cv, consistency),
            "success": True
        }

    def _rate_stability(self, cv: float, consistency: float) -> str:
        """Rate the stability of the walk-forward results"""
        if cv < 0.5 and consistency > 0.7:
            return "excellent"
        elif cv < 0.8 and consistency > 0.5:
            return "good"
        elif cv < 1.2 and consistency > 0.3:
            return "moderate"
        else:
            return "poor"

    async def generate_walk_forward_report(self, wf_results: Dict[str, Any],
                                         stability_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive walk-forward analysis report"""
        if not self.is_initialized:
            raise RuntimeError("Walk-forward service not initialized")

        logger.info("Generating walk-forward analysis report")

        report = {
            "title": f"Walk-Forward Analysis Report - {wf_results['strategy']}",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "strategy": wf_results["strategy"],
                "total_windows": wf_results["total_windows"],
                "analysis_period": f"{wf_results['all_windows'][0]['is_start']} to {wf_results['all_windows'][-1]['oos_end']}",
                "average_oos_performance": wf_results["average_oos_score"],
                "stability_rating": stability_analysis["stability_rating"]
            },
            "performance_metrics": {
                "mean_oos_score": stability_analysis["mean_oos_score"],
                "std_oos_score": stability_analysis["std_oos_score"],
                "coefficient_of_variation": stability_analysis["coefficient_of_variation"],
                "consistency_ratio": stability_analysis["consistency_ratio"],
                "average_degradation": stability_analysis["average_degradation"]
            },
            "window_details": wf_results["all_windows"],
            "recommendations": self._generate_recommendations(stability_analysis),
            "success": True
        }

        return report

    def _generate_recommendations(self, stability_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on stability analysis"""
        recommendations = []

        rating = stability_analysis["stability_rating"]
        consistency = stability_analysis["consistency_ratio"]
        cv = stability_analysis["coefficient_of_variation"]

        if rating == "excellent":
            recommendations.append("Strategy shows excellent stability and consistency. Ready for live deployment.")
        elif rating == "good":
            recommendations.append("Strategy shows good stability. Consider minor parameter adjustments.")
        elif rating == "moderate":
            recommendations.append("Strategy shows moderate stability. Further optimization recommended.")
        else:
            recommendations.append("Strategy shows poor stability. Significant improvements needed.")

        if consistency < 0.5:
            recommendations.append("Low consistency ratio indicates strategy may be overfitting. Consider simpler approach.")

        if cv > 1.0:
            recommendations.append("High coefficient of variation suggests unstable performance. Review parameter ranges.")

        return recommendations

    def get_default_config(self) -> Dict[str, Any]:
        """Get default walk-forward configuration"""
        return {
            "in_sample_window": 252,  # ~1 year of daily data
            "out_sample_window": 21,  # ~1 month
            "step_size": 21,  # Monthly steps
            "min_windows": 12,  # Minimum 1 year of analysis
            "stability_threshold": 0.7  # Minimum consistency ratio
        }


# Simplified model classes for API compatibility
class WalkForwardAnalysisRequest:
    """Walk-forward analysis request model"""
    def __init__(self, strategy: str, param_ranges: Dict[str, List], config: Dict[str, Any] = None):
        self.strategy = strategy
        self.param_ranges = param_ranges
        self.config = config or {}

class WalkForwardAnalysisResponse:
    """Walk-forward analysis response model"""
    def __init__(self, success: bool, analysis: Dict[str, Any] = None, error: str = None):
        self.success = success
        self.analysis = analysis or {}
        self.error = error

class StabilityAnalysisRequest:
    """Stability analysis request model"""
    def __init__(self, walk_forward_results: Dict[str, Any]):
        self.walk_forward_results = walk_forward_results

class StabilityAnalysisResponse:
    """Stability analysis response model"""
    def __init__(self, success: bool, stability: Dict[str, Any] = None, error: str = None):
        self.success = success
        self.stability = stability or {}
        self.error = error