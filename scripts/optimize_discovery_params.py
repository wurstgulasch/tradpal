#!/usr/bin/env python3
"""
Discovery Parameter Optimization Benchmark

Systematically tests different GA parameter combinations to find optimal settings
for trading strategy discovery.
"""

import sys
import os
import time
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Simple progress bar fallback
    class tqdm:
        def __init__(self, total=None, desc=""):
            self.total = total
            self.desc = desc
            self.n = 0
        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"{self.desc}: {self.n}/{self.total}")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.discovery import DiscoveryOptimizer
from src.fitness import calculate_fitness_from_metrics
from config.settings import SYMBOL, TIMEFRAME

@dataclass
class BenchmarkResult:
    """Result of a parameter benchmark run."""
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    best_fitness: float
    avg_fitness: float
    std_fitness: float
    convergence_generation: int
    runtime_seconds: float
    unique_configs_found: int
    diversity_score: float

class DiscoveryParameterOptimizer:
    """Optimizes GA parameters for discovery mode."""

    def __init__(self):
        self.symbol = SYMBOL
        self.timeframe = TIMEFRAME
        self.benchmark_results = []

        # Parameter ranges to test (reduced for faster benchmarking)
        self.parameter_ranges = {
            'population_size': [50, 100, 150],      # 3 options
            'generations': [20, 30, 40],            # 3 options
            'mutation_rate': [0.10, 0.15, 0.20],   # 3 options
            'crossover_rate': [0.80, 0.85, 0.90]   # 3 options
        }

    def run_single_benchmark(self, params: Dict[str, Any]) -> BenchmarkResult:
        """Run a single benchmark with given parameters."""
        start_time = time.time()

        try:
            # Create optimizer with specific parameters
            optimizer = DiscoveryOptimizer(
                population_size=params['population_size'],
                generations=params['generations'],
                mutation_rate=params['mutation_rate'],
                crossover_rate=params['crossover_rate']
            )

            # Run optimization (with shorter backtest for speed)
            results = optimizer.optimize(
                symbol=self.symbol,
                timeframe=self.timeframe,
                max_evaluations=500,  # Reduced for benchmark speed
                show_progress=False
            )

            runtime = time.time() - start_time

            # Extract metrics
            fitness_scores = [r.fitness for r in results]
            best_fitness = max(fitness_scores) if fitness_scores else 0
            avg_fitness = np.mean(fitness_scores) if fitness_scores else 0
            std_fitness = np.std(fitness_scores) if fitness_scores else 0

            # Estimate convergence (when fitness improvement slows)
            convergence_gen = self._estimate_convergence(results)

            # Calculate diversity (unique configurations found)
            unique_configs = len(set(str(r.config) for r in results))
            diversity_score = unique_configs / len(results) if results else 0

            return BenchmarkResult(
                population_size=params['population_size'],
                generations=params['generations'],
                mutation_rate=params['mutation_rate'],
                crossover_rate=params['crossover_rate'],
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                std_fitness=std_fitness,
                convergence_generation=convergence_gen,
                runtime_seconds=runtime,
                unique_configs_found=unique_configs,
                diversity_score=diversity_score
            )

        except Exception as e:
            print(f"❌ Benchmark failed for params {params}: {e}")
            runtime = time.time() - start_time
            return BenchmarkResult(
                population_size=params['population_size'],
                generations=params['generations'],
                mutation_rate=params['mutation_rate'],
                crossover_rate=params['crossover_rate'],
                best_fitness=0,
                avg_fitness=0,
                std_fitness=0,
                convergence_generation=0,
                runtime_seconds=runtime,
                unique_configs_found=0,
                diversity_score=0
            )

    def _estimate_convergence(self, results: List) -> int:
        """Estimate at which generation convergence occurred."""
        if len(results) < 10:
            return 0

        # Group by generation (assuming results are ordered)
        generation_size = len(results) // max(1, getattr(results[0], 'generations', 10))
        fitness_by_gen = []

        for i in range(0, len(results), generation_size):
            gen_results = results[i:i+generation_size]
            gen_fitness = [r.fitness for r in gen_results]
            fitness_by_gen.append(max(gen_fitness) if gen_fitness else 0)

        # Find convergence (when improvement < 1% for 3 consecutive generations)
        for i in range(3, len(fitness_by_gen)):
            recent = fitness_by_gen[i-3:i+1]
            improvements = [(recent[j] - recent[j-1]) / max(abs(recent[j-1]), 1e-6)
                          for j in range(1, len(recent))]
            if all(imp < 0.01 for imp in improvements):  # Less than 1% improvement
                return i

        return len(fitness_by_gen)

    def run_full_benchmark(self, max_workers: int = 4) -> pd.DataFrame:
        """Run comprehensive benchmark of all parameter combinations."""
        print("🔬 **Discovery Parameter Optimization Benchmark**")
        print("=" * 60)

        # Generate all parameter combinations
        param_combinations = []
        for pop in self.parameter_ranges['population_size']:
            for gen in self.parameter_ranges['generations']:
                for mut in self.parameter_ranges['mutation_rate']:
                    for cross in self.parameter_ranges['crossover_rate']:
                        param_combinations.append({
                            'population_size': pop,
                            'generations': gen,
                            'mutation_rate': mut,
                            'crossover_rate': cross
                        })

        print(f"📊 Testing {len(param_combinations)} parameter combinations...")

        # Run benchmarks in parallel
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.run_single_benchmark, params)
                      for params in param_combinations]

            with tqdm(total=len(futures), desc="Benchmarking") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

        # Convert to DataFrame
        df = pd.DataFrame([{
            'population_size': r.population_size,
            'generations': r.generations,
            'mutation_rate': r.mutation_rate,
            'crossover_rate': r.crossover_rate,
            'best_fitness': r.best_fitness,
            'avg_fitness': r.avg_fitness,
            'std_fitness': r.std_fitness,
            'convergence_generation': r.convergence_generation,
            'runtime_seconds': r.runtime_seconds,
            'unique_configs_found': r.unique_configs_found,
            'diversity_score': r.diversity_score
        } for r in results])

        return df

    def find_optimal_parameters(self, benchmark_df: pd.DataFrame) -> Dict[str, Any]:
        """Find the optimal parameter combination based on benchmark results."""
        if benchmark_df.empty:
            return {}

        # Score each combination (weighted metrics)
        benchmark_df['composite_score'] = (
            benchmark_df['best_fitness'] * 0.4 +      # Best fitness (40%)
            benchmark_df['avg_fitness'] * 0.2 +       # Average fitness (20%)
            benchmark_df['diversity_score'] * 0.2 +   # Diversity (20%)
            (1 / (benchmark_df['runtime_seconds'] + 1)) * 0.2  # Efficiency (20%)
        )

        # Find best combination
        best_idx = benchmark_df['composite_score'].idxmax()
        best_params = benchmark_df.loc[best_idx]

        optimal_params = {
            'population_size': int(best_params['population_size']),
            'generations': int(best_params['generations']),
            'mutation_rate': float(best_params['mutation_rate']),
            'crossover_rate': float(best_params['crossover_rate']),
            'composite_score': float(best_params['composite_score']),
            'best_fitness': float(best_params['best_fitness']),
            'avg_fitness': float(best_params['avg_fitness']),
            'runtime_seconds': float(best_params['runtime_seconds'])
        }

        return optimal_params

    def generate_parameter_recommendations(self, benchmark_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed parameter recommendations."""
        optimal = self.find_optimal_parameters(benchmark_df)

        if not optimal:
            return {}

        # Analyze parameter effects
        param_analysis = {}

        for param in ['population_size', 'generations', 'mutation_rate', 'crossover_rate']:
            param_effect = benchmark_df.groupby(param)['best_fitness'].mean()
            param_analysis[param] = {
                'best_value': param_effect.idxmax(),
                'best_score': param_effect.max(),
                'effect_size': param_effect.max() - param_effect.min()
            }

        recommendations = {
            'optimal_parameters': optimal,
            'parameter_analysis': param_analysis,
            'benchmark_summary': {
                'total_combinations_tested': len(benchmark_df),
                'best_composite_score': optimal['composite_score'],
                'avg_runtime_seconds': benchmark_df['runtime_seconds'].mean(),
                'fitness_improvement_range': f"{benchmark_df['best_fitness'].min():.2f} - {benchmark_df['best_fitness'].max():.2f}"
            }
        }

        return recommendations

def main():
    """Main benchmark function."""
    optimizer = DiscoveryParameterOptimizer()

    # Run full benchmark
    benchmark_df = optimizer.run_full_benchmark(max_workers=4)

    # Generate recommendations
    recommendations = optimizer.generate_parameter_recommendations(benchmark_df)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"discovery_benchmark_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'recommendations': recommendations,
            'benchmark_data': benchmark_df.to_dict('records')
        }, f, indent=2, default=str)

    # Print results
    print("\n🎯 **Benchmark Results**")
    print("=" * 40)

    opt = recommendations['optimal_parameters']
    print(f"🏆 **Optimale Parameter:**")
    print(f"   Population Size: {opt['population_size']}")
    print(f"   Generations: {opt['generations']}")
    print(f"   Mutation Rate: {opt['mutation_rate']}")
    print(f"   Crossover Rate: {opt['crossover_rate']}")
    print(f"   Composite Score: {opt['composite_score']:.3f}")
    print(f"   Best Fitness: {opt['best_fitness']:.3f}")
    print(f"   Runtime: {opt['runtime_seconds']:.1f}s")

    print(f"\n💾 Results saved to: {results_file}")

    # Parameter analysis
    print(f"\n📊 **Parameter Analysis:**")
    for param, analysis in recommendations['parameter_analysis'].items():
        print(f"   {param}: Best = {analysis['best_value']}, Score = {analysis['best_score']:.3f}")

    print("\n✅ **Benchmark completed successfully!**")

if __name__ == "__main__":
    main()