#!/usr/bin/env python3
"""
Quick Discovery Parameter Optimization

Fast parameter optimization for GA settings using statistical sampling.
"""

import sys
import os
import time
import json
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.discovery import DiscoveryOptimizer
from config.settings import SYMBOL, TIMEFRAME

def quick_parameter_test():
    """Quick test of different parameter combinations."""
    print("⚡ **Quick Discovery Parameter Test**")
    print("=" * 40)

    # Test configurations (carefully selected combinations)
    test_configs = [
        # Current defaults
        {'name': 'Current Defaults', 'population_size': 100, 'generations': 30, 'mutation_rate': 0.15, 'crossover_rate': 0.85},

        # Population size variations
        {'name': 'Small Population', 'population_size': 50, 'generations': 30, 'mutation_rate': 0.15, 'crossover_rate': 0.85},
        {'name': 'Large Population', 'population_size': 150, 'generations': 30, 'mutation_rate': 0.15, 'crossover_rate': 0.85},

        # Generation variations
        {'name': 'Few Generations', 'population_size': 100, 'generations': 15, 'mutation_rate': 0.15, 'crossover_rate': 0.85},
        {'name': 'Many Generations', 'population_size': 100, 'generations': 50, 'mutation_rate': 0.15, 'crossover_rate': 0.85},

        # Mutation variations
        {'name': 'Low Mutation', 'population_size': 100, 'generations': 30, 'mutation_rate': 0.05, 'crossover_rate': 0.85},
        {'name': 'High Mutation', 'population_size': 100, 'generations': 30, 'mutation_rate': 0.25, 'crossover_rate': 0.85},

        # Crossover variations
        {'name': 'Low Crossover', 'population_size': 100, 'generations': 30, 'mutation_rate': 0.15, 'crossover_rate': 0.70},
        {'name': 'High Crossover', 'population_size': 100, 'generations': 30, 'mutation_rate': 0.15, 'crossover_rate': 0.95},

        # Balanced combinations
        {'name': 'Conservative', 'population_size': 80, 'generations': 25, 'mutation_rate': 0.10, 'crossover_rate': 0.80},
        {'name': 'Aggressive', 'population_size': 120, 'generations': 35, 'mutation_rate': 0.20, 'crossover_rate': 0.90},
    ]

    results = []

    for i, config in enumerate(test_configs, 1):
        print(f"\n🧪 Testing {i}/{len(test_configs)}: {config['name']}")
        print(f"   Params: Pop={config['population_size']}, Gen={config['generations']}, Mut={config['mutation_rate']}, Cross={config['crossover_rate']}")

        start_time = time.time()

        try:
            # Create optimizer
            optimizer = DiscoveryOptimizer(
                population_size=config['population_size'],
                generations=config['generations'],
                mutation_rate=config['mutation_rate'],
                crossover_rate=config['crossover_rate']
            )

            # Run quick optimization
            results_list = optimizer.optimize(
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                max_evaluations=300,  # Quick test
                show_progress=False
            )

            runtime = time.time() - start_time

            # Extract metrics
            if results_list:
                fitness_scores = [r.fitness for r in results_list]
                best_fitness = max(fitness_scores)
                avg_fitness = np.mean(fitness_scores)
                std_fitness = np.std(fitness_scores)
                unique_configs = len(set(str(r.config) for r in results_list))
                diversity = unique_configs / len(results_list)
            else:
                best_fitness = avg_fitness = std_fitness = diversity = 0

            result = {
                'name': config['name'],
                'population_size': config['population_size'],
                'generations': config['generations'],
                'mutation_rate': config['mutation_rate'],
                'crossover_rate': config['crossover_rate'],
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': std_fitness,
                'diversity_score': diversity,
                'runtime_seconds': runtime,
                'success': True
            }

            print(f"✅ Success: Best Fitness = {best_fitness:.3f}, Runtime = {runtime:.1f}s")
        except Exception as e:
            runtime = time.time() - start_time
            result = {
                'name': config['name'],
                'population_size': config['population_size'],
                'generations': config['generations'],
                'mutation_rate': config['mutation_rate'],
                'crossover_rate': config['crossover_rate'],
                'best_fitness': 0,
                'avg_fitness': 0,
                'std_fitness': 0,
                'diversity_score': 0,
                'runtime_seconds': runtime,
                'success': False,
                'error': str(e)
            }
            print(f"❌ Failed: {e}")

        results.append(result)

    return pd.DataFrame(results)

def analyze_results(df: pd.DataFrame):
    """Analyze benchmark results and provide recommendations."""
    print("\n📊 **Analysis Results**")
    print("=" * 40)

    # Filter successful runs
    successful = df[df['success'] == True]

    if successful.empty:
        print("❌ No successful runs to analyze")
        return {}

    # Find best configuration
    best_idx = successful['best_fitness'].idxmax()
    best_config = successful.loc[best_idx]

    print("🏆 **Best Configuration:**")
    print(f"   Name: {best_config['name']}")
    print(f"   Population: {best_config['population_size']}")
    print(f"   Generations: {best_config['generations']}")
    print(f"   Mutation Rate: {best_config['mutation_rate']}")
    print(f"   Crossover Rate: {best_config['crossover_rate']}")
    print(f"   Best Fitness: {best_config['best_fitness']:.3f}")

    # Parameter analysis
    print("\n📈 **Parameter Impact Analysis:**")

    for param in ['population_size', 'generations', 'mutation_rate', 'crossover_rate']:
        if param in successful.columns:
            param_effect = successful.groupby(param)['best_fitness'].mean()
            best_val = param_effect.idxmax()
            best_score = param_effect.max()
            print(f"   {param}: Best = {best_val}, Avg Fitness = {best_score:.3f}")

    # Efficiency analysis
    print("\n⚡ **Efficiency Analysis:**")
    efficient = successful[successful['runtime_seconds'] < successful['runtime_seconds'].median()]
    if not efficient.empty:
        efficient_best = efficient.loc[efficient['best_fitness'].idxmax()]
        print(f"   Fast & Good: Fitness = {efficient_best['best_fitness']:.3f}, Time = {efficient_best['runtime_seconds']:.1f}s")

    # Recommendations
    recommendations = {
        'optimal_config': {
            'population_size': int(best_config['population_size']),
            'generations': int(best_config['generations']),
            'mutation_rate': float(best_config['mutation_rate']),
            'crossover_rate': float(best_config['crossover_rate'])
        },
        'efficiency_config': {
            'population_size': int(efficient_best['population_size']) if not efficient.empty else int(best_config['population_size']),
            'generations': int(efficient_best['generations']) if not efficient.empty else int(best_config['generations']),
            'mutation_rate': float(efficient_best['mutation_rate']) if not efficient.empty else float(best_config['mutation_rate']),
            'crossover_rate': float(efficient_best['crossover_rate']) if not efficient.empty else float(best_config['crossover_rate'])
        } if not efficient.empty else None
    }

    return recommendations

def main():
    """Main function."""
    # Run quick tests
    results_df = quick_parameter_test()

    # Analyze results
    recommendations = analyze_results(results_df)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"discovery_quick_test_{timestamp}.json"

    output = {
        'recommendations': recommendations,
        'results': results_df.to_dict('records'),
        'timestamp': timestamp
    }

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {results_file}")

    # Print final recommendations
    if recommendations:
        print("\n🎯 **Final Recommendations:**")
        opt = recommendations['optimal_config']
        print("**Optimal Configuration (Best Performance):**")
        print(f"   Population Size: {opt['population_size']}")
        print(f"   Generations: {opt['generations']}")
        print(f"   Mutation Rate: {opt['mutation_rate']}")
        print(f"   Crossover Rate: {opt['crossover_rate']}")

        if recommendations.get('efficiency_config'):
            eff = recommendations['efficiency_config']
            print("**Efficient Configuration (Fast & Good):**")
            print(f"   Population Size: {eff['population_size']}")
            print(f"   Generations: {eff['generations']}")
            print(f"   Mutation Rate: {eff['mutation_rate']}")
            print(f"   Crossover Rate: {eff['crossover_rate']}")

    print("\n✅ **Quick parameter test completed!**")

if __name__ == "__main__":
    main()