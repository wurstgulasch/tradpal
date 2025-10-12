#!/usr/bin/env python3
"""
Simple test for GA parameter generation with new indicators.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Test basic GA parameter generation
from deap import base, creator, tools
import random

def test_ga_setup():
    """Test GA parameter setup with new indicators."""
    print("Testing GA parameter setup with enhanced indicators...")

    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Define the toolbox
    toolbox = base.Toolbox()

    # Attribute generators for each parameter
    toolbox.register("ema_short", random.randint, 5, 50)
    toolbox.register("ema_long", random.randint, 10, 200)
    toolbox.register("rsi_period", random.randint, 5, 30)
    toolbox.register("rsi_oversold", random.randint, 20, 40)
    toolbox.register("rsi_overbought", random.randint, 60, 80)
    toolbox.register("bb_period", random.randint, 10, 50)
    toolbox.register("bb_std_dev", random.uniform, 1.5, 3.0)
    toolbox.register("atr_period", random.randint, 5, 30)
    toolbox.register("macd_fast", random.randint, 8, 20)
    toolbox.register("macd_slow", random.randint, 20, 40)
    toolbox.register("macd_signal", random.randint, 5, 15)
    toolbox.register("stoch_k", random.randint, 5, 21)
    toolbox.register("stoch_d", random.randint, 3, 8)

    # Boolean attributes
    toolbox.register("enable_rsi", random.choice, [True, False])
    toolbox.register("enable_bb", random.choice, [True, False])
    toolbox.register("enable_atr", random.choice, [True, False])
    toolbox.register("enable_adx", random.choice, [True, False])
    toolbox.register("enable_macd", random.choice, [True, False])
    toolbox.register("enable_obv", random.choice, [True, False])
    toolbox.register("enable_stoch", random.choice, [True, False])
    toolbox.register("enable_cmf", random.choice, [True, False])

    # Individual structure: 21 parameters
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.ema_short, toolbox.ema_long, toolbox.rsi_period,
                     toolbox.rsi_oversold, toolbox.rsi_overbought, toolbox.bb_period,
                     toolbox.bb_std_dev, toolbox.atr_period, toolbox.macd_fast,
                     toolbox.macd_slow, toolbox.macd_signal, toolbox.stoch_k,
                     toolbox.stoch_d, toolbox.enable_rsi, toolbox.enable_bb,
                     toolbox.enable_atr, toolbox.enable_adx, toolbox.enable_macd,
                     toolbox.enable_obv, toolbox.enable_stoch, toolbox.enable_cmf), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Generate a sample individual
    individual = toolbox.individual()
    print(f"Generated individual with {len(individual)} parameters:")
    print(f"  EMA periods: {individual[0]}, {individual[1]}")
    print(f"  RSI: period={individual[2]}, oversold={individual[3]}, overbought={individual[4]}, enabled={individual[13]}")
    print(f"  BB: period={individual[5]}, std_dev={individual[6]:.2f}, enabled={individual[14]}")
    print(f"  ATR: period={individual[7]}, enabled={individual[15]}")
    print(f"  MACD: fast={individual[8]}, slow={individual[9]}, signal={individual[10]}, enabled={individual[17]}")
    print(f"  Stochastic: k={individual[11]}, d={individual[12]}, enabled={individual[19]}")
    print(f"  OBV: enabled={individual[18]}")
    print(f"  CMF: enabled={individual[20]}")

    # Test config conversion
    config = {
        'ema': {'enabled': True, 'periods': [individual[0], individual[1]]},
        'rsi': {'enabled': individual[13], 'period': individual[2], 'oversold': individual[3], 'overbought': individual[4]},
        'bb': {'enabled': individual[14], 'period': individual[5], 'std_dev': individual[6]},
        'atr': {'enabled': individual[15], 'period': individual[7]},
        'adx': {'enabled': individual[16], 'period': 14},
        'macd': {'enabled': individual[17], 'fast_period': individual[8], 'slow_period': individual[9], 'signal_period': individual[10]},
        'obv': {'enabled': individual[18]},
        'stochastic': {'enabled': individual[19], 'k_period': individual[11], 'd_period': individual[12]},
        'cmf': {'enabled': individual[20]},
        'fibonacci': {'enabled': False}
    }

    print("\nConverted to configuration:")
    print(f"  MACD enabled: {config['macd']['enabled']}")
    print(f"  OBV enabled: {config['obv']['enabled']}")
    print(f"  Stochastic enabled: {config['stochastic']['enabled']}")
    print(f"  Chaikin Money Flow enabled: {config['cmf']['enabled']}")

    # Test population generation
    population = toolbox.population(n=5)
    print(f"\nGenerated population of {len(population)} individuals")
    for i, ind in enumerate(population):
        print(f"  Individual {i+1}: {len(ind)} parameters")

    print("\n✅ GA parameter setup test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_ga_setup()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)