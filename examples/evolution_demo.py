#!/usr/bin/env python3
"""
Demonstration: Evolution vs. Random Search
"""

import random

def fitness_function(ema_period):
    """Optimum bei EMA 50"""
    if ema_period < 5 or ema_period > 200: return 0
    return 100 - abs(ema_period - 50)

print('=== Warum Evolution besser ist als nur große Population ===')
print()
print('🎯 VERGLEICH: 1 Generation mit 1000 Configs vs. 10 Generationen mit 100 Configs')
print()

print('📊 METHODE 1: 1 Generation, 1000 zufällige Configs')
random_configs = [random.randint(5, 200) for _ in range(1000)]
random_fitnesses = [fitness_function(p) for p in random_configs]
best_random = max(random_fitnesses)
excellent_random = sum(1 for f in random_fitnesses if f >= 95)

print(f'  Beste Fitness: {best_random}')
print(f'  Exzellente Configs (≥95): {excellent_random}/1000 = {excellent_random/10:.1f}%')
print(f'  Rechenaufwand: 1000 Backtests')
print()

print('📊 METHODE 2: 10 Generationen, 100 Configs + Evolution')
population = [random.randint(5, 200) for _ in range(100)]
for gen in range(10):
    fitnesses = [fitness_function(p) for p in population]
    # Behalte Top 20% (Tournament Selection)
    sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
    elites = [population[i] for i in sorted_indices[:20]]

    # Erstelle neue Generation durch Crossover + Mutation
    new_population = elites.copy()  # Elitism
    while len(new_population) < 100:
        parent1, parent2 = random.sample(elites, 2)
        child = (parent1 + parent2) // 2
        # Mutation
        if random.random() < 0.2:  # 20% Mutationsrate
            child += random.randint(-5, 5)
            child = max(5, min(200, child))
        new_population.append(child)
    population = new_population

final_fitnesses = [fitness_function(p) for p in population]
best_evolved = max(final_fitnesses)
excellent_evolved = sum(1 for f in final_fitnesses if f >= 95)

print(f'  Beste Fitness: {best_evolved}')
print(f'  Exzellente Configs (≥95): {excellent_evolved}/100 = {excellent_evolved}%')
print(f'  Rechenaufwand: 10 × 100 = 1000 Backtests (gleicher Aufwand!)')
print()

print('🎉 EVOLUTION GEWINNT:')
print(f'  • {excellent_evolved}% vs {excellent_random/10:.1f}% exzellente Configs')
print(f'  • {best_evolved} vs {best_random} beste Fitness')
print('  • Systematische Verbesserung über Generationen')
print('  • Kombination bester Eigenschaften von Eltern')