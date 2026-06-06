from RandomNumberGenerator import RandomNumberGenerator
import numpy as np
import random
import time

from flow_shop import NUMBER_OF_JOBS, NUMBER_OF_MACHINES, ant_colony_alg
from flow_shop_extended import ant_colony_alg as ant_colony_alg_extended

if __name__ == "__main__":
    rnd = RandomNumberGenerator(735864)

    processing_times = np.array(
        [
            [rnd.nextInt(1, 35) for _ in range(NUMBER_OF_MACHINES)]
            for _ in range(NUMBER_OF_JOBS)
        ]
    )

    ITERATIONS = 100
    N_ANTS = 200
    PHEROMONE_EFFECT = 0.5
    HEURISTIC_EFFECT = 1.0
    EVAPORATION = 0.15
    PHEROMONE_REINFORCEMENT = 20
    # ORDER_CONSTRAINTS = [(1, 3), (1, 2)]

    NUM_RUNS = 15
    random.seed(42)
    SEEDS = [random.randint(0, 100000) for _ in range(NUM_RUNS)]

    ant_results = []
    ant_times = []
    ant_ext_results = []
    ant_ext_times = []

    print(f"Running algorithms {NUM_RUNS} times...\n")

    for i, seed in enumerate(SEEDS):
        # Standard ACO
        random.seed(seed)
        start_time = time.perf_counter()
        ant_solution = ant_colony_alg(
            processing_times,
            N_ANTS,
            ITERATIONS,
            PHEROMONE_EFFECT,
            HEURISTIC_EFFECT,
            EVAPORATION,
            PHEROMONE_REINFORCEMENT,
        )
        ant_times.append(time.perf_counter() - start_time)
        ant_results.append(ant_solution[1])

        # Extended ACO
        random.seed(seed)
        start_time = time.perf_counter()
        ant_ext_solution = ant_colony_alg_extended(
            processing_times,
            N_ANTS,
            ITERATIONS,
            PHEROMONE_EFFECT,
            HEURISTIC_EFFECT,
            EVAPORATION,
            PHEROMONE_REINFORCEMENT,
        )
        ant_ext_times.append(time.perf_counter() - start_time)
        ant_ext_results.append(ant_ext_solution[1])

        print(
            f"Run {i+1:02d}/{NUM_RUNS} (Seed {seed:05d}) -> "
            f"Standard: {ant_solution[1]} | Extended: {ant_ext_solution[1]}"
        )

    print("\n--- Final Results ---")
    print("Standard Ant Colony Algorithm:")
    print(f"  Min Cmax: {min(ant_results)}")
    print(f"  Max Cmax: {max(ant_results)}")
    print(f"  Avg Cmax: {sum(ant_results) / NUM_RUNS:.2f}")
    print(f"  Avg Time: {sum(ant_times) / NUM_RUNS:.2f}s\n")

    print("Extended Ant Colony Algorithm:")
    print(f"  Min Cmax: {min(ant_ext_results)}")
    print(f"  Max Cmax: {max(ant_ext_results)}")
    print(f"  Avg Cmax: {sum(ant_ext_results) / NUM_RUNS:.2f}")
    print(f"  Avg Time: {sum(ant_ext_times) / NUM_RUNS:.2f}s")
