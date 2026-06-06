import random
import itertools
from RandomNumberGenerator import RandomNumberGenerator
import numpy as np
import math
import time

from flow_shop import (
    count_time,
    NUMBER_OF_JOBS,
    NUMBER_OF_MACHINES,
    get_available_jobs,
    generate_valid_permutations,
    build_dependencies,
)


def calculate_population_diversity(solutions: list[tuple[list[int], int]], n: int):
    """
    The diversity is measured by comparing the edges (transitions between jobs)
    chosen by each ant. For every pair of ants, the distance is calculated as
    the number of non-shared edges. The average distance across all pairs is
    then normalized by the maximum possible number of edges (n - 1).

    Returns:
        A normalized diversity score as a float between 0.0 and 1.0.
        A value close to 0.0 indicates low diversity (stagnation/convergence),
        while a value close to 1.0 indicates high diversity (exploration).
    """

    n_ants = len(solutions)
    if n_ants <= 1:
        return 1.0

    edge_sets = []
    for perm, _ in solutions:
        edges = set((perm[k], perm[k + 1]) for k in range(n - 1))
        edge_sets.append(edges)

    total_distance = 0.0
    pairs_count = 0

    for i in range(n_ants):
        for j in range(i + 1, n_ants):
            shared_edges = len(edge_sets[i].intersection(edge_sets[j]))
            distance = (n - 1) - shared_edges
            total_distance += distance
            pairs_count += 1

    avg_distance = total_distance / pairs_count
    normalized_diversity = (avg_distance / (n - 1)) if n > 1 else 0.0

    return normalized_diversity


# Hyperparameters allow for balancing between exploration and exploitation
def ant_colony_alg(
    processing_times: np.typing.NDArray[np.int_],
    n_ants: int,
    iterations: int,
    pheromone_efect: float,
    heuristic_efect: float,
    evaporation: float,
    pheromone_reinforcement: int,
    order_constraints: list[tuple[int, int]] | None = None,
):
    # Ant colony algorithm extended by dynamic pheromones (feedback driven strategies rather than time based)
    n = len(processing_times)
    pheromones = [[1.0] * n for _ in range(n)]
    best_perm: list[int] | None = None
    best_cmax = math.inf
    stagnation_counter = 0

    dependencies = build_dependencies(n, order_constraints)

    for _ in range(iterations):
        if stagnation_counter > 30:  # hyperparameter
            # pheromone smoothing
            for i in range(n):
                for j in range(n):
                    pheromones[i][j] = (
                        1.0 + (pheromones[i][j] - 1.0) * 0.5
                    )  # 0.5 - hyperparameter

            if best_perm is not None:
                # Reinforce the global best so it survives the reset
                reinforcement_value = (
                    pheromone_reinforcement / best_cmax
                ) * 1.5  # Reduced from 5 to prevent immediate re-stagnation
                for i in range(n - 1):
                    a = best_perm[i]
                    b = best_perm[i + 1]
                    pheromones[a][b] += reinforcement_value
            stagnation_counter = 0

        solutions: list[tuple[list[int], int]] = []
        for _ in range(n_ants):
            perm: list[int] = []
            unvisited = list(range(n))
            start = random.choice(get_available_jobs(unvisited, dependencies))
            perm.append(start)
            unvisited.remove(start)

            while unvisited:
                current = perm[-1]
                probs: list[tuple[int, float]] = []
                jobs = get_available_jobs(unvisited, dependencies)
                for next_job in jobs:
                    heuristic = 1.0 / (
                        processing_times[next_job][0] + processing_times[next_job][-1]
                    )
                    pheromone_trail = pheromones[current][next_job]
                    prob = (pheromone_trail**pheromone_efect) * (
                        heuristic**heuristic_efect
                    )
                    probs.append((next_job, prob))

                total = sum(p for _, p in probs)
                probs = [(job, p / total) for job, p in probs]
                cummulative_value = 0
                chosen = None
                gambling = random.random()
                for job, p in probs:
                    cummulative_value += p
                    if gambling <= cummulative_value:
                        chosen = job
                        break

                if chosen is not None:
                    perm.append(chosen)
                    unvisited.remove(chosen)

            cmax = count_time(processing_times, perm)
            solutions.append((perm, cmax))

        current_diversity = calculate_population_diversity(solutions, n)
        actual_evaporation = evaporation * (
            1.5 - current_diversity
        )  # 1.5 - hyperparameter
        actual_evaporation = min(
            max(actual_evaporation, 0.02), 0.25
        )  # 0.2 and 0.25 - hyperparameters

        for i in range(n):
            for j in range(n):
                pheromones[i][j] *= 1 - actual_evaporation

        best_in_iter = min(solutions, key=lambda x: x[1])
        for i in range(n - 1):
            a = best_in_iter[0][i]
            b = best_in_iter[0][i + 1]
            pheromones[a][b] += pheromone_reinforcement / best_in_iter[1]

        # elitist strategy - reinforce the global best path after each iteration
        # it directs the search of all ants to construct a solution to contain links of the current best path
        if best_perm is not None:
            for i in range(n - 1):
                a = best_perm[i]
                b = best_perm[i + 1]
                pheromones[a][b] += pheromone_reinforcement / best_cmax

        if best_in_iter[1] < best_cmax:
            best_perm = best_in_iter[0].copy()
            best_cmax = best_in_iter[1]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

    return best_perm, best_cmax


if __name__ == "__main__":
    rnd = RandomNumberGenerator(735864)

    # rows (inner lists) - jobs; columns (inner list's items) - machines
    processing_times = np.array(
        [
            [rnd.nextInt(1, 35) for _ in range(NUMBER_OF_MACHINES)]
            for _ in range(NUMBER_OF_JOBS)
        ]
    )

    order_constraints = [(1, 3), (4, 6)]

    solution = ant_colony_alg(
        processing_times, 100, 40, 0.5, 1.0, 0.15, 20, order_constraints
    )

    b = len(processing_times)
    brutelist = list(range(b))
    deps = build_dependencies(b, order_constraints)
    bruteforce = (
        list(generate_valid_permutations(brutelist, deps))
        if deps is not None
        else [list(p) for p in itertools.permutations(brutelist)]
    )
    brute_best = math.inf
    # Search space is !NUMBER_OF_JOBS (when it's 8 it calculates solution in ~1.7s when bruteforcing)
    start_timestamp = time.perf_counter()
    for permutation in bruteforce:
        cmax = count_time(processing_times, permutation)
        if brute_best > cmax:
            brute_best = cmax
    end_timestamp = time.perf_counter()
    print(end_timestamp - start_timestamp)
    print("brute best: ", brute_best)
    print("ant best: ", solution[1])
