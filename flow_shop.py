import math
import random

import numpy as np

from RandomNumberGenerator import RandomNumberGenerator
from brute_force_pfssp import brute_force_pfssp
from utils import (
    build_dependencies,
    count_time,
    get_available_jobs,
    NUMBER_OF_JOBS,
    NUMBER_OF_MACHINES,
)


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
    n = len(processing_times)
    pheromones = [[1.0] * n for _ in range(n)]
    best_perm: list[int] | None = None
    best_cmax = math.inf

    # precompute dependencies
    dependencies = build_dependencies(n, order_constraints)

    for _ in range(iterations):
        solutions = []
        for _ in range(n_ants):
            # in the loop single ant constructs the path
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
                    # heuristic value favors jobs that have short processing times
                    # on the first machine and the final machine (similarily to Johnson's rule)
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
            if cmax < best_cmax:
                best_perm = perm[:]
                best_cmax = cmax

        for i in range(n):
            for j in range(n):
                pheromones[i][j] *= 1 - evaporation

        best_in_iter = min(solutions, key=lambda x: x[1])
        for i in range(n - 1):
            a = best_in_iter[0][i]
            b = best_in_iter[0][i + 1]
            pheromones[a][b] += pheromone_reinforcement / best_in_iter[1]
            # print(pheromone_reinforcement/best_in_iter[1])
            # print(unvisited)
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

    brute_best = brute_force_pfssp(processing_times, order_constraints)

    print("brute best: ", brute_best)
    print("ant best: ", solution[1])
