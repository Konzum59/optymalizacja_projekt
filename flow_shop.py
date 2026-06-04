import random
import itertools
from RandomNumberGenerator import RandomNumberGenerator
import numpy as np
import math

NUMBER_OF_JOBS = 8
NUMBER_OF_MACHINES = 8


def count_time(
    processing_times: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    permutation: list[int],
):
    """
    Args:
        processing_times: A 2D array representing processing times, where each row represents a single job
            and each column represents a successive machine. Specifically, `processing_times[i][j]` is the
            time required to process job `i` on machine `j`.
    """
    scheduled_times = processing_times[permutation].copy()
    for row in range(1, NUMBER_OF_JOBS):
        scheduled_times[row][0] += scheduled_times[row - 1][0]
    for col in range(1, NUMBER_OF_MACHINES):
        scheduled_times[0][col] += scheduled_times[0][col - 1]
    for row in range(1, NUMBER_OF_JOBS):
        for col in range(1, NUMBER_OF_MACHINES):
            left = scheduled_times[row - 1][col]
            top = scheduled_times[row][col - 1]
            scheduled_times[row][col] += max(top, left)
    Cmax = scheduled_times.max()
    return Cmax


def ant_colony_alg(
    processing_times: np.typing.NDArray[np.int_],
    n_ants: int,
    iterations: int,
    pheromone_efect: float,
    heuristic_efect: float,
    evaporation: float,
    pheromone_reinforcement: int,
):
    n = len(processing_times)
    pheromones = [[1.0] * n for _ in range(n)]
    best_perm = None
    best_cmax = math.inf
    for _ in range(iterations):
        solutions = []
        for _ in range(n_ants):
            perm: list[int] = []
            unvisited = list(range(n))
            start = random.choice(unvisited)
            perm.append(start)
            unvisited.remove(start)

            while unvisited:
                current = perm[-1]
                probs: list[tuple[int, float]] = []
                for next_job in unvisited:
                    heuristic = 1.0 / (
                        processing_times[next_job][0] + processing_times[next_job][-1]
                    )

                    prob = (pheromones[current][next_job] ** pheromone_efect) * (
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

    solution = ant_colony_alg(processing_times, 100, 40, 0.5, 1.0, 0.15, 20)
    print(solution[1])

    b = len(processing_times)
    brutelist = list(range(b))
    bruteforce = [list(p) for p in itertools.permutations(brutelist)]
    brute_best = math.inf
    # Search space is !NUMBER_OF_JOBS (when it's 8 it calculates solution in <1s)
    for permutation in bruteforce:
        cmax = count_time(processing_times, permutation)
        if brute_best > cmax:
            brute_best = cmax
    print("brute best: ", brute_best)
    print("ant best: ", solution[1])
