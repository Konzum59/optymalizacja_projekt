import random
import itertools
from RandomNumberGenerator import RandomNumberGenerator
import numpy as np
import math


def count_time(array, permutation):
    processing_times_array = array[permutation].copy()
    for row in range(1, number_of_items):
        processing_times_array[row][0] += processing_times_array[row - 1][0]
    for col in range(1, number_of_machines):
        processing_times_array[0][col] += processing_times_array[0][col - 1]
    for row in range(1, number_of_items):
        for col in range(1, number_of_machines):
            left = processing_times_array[row - 1][col]
            top = processing_times_array[row][col - 1]
            processing_times_array[row][col] += max(top, left)
    Cmax = processing_times_array.max()
    return Cmax


def ant_colony_alg(
    processing_times,
    n_ants,
    iterations,
    pheromone_efect,
    heuristic_efect,
    evaporation,
    pheromone_reinforcement,
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

                if chosen:
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

    number_of_items = 8
    number_of_machines = 8
    items = np.array(
        [
            [rnd.nextInt(1, 35) for _ in range(number_of_machines)]
            for _ in range(number_of_items)
        ]
    )

    solution = ant_colony_alg(items, 100, 40, 0.5, 1.0, 0.15, 20)
    print(solution[1])

    b = len(items)
    brutelist = list(range(b))
    bruteforce = [list(p) for p in itertools.permutations(brutelist)]
    brute_best = math.inf
    for permutation in bruteforce:
        cmax = count_time(items, permutation)
        if brute_best > cmax:
            brute_best = cmax
    print("brute best: ", brute_best)
    print("ant best: ", solution[1])
