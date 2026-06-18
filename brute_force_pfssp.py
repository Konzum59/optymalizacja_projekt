import math
import time

import numpy as np

from utils import build_dependencies, count_time, generate_valid_permutations


def brute_force_pfssp(
    processing_times: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    order_constraints: list[tuple[int, int]] | None = None,
):
    b = len(processing_times)
    brutelist = list(range(b))
    deps = build_dependencies(b, order_constraints)
    bruteforce = list(generate_valid_permutations(brutelist, deps))

    brute_best = math.inf
    start_timestamp = time.perf_counter()
    for permutation in bruteforce:
        cmax = count_time(processing_times, permutation)
        if brute_best > cmax:
            brute_best = cmax

    end_timestamp = time.perf_counter()
    print(end_timestamp - start_timestamp)
    return brute_best
