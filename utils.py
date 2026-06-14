import numpy as np

NUMBER_OF_JOBS = 8
NUMBER_OF_MACHINES = 8


def count_time(
    processing_times: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    permutation: list[int],
):
    """
    Objective function

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


def build_dependencies(
    num_jobs: int, order_constraints: list[tuple[int, int]] | None
) -> dict[int, set[int]] | None:
    """
    Bulids dependencies as a result of order constraints.
    """
    if order_constraints is None:
        return None
    dependencies = {i: set() for i in range(num_jobs)}
    for before, after in order_constraints:
        dependencies[after].add(before)
    return dependencies


def get_available_jobs(
    unvisited: list[int], dependencies: dict[int, set[int]] | None
) -> list[int]:
    if dependencies is None:
        return unvisited
    unvisited_set = set(unvisited)
    return [job for job in unvisited if dependencies[job].isdisjoint(unvisited_set)]


def generate_valid_permutations(
    unvisited: list[int],
    dependencies: dict[int, set[int]] | None,
    current_perm: list[int] | None = None,
):
    """
    Used for generating search space.
    """
    if current_perm is None:
        current_perm = []
    if not unvisited:
        yield current_perm
        return

    for job in get_available_jobs(unvisited, dependencies):
        next_unvisited = [j for j in unvisited if j != job]
        yield from generate_valid_permutations(
            next_unvisited, dependencies, current_perm + [job]
        )
