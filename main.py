import time
import random
import numpy as np
import json
import os

from RandomNumberGenerator import RandomNumberGenerator
from flow_shop import ant_colony_alg
from flow_shop_extended import ant_colony_alg as ant_colony_alg_extended
from brute_force_pfssp import brute_force_pfssp


def generate_processing_times(num_jobs, num_machines, seed=735864):
    rnd = RandomNumberGenerator(seed)
    return np.array(
        [[rnd.nextInt(1, 35) for _ in range(num_machines)] for _ in range(num_jobs)]
    )


def display_results(results_data):
    for r in results_data:
        print(f"\n{'='*60}\nEvaluating Scenario: {r['scenario']}\n{'='*60}")
        if r.get("opt_cmax"):
            print(f"-> Optimal Cmax found: {r['opt_cmax']}")

        print("\n--- Basic ACO Performance ---")
        print(
            f"  Best: {r['basic']['best']} | Worst: {r['basic']['worst']} | Avg: {r['basic']['avg']:.2f}"
        )
        print(f"  Avg Exec Time: {r['basic']['avg_time']:.2f}s")
        if "gap" in r["basic"]:
            print(f"  Avg Error Gap from Optimal: {r['basic']['gap']:.2f}%")

        print("\n--- Extended ACO Performance ---")
        print(
            f"  Best: {r['ext']['best']} | Worst: {r['ext']['worst']} | Avg: {r['ext']['avg']:.2f}"
        )
        print(f"  Avg Exec Time: {r['ext']['avg_time']:.2f}s")
        if "gap" in r["ext"]:
            print(f"  Avg Error Gap from Optimal: {r['ext']['gap']:.2f}%")
        elif "improvement" in r["ext"]:
            print(f"  Avg Improvement vs Basic ACO: {r['ext']['improvement']:.2f}%")


def run_experiments(force_rerun=False):
    RESULTS_FILE = "experiment_results.json"
    if not force_rerun and os.path.exists(RESULTS_FILE):
        print(f"Loading existing results from {RESULTS_FILE}...")
        with open(RESULTS_FILE, "r") as f:
            saved_results = json.load(f)
        display_results(saved_results)
        return

    # Shared Base hyperparameters
    # TIP: You can wrap the below inside another loop to test different Parameter combos
    params = {
        "iterations": 60,
        "n_ants": 100,
        "pheromone_effect": 0.5,
        "heuristic_effect": 1.0,
        "evaporation": 0.15,
        "pheromone_reinforcement": 20,
    }
    # order_constraints = [(1, 3), (2, 4)]

    # Define benchmark instances
    scenarios = [
        {"name": "Small Instance (8x8)", "jobs": 8, "machines": 8, "brute_force": True},
        {
            "name": "Medium Instance (20x10)",
            "jobs": 20,
            "machines": 10,
            "brute_force": False,
        },
        {
            "name": "Large Instance (40x20)",
            "jobs": 40,
            "machines": 20,
            "brute_force": False,
        },
    ]

    NUM_RUNS = 10
    seeds = [random.randint(0, 100000) for _ in range(NUM_RUNS)]

    experiment_data = []

    for scenario in scenarios:
        print(f"\n{'='*60}\nEvaluating Scenario: {scenario['name']}\n{'='*60}")
        pt = generate_processing_times(scenario["jobs"], scenario["machines"])

        opt_cmax = None
        if scenario["brute_force"]:
            print("Running Brute Force for optimal baseline...")
            opt_cmax = int(brute_force_pfssp(pt))
            print(f"-> Optimal Cmax found: {opt_cmax}")

        basic_results, ext_results = [], []
        basic_times, ext_times = [], []

        print(f"Running algorithms {NUM_RUNS} times with varying seeds...")
        for _, seed in enumerate(seeds):
            # Basic ACO
            random.seed(seed)
            start_time = time.perf_counter()
            basic_sol = ant_colony_alg(
                pt,
                params["n_ants"],
                params["iterations"],
                params["pheromone_effect"],
                params["heuristic_effect"],
                params["evaporation"],
                params["pheromone_reinforcement"],
            )
            basic_times.append(time.perf_counter() - start_time)
            basic_results.append(basic_sol[1])

            # Extended ACO
            random.seed(seed)
            start_time = time.perf_counter()
            ext_sol = ant_colony_alg_extended(
                pt,
                params["n_ants"],
                params["iterations"],
                params["pheromone_effect"],
                params["heuristic_effect"],
                params["evaporation"],
                params["pheromone_reinforcement"],
            )
            ext_times.append(time.perf_counter() - start_time)
            ext_results.append(ext_sol[1])

        scenario_dict = {
            "scenario": scenario["name"],
            "opt_cmax": opt_cmax,
            "basic": {
                "best": int(min(basic_results)),
                "worst": int(max(basic_results)),
                "avg": float(np.mean(basic_results)),
                "avg_time": float(np.mean(basic_times)),
            },
            "ext": {
                "best": int(min(ext_results)),
                "worst": int(max(ext_results)),
                "avg": float(np.mean(ext_results)),
                "avg_time": float(np.mean(ext_times)),
            },
        }

        # Aggregate and Display Metrics
        print("\n--- Basic ACO Performance ---")
        print(
            f"  Best: {min(basic_results)} | Worst: {max(basic_results)} | Avg: {np.mean(basic_results):.2f}"
        )
        print(f"  Avg Exec Time: {np.mean(basic_times):.2f}s")
        if opt_cmax:
            gap = (np.mean(basic_results) - opt_cmax) / opt_cmax * 100
            scenario_dict["basic"]["gap"] = float(gap)
            print(f"  Avg Error Gap from Optimal: {gap:.2f}%")

        print("\n--- Extended ACO Performance ---")
        print(
            f"  Best: {min(ext_results)} | Worst: {max(ext_results)} | Avg: {np.mean(ext_results):.2f}"
        )
        print(f"  Avg Exec Time: {np.mean(ext_times):.2f}s")
        if opt_cmax:
            gap = (np.mean(ext_results) - opt_cmax) / opt_cmax * 100
            scenario_dict["ext"]["gap"] = float(gap)
            print(f"  Avg Error Gap from Optimal: {gap:.2f}%")
        else:
            improvement = (
                (np.mean(basic_results) - np.mean(ext_results))
                / np.mean(basic_results)
                * 100
            )
            scenario_dict["ext"]["improvement"] = float(improvement)
            print(f"  Avg Improvement vs Basic ACO: {improvement:.2f}%")

        experiment_data.append(scenario_dict)

    with open(RESULTS_FILE, "w") as f:
        json.dump(experiment_data, f, indent=4)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiments()
