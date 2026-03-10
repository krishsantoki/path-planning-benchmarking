import numpy as np
import csv
import time
import os
from grid import Grid
from dijkstra import Dijkstra
from astar import AStar
from rrt_connect_kd import RRTConnectKD
from rrt_star_kd import RRTStarKD


class BenchmarkRunner:
    """Run all algorithms across multiple trials and obstacle densities."""

    def __init__(self, grid_size=100, trials=200,
                 densities=[0.1, 0.2, 0.3], connectivity=4):
        """
        Args:
            grid_size: Width and height of square grid.
            trials: Number of trials per algorithm per density.
            densities: List of obstacle density levels.
            connectivity: 4 or 8 for grid-based algorithms.
        """
        self.grid_size = grid_size
        self.trials = trials
        self.densities = densities
        self.connectivity = connectivity

        self.algorithms = {
            "Dijkstra": self._run_dijkstra,
            "A*": self._run_astar,
            "RRT-Connect": self._run_rrt_connect,
            "RRT*": self._run_rrt_star,
        }

        self.results = []

    def _run_dijkstra(self, grid, start, goal):
        algo = Dijkstra(grid, self.connectivity)
        return algo.search(start, goal)

    def _run_astar(self, grid, start, goal):
        algo = AStar(grid, self.connectivity, heuristic="euclidean")
        return algo.search(start, goal)

    def _run_rrt_connect(self, grid, start, goal):
        algo = RRTConnectKD(grid, step_size=5, max_iterations=5000, connect_threshold=3.0)
        return algo.search(start, goal)

    def _run_rrt_star(self, grid, start, goal):
        algo = RRTStarKD(grid, step_size=8, max_iterations=800,
                         goal_bias=0.2, goal_threshold=12.0, rewire_radius=15.0)
        return algo.search(start, goal)

    def run(self):
        """Run full benchmark suite."""
        total_runs = len(self.algorithms) * len(self.densities) * self.trials
        run_count = 0

        for density in self.densities:
            print(f"\n{'='*60}")
            print(f"Obstacle Density: {density*100:.0f}%")
            print(f"{'='*60}")

            for algo_name, algo_func in self.algorithms.items():
                print(f"\n  Running {algo_name}...")
                successes = 0
                failures = 0

                for trial in range(self.trials):
                    run_count += 1
                    seed = trial  # Same grid for all algorithms at same trial

                    grid = Grid(self.grid_size, self.grid_size,
                                obstacle_density=density, seed=seed)
                    start = (0, 0)
                    goal = (self.grid_size - 1, self.grid_size - 1)

                    try:
                        result = algo_func(grid, start, goal)

                        if result["path"] is not None:
                            successes += 1
                            self.results.append({
                                "algorithm": algo_name,
                                "density": density,
                                "trial": trial,
                                "success": True,
                                "planning_time_ms": result["planning_time_ms"],
                                "path_length": len(result["path"]),
                                "path_cost": result["cost"],
                                "nodes_explored": result["nodes_explored"],
                                "memory_mb": result["memory_mb"],
                            })
                        else:
                            failures += 1
                            self.results.append({
                                "algorithm": algo_name,
                                "density": density,
                                "trial": trial,
                                "success": False,
                                "planning_time_ms": result["planning_time_ms"],
                                "path_length": None,
                                "path_cost": None,
                                "nodes_explored": result["nodes_explored"],
                                "memory_mb": result["memory_mb"],
                            })
                    except Exception as e:
                        failures += 1
                        self.results.append({
                            "algorithm": algo_name,
                            "density": density,
                            "trial": trial,
                            "success": False,
                            "planning_time_ms": None,
                            "path_length": None,
                            "path_cost": None,
                            "nodes_explored": None,
                            "memory_mb": None,
                        })

                    # Progress update every 50 trials
                    if (trial + 1) % 50 == 0:
                        print(f"    Trial {trial + 1}/{self.trials} "
                              f"({run_count}/{total_runs} total)")

                success_rate = successes / self.trials * 100
                print(f"  {algo_name} done: {success_rate:.1f}% success rate")

        print(f"\n{'='*60}")
        print(f"Benchmark complete. {len(self.results)} results recorded.")

    def save_csv(self, filepath="results/benchmark_results.csv"):
        """Save all results to CSV."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "algorithm", "density", "trial", "success",
                "planning_time_ms", "path_length", "path_cost",
                "nodes_explored", "memory_mb"
            ])
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Results saved to {filepath}")

    def print_summary(self):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print(f"{'BENCHMARK SUMMARY':^80}")
        print(f"{'='*80}")
        print(f"Grid: {self.grid_size}x{self.grid_size} | "
              f"Trials: {self.trials} | "
              f"Connectivity: {self.connectivity}")
        print(f"{'='*80}")

        for density in self.densities:
            print(f"\n--- Obstacle Density: {density*100:.0f}% ---")
            print(f"{'Algorithm':<15} {'Success%':>8} {'Time(ms)':>10} "
                  f"{'PathCost':>10} {'Nodes':>8} {'Memory(MB)':>10}")
            print("-" * 65)

            for algo_name in self.algorithms:
                algo_results = [r for r in self.results
                                if r["algorithm"] == algo_name
                                and r["density"] == density]

                success_results = [r for r in algo_results if r["success"]]
                success_rate = len(success_results) / len(algo_results) * 100

                if success_results:
                    avg_time = np.mean([r["planning_time_ms"] for r in success_results])
                    avg_cost = np.mean([r["path_cost"] for r in success_results])
                    avg_nodes = np.mean([r["nodes_explored"] for r in success_results])
                    avg_memory = np.mean([r["memory_mb"] for r in success_results])

                    print(f"{algo_name:<15} {success_rate:>7.1f}% {avg_time:>10.2f} "
                          f"{avg_cost:>10.2f} {avg_nodes:>8.0f} {avg_memory:>10.4f}")
                else:
                    print(f"{algo_name:<15} {success_rate:>7.1f}%     ---        "
                          f"---      ---        ---")


if __name__ == "__main__":
    print("Starting benchmark...")
    print("This will take a while. Go grab coffee.\n")

    runner = BenchmarkRunner(
        grid_size=50,
        trials=50,
        densities=[0.1, 0.2, 0.3],
        connectivity=4
    )

    runner.run()
    runner.save_csv()
    runner.print_summary()
