import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import heapq
import time
import tracemalloc
import numpy as np
from grid import Grid


class Dijkstra:
    """Dijkstra's shortest path algorithm on a 2D grid."""

    def __init__(self, grid, connectivity=4):
        """
        Args:
            grid: Grid object.
            connectivity: 4 or 8 neighbor connectivity.
        """
        self.grid = grid
        self.connectivity = connectivity

    def search(self, start, goal):
        """
        Find shortest path from start to goal.

        Args:
            start: (row, col) tuple.
            goal: (row, col) tuple.

        Returns:
            dict with keys:
                path: list of (row, col) from start to goal, or None if no path.
                cost: total path cost.
                nodes_explored: number of cells expanded.
                planning_time_ms: time taken in milliseconds.
                memory_mb: peak memory usage in MB.
        """
        # Start tracking memory
        tracemalloc.start()
        start_time = time.perf_counter()

        # Priority queue: (cost, row, col)
        open_list = []
        heapq.heappush(open_list, (0.0, start[0], start[1]))

        # Cost to reach each cell
        g_cost = {start: 0.0}

        # Parent tracking for path reconstruction
        parent = {start: None}

        # Track explored nodes
        explored = set()
        nodes_explored = 0

        path = None
        total_cost = float('inf')

        while open_list:
            current_cost, row, col = heapq.heappop(open_list)
            current = (row, col)

            # Skip if already explored (duplicate in queue)
            if current in explored:
                continue

            explored.add(current)
            nodes_explored += 1

            # Goal reached
            if current == goal:
                path = self._reconstruct_path(parent, goal)
                total_cost = current_cost
                break

            # Expand neighbors
            for nr, nc, move_cost in self.grid.get_neighbors(row, col, self.connectivity):
                neighbor = (nr, nc)
                new_cost = current_cost + move_cost

                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    parent[neighbor] = current
                    heapq.heappush(open_list, (new_cost, nr, nc))

        # Stop tracking
        end_time = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "path": path,
            "cost": total_cost if path else None,
            "nodes_explored": nodes_explored,
            "planning_time_ms": (end_time - start_time) * 1000,
            "memory_mb": peak_memory / (1024 * 1024),
            "explored_cells": explored
        }

    def _reconstruct_path(self, parent, goal):
        """Walk back from goal to start using parent dict."""
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path


if __name__ == "__main__":
    # Test on a small grid
    g = Grid(20, 20, obstacle_density=0.2, seed=42)
    start = (0, 0)
    goal = (19, 19)

    dijkstra = Dijkstra(g, connectivity=4)
    result = dijkstra.search(start, goal)

    if result["path"]:
        print(f"Path found! Length: {len(result['path'])} cells")
        print(f"Cost: {result['cost']:.2f}")
        print(f"Nodes explored: {result['nodes_explored']}")
        print(f"Time: {result['planning_time_ms']:.2f} ms")
        print(f"Memory: {result['memory_mb']:.4f} MB")
    else:
        print("No path found.")
