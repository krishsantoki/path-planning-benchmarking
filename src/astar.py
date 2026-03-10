import heapq
import time
import tracemalloc
import numpy as np
from grid import Grid


class AStar:
    """A* search algorithm on a 2D grid."""

    def __init__(self, grid, connectivity=4, heuristic="euclidean"):
        """
        Args:
            grid: Grid object.
            connectivity: 4 or 8 neighbor connectivity.
            heuristic: 'euclidean' or 'manhattan'.
        """
        self.grid = grid
        self.connectivity = connectivity
        self.heuristic = heuristic

    def _h(self, node, goal):
        """Compute heuristic estimate from node to goal."""
        dr = abs(node[0] - goal[0])
        dc = abs(node[1] - goal[1])

        if self.heuristic == "manhattan":
            return dr + dc
        else:  # euclidean
            return np.sqrt(dr**2 + dc**2)

    def search(self, start, goal):
        """
        Find shortest path from start to goal using A*.

        Args:
            start: (row, col) tuple.
            goal: (row, col) tuple.

        Returns:
            dict with: path, cost, nodes_explored, planning_time_ms, memory_mb, explored_cells.
        """
        tracemalloc.start()
        start_time = time.perf_counter()

        # Priority queue: (f_cost, row, col)
        # f = g + h  (actual cost + heuristic estimate)
        open_list = []
        heapq.heappush(open_list, (self._h(start, goal), 0.0, start[0], start[1]))
        # Storing (f_cost, g_cost, row, col) — g_cost as tiebreaker

        g_cost = {start: 0.0}
        parent = {start: None}
        explored = set()
        nodes_explored = 0

        path = None
        total_cost = float('inf')

        while open_list:
            f, g, row, col = heapq.heappop(open_list)
            current = (row, col)

            if current in explored:
                continue

            explored.add(current)
            nodes_explored += 1

            # Goal reached
            if current == goal:
                path = self._reconstruct_path(parent, goal)
                total_cost = g
                break

            # Expand neighbors
            for nr, nc, move_cost in self.grid.get_neighbors(row, col, self.connectivity):
                neighbor = (nr, nc)
                new_g = g + move_cost

                if neighbor not in g_cost or new_g < g_cost[neighbor]:
                    g_cost[neighbor] = new_g
                    f_new = new_g + self._h(neighbor, goal)
                    parent[neighbor] = current
                    heapq.heappush(open_list, (f_new, new_g, nr, nc))

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
    # Same grid and test as Dijkstra for direct comparison
    g = Grid(20, 20, obstacle_density=0.2, seed=42)
    start = (0, 0)
    goal = (19, 19)

    astar = AStar(g, connectivity=4, heuristic="euclidean")
    result = astar.search(start, goal)

    if result["path"]:
        print(f"Path found! Length: {len(result['path'])} cells")
        print(f"Cost: {result['cost']:.2f}")
        print(f"Nodes explored: {result['nodes_explored']}")
        print(f"Time: {result['planning_time_ms']:.2f} ms")
        print(f"Memory: {result['memory_mb']:.4f} MB")
    else:
        print("No path found.")
