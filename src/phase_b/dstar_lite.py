import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import heapq
import time
import tracemalloc
import numpy as np
from grid import Grid


class DStarLite:
    """
    D* Lite algorithm for incremental replanning on dynamic grids.

    Plans backwards from goal to start. When obstacles change,
    only affected nodes are re-expanded instead of replanning from scratch.
    """

    def __init__(self, grid, connectivity=4):
        self.grid = grid
        self.connectivity = connectivity

        self.g = {}      # g-values (cost from node to goal)
        self.rhs = {}    # rhs-values (one-step lookahead)
        self.U = []      # Priority queue
        self.km = 0      # Key modifier for consistency after robot moves

        self.start = None
        self.goal = None

        # For metrics
        self.nodes_expanded = 0
        self.replan_times = []

    def _heuristic(self, a, b):
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _calculate_key(self, s):
        """Calculate priority key for node s."""
        g_val = self.g.get(s, float('inf'))
        rhs_val = self.rhs.get(s, float('inf'))
        min_val = min(g_val, rhs_val)
        return (min_val + self._heuristic(self.start, s) + self.km, min_val)

    def _get_neighbors(self, row, col):
        """Get all valid neighbors with costs (including obstacles for replanning)."""
        neighbors = []
        cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in cardinal:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.grid.height and 0 <= nc < self.grid.width:
                if self.grid.grid[nr, nc] == 0:
                    neighbors.append(((nr, nc), 1.0))
                else:
                    neighbors.append(((nr, nc), float('inf')))

        if self.connectivity == 8:
            for dr, dc in diagonal:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.grid.height and 0 <= nc < self.grid.width:
                    if self.grid.grid[nr, nc] == 0:
                        neighbors.append(((nr, nc), np.sqrt(2)))
                    else:
                        neighbors.append(((nr, nc), float('inf')))

        return neighbors

    def _get_predecessors(self, s):
        """Get cells that can reach s (same as neighbors on a grid)."""
        return self._get_neighbors(s[0], s[1])

    def _get_successors(self, s):
        """Get cells reachable from s."""
        return self._get_neighbors(s[0], s[1])

    def _update_vertex(self, u):
        """Update rhs value and queue membership for node u."""
        if u != self.goal:
            # rhs(u) = min over successors s' of (c(u,s') + g(s'))
            min_val = float('inf')
            for neighbor, cost in self._get_successors(u):
                g_val = self.g.get(neighbor, float('inf'))
                if cost + g_val < min_val:
                    min_val = cost + g_val
            self.rhs[u] = min_val

        # Remove u from queue if present
        self.U = [(k, s) for k, s in self.U if s != u]
        heapq.heapify(self.U)

        # If inconsistent, add back to queue
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            heapq.heappush(self.U, (self._calculate_key(u), u))

    def _compute_shortest_path(self):
        """Expand nodes until start is consistent and has minimum key."""
        while True:
            if not self.U:
                break

            # Check top key vs start key
            top_key, _ = self.U[0]
            start_key = self._calculate_key(self.start)
            g_start = self.g.get(self.start, float('inf'))
            rhs_start = self.rhs.get(self.start, float('inf'))

            if not (top_key < start_key or rhs_start != g_start):
                break

            k_old, u = heapq.heappop(self.U)
            self.nodes_expanded += 1
            k_new = self._calculate_key(u)

            if k_old < k_new:
                # Reinsert with updated key
                heapq.heappush(self.U, (k_new, u))
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                # Overconsistent — make consistent
                self.g[u] = self.rhs[u]
                for pred, cost in self._get_predecessors(u):
                    self._update_vertex(pred)
            else:
                # Underconsistent — reset
                self.g[u] = float('inf')
                self._update_vertex(u)
                for pred, cost in self._get_predecessors(u):
                    self._update_vertex(pred)

    def initialize(self, start, goal):
        """Initialize D* Lite before first search."""
        self.start = start
        self.goal = goal
        self.km = 0
        self.g = {}
        self.rhs = {}
        self.U = []
        self.nodes_expanded = 0
        self.replan_times = []

        # Initialize all cells to infinity
        self.rhs[self.goal] = 0
        heapq.heappush(self.U, (self._calculate_key(self.goal), self.goal))

    def initial_plan(self):
        """Compute initial shortest path."""
        tracemalloc.start()
        start_time = time.perf_counter()

        self._compute_shortest_path()

        end_time = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        plan_time = (end_time - start_time) * 1000
        self.replan_times.append(("initial", plan_time))

        return {
            "planning_time_ms": plan_time,
            "memory_mb": peak_memory / (1024 * 1024),
            "nodes_expanded": self.nodes_expanded,
        }

    def replan(self, changed_cells, new_start=None):
        """
        Incrementally update the plan after obstacle changes.

        Args:
            changed_cells: List of (row, col) cells that changed.
            new_start: Updated robot position (if robot has moved).

        Returns:
            dict with replan metrics.
        """
        start_time = time.perf_counter()
        nodes_before = self.nodes_expanded

        if new_start is not None:
            # Update km for consistency
            self.km += self._heuristic(self.start, new_start)
            self.start = new_start

        # Update affected vertices
        for (row, col) in changed_cells:
            cell = (row, col)
            # Update the cell itself
            self._update_vertex(cell)
            # Update all neighbors of changed cell
            for neighbor, cost in self._get_neighbors(row, col):
                self._update_vertex(neighbor)

        self._compute_shortest_path()

        end_time = time.perf_counter()
        replan_time = (end_time - start_time) * 1000
        nodes_replanned = self.nodes_expanded - nodes_before
        self.replan_times.append(("replan", replan_time))

        return {
            "replan_time_ms": replan_time,
            "nodes_replanned": nodes_replanned,
        }

    def get_path(self):
        """Extract current path from start to goal by following minimum g-values."""
        if self.g.get(self.start, float('inf')) == float('inf'):
            return None

        path = [self.start]
        current = self.start
        max_steps = self.grid.width * self.grid.height

        for _ in range(max_steps):
            if current == self.goal:
                break

            # Move to successor with minimum (cost + g)
            best_next = None
            best_cost = float('inf')

            for neighbor, move_cost in self._get_successors(current):
                g_val = self.g.get(neighbor, float('inf'))
                total = move_cost + g_val
                if total < best_cost:
                    best_cost = total
                    best_next = neighbor

            if best_next is None or best_cost == float('inf'):
                return None

            path.append(best_next)
            current = best_next

        if current == self.goal:
            cost = sum(1.0 for _ in range(len(path) - 1))  # Simplified
            return path
        return None

    def get_path_cost(self):
        """Get cost of current path from start."""
        return self.g.get(self.start, float('inf'))


if __name__ == "__main__":
    from dynamic_grid import DynamicGrid

    g = DynamicGrid(20, 20, obstacle_density=0.15, seed=42)
    start = (0, 0)
    goal = (19, 19)

    dstar = DStarLite(g, connectivity=4)
    dstar.initialize(start, goal)

    # Initial plan
    metrics = dstar.initial_plan()
    path = dstar.get_path()
    print(f"Initial plan:")
    print(f"  Path length: {len(path) if path else 'No path'}")
    print(f"  Cost: {dstar.get_path_cost():.2f}")
    print(f"  Time: {metrics['planning_time_ms']:.2f} ms")
    print(f"  Nodes expanded: {metrics['nodes_expanded']}")

    # Simulate obstacle change
    print(f"\nAdding obstacle at (10, 10)...")
    g.update_obstacle(10, 10, True)
    replan_metrics = dstar.replan([(10, 10)])
    path = dstar.get_path()
    print(f"After replan:")
    print(f"  Path length: {len(path) if path else 'No path'}")
    print(f"  Cost: {dstar.get_path_cost():.2f}")
    print(f"  Replan time: {replan_metrics['replan_time_ms']:.2f} ms")
    print(f"  Nodes replanned: {replan_metrics['nodes_replanned']}")

    # Add obstacle on the path
    if path and len(path) > 5:
        block_cell = path[5]
        print(f"\nBlocking path at {block_cell}...")
        g.update_obstacle(block_cell[0], block_cell[1], True)
        replan_metrics = dstar.replan([block_cell])
        path = dstar.get_path()
        print(f"After replan:")
        print(f"  Path length: {len(path) if path else 'No path'}")
        print(f"  Cost: {dstar.get_path_cost():.2f}")
        print(f"  Replan time: {replan_metrics['replan_time_ms']:.2f} ms")
        print(f"  Nodes replanned: {replan_metrics['nodes_replanned']}")
