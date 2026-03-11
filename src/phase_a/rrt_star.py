import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import time
import tracemalloc
from grid import Grid


class RRTStar:
    """RRT* algorithm with rewiring for optimal path planning."""

    def __init__(self, grid, step_size=5, max_iterations=3000,
                 goal_bias=0.1, goal_threshold=3.0, rewire_radius=8.0):
        """
        Args:
            grid: Grid object.
            step_size: Distance to extend tree each step.
            max_iterations: Max sampling attempts.
            goal_bias: Probability of sampling the goal directly.
            goal_threshold: Distance to consider goal reached.
            rewire_radius: Radius to search for rewiring candidates.
        """
        self.grid = grid
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias
        self.goal_threshold = goal_threshold
        self.rewire_radius = rewire_radius

    def _distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _steer(self, from_node, to_node):
        dist = self._distance(from_node, to_node)
        if dist < self.step_size:
            return to_node
        ratio = self.step_size / dist
        new_row = from_node[0] + ratio * (to_node[0] - from_node[0])
        new_col = from_node[1] + ratio * (to_node[1] - from_node[1])
        return (new_row, new_col)

    def _is_collision_free(self, from_node, to_node):
        dist = self._distance(from_node, to_node)
        if dist == 0:
            return True
        steps = int(dist * 2) + 1
        for i in range(steps + 1):
            t = i / steps
            r = int(round(from_node[0] + t * (to_node[0] - from_node[0])))
            c = int(round(from_node[1] + t * (to_node[1] - from_node[1])))
            if not self.grid.is_free(r, c):
                return False
        return True

    def _nearest(self, tree, point):
        min_dist = float('inf')
        nearest_node = None
        for node in tree:
            d = self._distance(node, point)
            if d < min_dist:
                min_dist = d
                nearest_node = node
        return nearest_node

    def _near(self, tree, point, radius):
        """Find all nodes in tree within radius of point."""
        return [node for node in tree if self._distance(node, point) <= radius]

    def search(self, start, goal):
        """
        Find path from start to goal using RRT*.

        Returns:
            dict with: path, cost, nodes_explored, planning_time_ms, memory_mb,
                       tree, parent_map, cost_map, path_cost_history.
        """
        tracemalloc.start()
        start_time = time.perf_counter()

        start = (float(start[0]), float(start[1]))
        goal = (float(goal[0]), float(goal[1]))

        tree = [start]
        parent = {start: None}
        cost = {start: 0.0}  # Cost from start to each node

        best_goal_node = None
        best_goal_cost = float('inf')
        path_cost_history = []  # Track how path cost improves over iterations

        for i in range(self.max_iterations):
            # Sample: goal bias or random
            if np.random.random() < self.goal_bias:
                random_point = goal
            else:
                rand_row = np.random.uniform(0, self.grid.height - 1)
                rand_col = np.random.uniform(0, self.grid.width - 1)
                random_point = (rand_row, rand_col)

            # Find nearest and steer
            nearest = self._nearest(tree, random_point)
            new_node = self._steer(nearest, random_point)

            if not self._is_collision_free(nearest, new_node):
                continue

            # Find nearby nodes for rewiring
            nearby = self._near(tree, new_node, self.rewire_radius)

            # Choose best parent among nearby nodes
            best_parent = nearest
            best_cost = cost[nearest] + self._distance(nearest, new_node)

            for near_node in nearby:
                new_cost = cost[near_node] + self._distance(near_node, new_node)
                if new_cost < best_cost and self._is_collision_free(near_node, new_node):
                    best_parent = near_node
                    best_cost = new_cost

            # Add node to tree
            tree.append(new_node)
            parent[new_node] = best_parent
            cost[new_node] = best_cost

            # Rewire: check if new_node is a better parent for nearby nodes
            for near_node in nearby:
                rewire_cost = cost[new_node] + self._distance(new_node, near_node)
                if rewire_cost < cost[near_node] and self._is_collision_free(new_node, near_node):
                    parent[near_node] = new_node
                    cost[near_node] = rewire_cost

            # Check if goal reached
            if self._distance(new_node, goal) < self.goal_threshold:
                goal_cost = cost[new_node] + self._distance(new_node, goal)
                if goal_cost < best_goal_cost:
                    best_goal_node = new_node
                    best_goal_cost = goal_cost

            # Record best cost so far every 100 iterations
            if i % 100 == 0:
                path_cost_history.append((i, best_goal_cost))

        # Reconstruct path
        path = None
        if best_goal_node is not None:
            path = []
            current = best_goal_node
            while current is not None:
                path.append(current)
                current = parent.get(current)
            path.reverse()
            path.append(goal)  # Add actual goal

        end_time = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "path": path,
            "cost": best_goal_cost if path else None,
            "nodes_explored": len(tree),
            "planning_time_ms": (end_time - start_time) * 1000,
            "memory_mb": peak_memory / (1024 * 1024),
            "tree": tree,
            "parent_map": parent,
            "cost_map": cost,
            "path_cost_history": path_cost_history
        }


if __name__ == "__main__":
    g = Grid(20, 20, obstacle_density=0.2, seed=42)
    start = (0, 0)
    goal = (19, 19)

    np.random.seed(42)
    rrt_star = RRTStar(g, step_size=3, max_iterations=3000,
                       goal_bias=0.1, goal_threshold=3.0, rewire_radius=8.0)
    result = rrt_star.search(start, goal)

    if result["path"]:
        print(f"Path found! Waypoints: {len(result['path'])}")
        print(f"Cost: {result['cost']:.2f}")
        print(f"Tree nodes: {result['nodes_explored']}")
        print(f"Time: {result['planning_time_ms']:.2f} ms")
        print(f"Memory: {result['memory_mb']:.4f} MB")

        # Show how cost improved over iterations
        print("\nCost convergence:")
        for iteration, c in result["path_cost_history"]:
            if c < float('inf'):
                print(f"  Iteration {iteration}: cost = {c:.2f}")
    else:
        print("No path found.")
