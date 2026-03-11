import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import time
import tracemalloc
from scipy.spatial import KDTree
from grid import Grid


class RRTConnectKD:
    """Bidirectional RRT-Connect with KD-Tree optimized nearest neighbor."""

    def __init__(self, grid, step_size=5, max_iterations=5000, connect_threshold=2.0):
        self.grid = grid
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.connect_threshold = connect_threshold

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

    def _extend(self, tree, tree_array, parent_map, target, kd_tree):
        _, nearest_idx = kd_tree.query(list(target))
        nearest = tree[nearest_idx]
        new_node = self._steer(nearest, target)

        if self._is_collision_free(nearest, new_node):
            tree.append(new_node)
            tree_array.append(list(new_node))
            parent_map[new_node] = nearest

            if self._distance(new_node, target) < self.connect_threshold:
                return "reached", new_node
            return "advanced", new_node
        return "trapped", None

    def _connect(self, tree, tree_array, parent_map, target, kd_tree):
        while True:
            # Rebuild KDTree with new nodes
            kd_tree = KDTree(tree_array)
            status, new_node = self._extend(tree, tree_array, parent_map, target, kd_tree)
            if status == "reached":
                return "reached", new_node
            elif status == "trapped":
                return "trapped", None

    def search(self, start, goal):
        tracemalloc.start()
        start_time = time.perf_counter()

        start = (float(start[0]), float(start[1]))
        goal = (float(goal[0]), float(goal[1]))

        tree_a = [start]
        tree_a_array = [list(start)]
        tree_b = [goal]
        tree_b_array = [list(goal)]
        parent_a = {start: None}
        parent_b = {goal: None}

        path = None
        connect_node_a = None
        connect_node_b = None

        for i in range(self.max_iterations):
            # Rebuild KDTrees
            kd_a = KDTree(tree_a_array)

            rand_row = np.random.uniform(0, self.grid.height - 1)
            rand_col = np.random.uniform(0, self.grid.width - 1)
            random_point = (rand_row, rand_col)

            status_a, new_node_a = self._extend(tree_a, tree_a_array, parent_a, random_point, kd_a)

            if status_a != "trapped" and new_node_a is not None:
                kd_b = KDTree(tree_b_array)
                status_b, new_node_b = self._connect(tree_b, tree_b_array, parent_b, new_node_a, kd_b)

                if status_b == "reached" and new_node_b is not None:
                    connect_node_a = new_node_a
                    connect_node_b = new_node_b
                    break

            # Swap trees
            tree_a, tree_b = tree_b, tree_a
            tree_a_array, tree_b_array = tree_b_array, tree_a_array
            parent_a, parent_b = parent_b, parent_a

        # Reconstruct path
        if connect_node_a is not None:
            if start in parent_a:
                path_a = self._trace_path(parent_a, connect_node_a)
                path_b = self._trace_path(parent_b, connect_node_b)
            else:
                path_a = self._trace_path(parent_b, connect_node_b)
                path_b = self._trace_path(parent_a, connect_node_a)
                path_a, path_b = path_b, path_a

            if path_a and path_b:
                if self._distance(path_a[0], start) > self._distance(path_a[-1], start):
                    path_a.reverse()
                if self._distance(path_b[0], goal) > self._distance(path_b[-1], goal):
                    path_b.reverse()
                path_b.reverse()
                path = path_a + path_b

        end_time = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cost = None
        if path:
            cost = sum(self._distance(path[i], path[i+1]) for i in range(len(path)-1))

        all_tree_a = tree_a if start in parent_a else tree_b
        all_tree_b = tree_b if start in parent_a else tree_a

        return {
            "path": path,
            "cost": cost,
            "nodes_explored": len(tree_a) + len(tree_b),
            "planning_time_ms": (end_time - start_time) * 1000,
            "memory_mb": peak_memory / (1024 * 1024),
            "tree_start": all_tree_a,
            "tree_goal": all_tree_b,
            "parent_start": parent_a if start in parent_a else parent_b,
            "parent_goal": parent_b if start in parent_a else parent_a,
        }

    def _trace_path(self, parent_map, node):
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = parent_map.get(current)
        path.reverse()
        return path


if __name__ == "__main__":
    g = Grid(20, 20, obstacle_density=0.2, seed=42)
    start = (0, 0)
    goal = (19, 19)

    np.random.seed(42)
    rrt = RRTConnectKD(g, step_size=3, max_iterations=5000)
    result = rrt.search(start, goal)

    if result["path"]:
        print(f"Path found! Waypoints: {len(result['path'])}")
        print(f"Cost: {result['cost']:.2f}")
        print(f"Tree nodes: {result['nodes_explored']}")
        print(f"Time: {result['planning_time_ms']:.2f} ms")
        print(f"Memory: {result['memory_mb']:.4f} MB")
    else:
        print("No path found.")
