import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import heapq
import time
import tracemalloc
from grid import Grid


class DijkstraAnimated:
    """Dijkstra with step-by-step recording for animation."""

    def __init__(self, grid, connectivity=4):
        self.grid = grid
        self.connectivity = connectivity

    def search(self, start, goal):
        tracemalloc.start()
        start_time = time.perf_counter()

        open_list = []
        heapq.heappush(open_list, (0.0, start[0], start[1]))

        g_cost = {start: 0.0}
        parent = {start: None}
        explored = set()
        nodes_explored = 0
        exploration_order = []

        path = None
        total_cost = float('inf')

        while open_list:
            current_cost, row, col = heapq.heappop(open_list)
            current = (row, col)

            if current in explored:
                continue

            explored.add(current)
            nodes_explored += 1
            exploration_order.append(current)

            if current == goal:
                path = self._reconstruct_path(parent, goal)
                total_cost = current_cost
                break

            for nr, nc, move_cost in self.grid.get_neighbors(row, col, self.connectivity):
                neighbor = (nr, nc)
                new_cost = current_cost + move_cost
                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    parent[neighbor] = current
                    heapq.heappush(open_list, (new_cost, nr, nc))

        end_time = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "path": path,
            "cost": total_cost if path else None,
            "nodes_explored": nodes_explored,
            "planning_time_ms": (end_time - start_time) * 1000,
            "memory_mb": peak_memory / (1024 * 1024),
            "explored_cells": explored,
            "exploration_order": exploration_order
        }

    def _reconstruct_path(self, parent, goal):
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path


def visualize(grid, result, title="Dijkstra's Algorithm"):
    """
    Plot the grid showing obstacles, explored cells, and final path.

    Colors:
        White  = free cell
        Black  = obstacle
        Light blue = explored cell
        Red    = final path
        Green  = start
        Orange = goal
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Base grid: white (free) and black (obstacles)
    display = np.ones((grid.height, grid.width, 3))  # RGB, all white
    for r in range(grid.height):
        for c in range(grid.width):
            if grid.grid[r, c] == 1:
                display[r, c] = [0.0, 0.0, 0.0]  # black obstacle

    # Explored cells — light blue
    if result["explored_cells"]:
        for (r, c) in result["explored_cells"]:
            if grid.grid[r, c] == 0:
                display[r, c] = [0.68, 0.85, 0.95]

    # Final path — red
    if result["path"]:
        for (r, c) in result["path"]:
            display[r, c] = [0.9, 0.2, 0.2]

    # Start — green
    start = result["path"][0] if result["path"] else (0, 0)
    display[start[0], start[1]] = [0.2, 0.8, 0.2]

    # Goal — orange
    goal = result["path"][-1] if result["path"] else (grid.height - 1, grid.width - 1)
    display[goal[0], goal[1]] = [1.0, 0.6, 0.0]

    ax.imshow(display, interpolation='nearest')

    # Grid lines
    ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.3)
    ax.tick_params(which='minor', size=0)

    # Metrics text
    metrics = (
        f"Path Length: {len(result['path'])} cells\n"
        f"Cost: {result['cost']:.2f}\n"
        f"Nodes Explored: {result['nodes_explored']}\n"
        f"Time: {result['planning_time_ms']:.2f} ms\n"
        f"Memory: {result['memory_mb']:.4f} MB"
    )
    ax.text(
        grid.width + 0.5, grid.height / 2, metrics,
        fontsize=11, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=[0.0, 0.0, 0.0], label='Obstacle'),
        mpatches.Patch(color=[0.68, 0.85, 0.95], label='Explored'),
        mpatches.Patch(color=[0.9, 0.2, 0.2], label='Path'),
        mpatches.Patch(color=[0.2, 0.8, 0.2], label='Start'),
        mpatches.Patch(color=[1.0, 0.6, 0.0], label='Goal'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("../../visualizations/dijkstra_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to visualizations/dijkstra_result.png")


def visualize_animated(grid, result, title="Dijkstra's Algorithm", filename="dijkstra_animation.gif"):
    """Animated GIF showing node-by-node expansion then final path."""
    fig, ax = plt.subplots(figsize=(8, 8))

    base = np.ones((grid.height, grid.width, 3))
    for r in range(grid.height):
        for c in range(grid.width):
            if grid.grid[r, c] == 1:
                base[r, c] = [0.0, 0.0, 0.0]

    start = result["path"][0]
    goal = result["path"][-1]

    display = base.copy()
    display[start[0], start[1]] = [0.2, 0.8, 0.2]
    display[goal[0], goal[1]] = [1.0, 0.6, 0.0]

    img = ax.imshow(display, interpolation='nearest')
    ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.3)
    ax.tick_params(which='minor', size=0)
    ax.set_title(title, fontsize=14, fontweight='bold')

    exploration_order = result["exploration_order"]
    path = result["path"]

    step = max(1, len(exploration_order) // 80)
    frames = []

    for i in range(0, len(exploration_order), step):
        frames.append(("explore", exploration_order[:i + step]))
    frames.append(("explore", exploration_order))
    for i in range(len(path)):
        frames.append(("path", path[:i + 1]))

    def update(frame_idx):
        frame_type, cells = frames[frame_idx]
        d = base.copy()
        d[start[0], start[1]] = [0.2, 0.8, 0.2]
        d[goal[0], goal[1]] = [1.0, 0.6, 0.0]

        if frame_type == "explore":
            for (r, c) in cells:
                if (r, c) != start and (r, c) != goal and grid.grid[r, c] == 0:
                    d[r, c] = [0.68, 0.85, 0.95]
        elif frame_type == "path":
            for (r, c) in exploration_order:
                if (r, c) != start and (r, c) != goal and grid.grid[r, c] == 0:
                    d[r, c] = [0.68, 0.85, 0.95]
            for (r, c) in cells:
                if (r, c) != start and (r, c) != goal:
                    d[r, c] = [0.9, 0.2, 0.2]
            d[start[0], start[1]] = [0.2, 0.8, 0.2]
            d[goal[0], goal[1]] = [1.0, 0.6, 0.0]

        img.set_data(d)
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=50, blit=True, repeat=False)
    ani.save(f"../../visualizations/{filename}", writer='pillow', fps=15)
    plt.show()
    print(f"Saved to visualizations/{filename}")


if __name__ == "__main__":
    g = Grid(20, 20, obstacle_density=0.2, seed=42)
    start = (0, 0)
    goal = (19, 19)

    dijkstra = DijkstraAnimated(g, connectivity=4)
    result = dijkstra.search(start, goal)

    if result["path"]:
        print(f"Path found! Length: {len(result['path'])} cells")
        print(f"Nodes explored: {result['nodes_explored']}")

        visualize(g, result)
        visualize_animated(g, result)
    else:
        print("No path found.")
