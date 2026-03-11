import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from grid import Grid
from rrt_connect import RRTConnect


def visualize_static(grid, result, title="RRT-Connect"):
    """Static PNG showing both trees and final path."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw grid
    display = np.ones((grid.height, grid.width, 3))
    for r in range(grid.height):
        for c in range(grid.width):
            if grid.grid[r, c] == 1:
                display[r, c] = [0.0, 0.0, 0.0]

    ax.imshow(display, interpolation='nearest', extent=[0, grid.width, grid.height, 0])

    # Draw tree edges - start tree (blue)
    parent_start = result["parent_start"]
    for node, par in parent_start.items():
        if par is not None:
            ax.plot([par[1], node[1]], [par[0], node[0]],
                    color=[0.68, 0.85, 0.95], linewidth=0.8, alpha=0.7)

    # Draw tree edges - goal tree (light green)
    parent_goal = result["parent_goal"]
    for node, par in parent_goal.items():
        if par is not None:
            ax.plot([par[1], node[1]], [par[0], node[0]],
                    color=[0.7, 0.95, 0.7], linewidth=0.8, alpha=0.7)

    # Draw final path (red, thick)
    if result["path"]:
        path = result["path"]
        for i in range(len(path) - 1):
            ax.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]],
                    color=[0.9, 0.2, 0.2], linewidth=2.5)

    # Start and goal markers
    if result["path"]:
        ax.plot(result["path"][0][1], result["path"][0][0], 'o',
                color=[0.2, 0.8, 0.2], markersize=12, zorder=5)
        ax.plot(result["path"][-1][1], result["path"][-1][0], 'o',
                color=[1.0, 0.6, 0.0], markersize=12, zorder=5)

    # Metrics
    metrics = (
        f"Waypoints: {len(result['path']) if result['path'] else 0}\n"
        f"Cost: {result['cost']:.2f}\n"
        f"Tree Nodes: {result['nodes_explored']}\n"
        f"Time: {result['planning_time_ms']:.2f} ms\n"
        f"Memory: {result['memory_mb']:.4f} MB"
    )
    ax.text(grid.width + 0.5, grid.height / 2, metrics, fontsize=11,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    legend_patches = [
        mpatches.Patch(color=[0.0, 0.0, 0.0], label='Obstacle'),
        mpatches.Patch(color=[0.68, 0.85, 0.95], label='Start Tree'),
        mpatches.Patch(color=[0.7, 0.95, 0.7], label='Goal Tree'),
        mpatches.Patch(color=[0.9, 0.2, 0.2], label='Path'),
        mpatches.Patch(color=[0.2, 0.8, 0.2], label='Start'),
        mpatches.Patch(color=[1.0, 0.6, 0.0], label='Goal'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, grid.width)
    ax.set_ylim(grid.height, 0)

    plt.tight_layout()
    plt.savefig("../../visualizations/rrt_connect_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to visualizations/rrt_connect_result.png")


def visualize_animated(grid, result, title="RRT-Connect", filename="rrt_connect_animation.gif"):
    """Animated GIF showing both trees growing and connecting."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw grid background
    display = np.ones((grid.height, grid.width, 3))
    for r in range(grid.height):
        for c in range(grid.width):
            if grid.grid[r, c] == 1:
                display[r, c] = [0.0, 0.0, 0.0]

    ax.imshow(display, interpolation='nearest', extent=[0, grid.width, grid.height, 0])
    ax.set_xlim(0, grid.width)
    ax.set_ylim(grid.height, 0)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Collect all edges in order for animation
    edges_start = []
    parent_start = result["parent_start"]
    for node, par in parent_start.items():
        if par is not None:
            edges_start.append((par, node))

    edges_goal = []
    parent_goal = result["parent_goal"]
    for node, par in parent_goal.items():
        if par is not None:
            edges_goal.append((par, node))

    # Interleave edges from both trees
    all_edges = []
    max_len = max(len(edges_start), len(edges_goal))
    for i in range(max_len):
        if i < len(edges_start):
            all_edges.append(("start", edges_start[i]))
        if i < len(edges_goal):
            all_edges.append(("goal", edges_goal[i]))

    path = result["path"]

    # Start and goal markers
    start_pt = result["tree_start"][0]
    goal_pt = result["tree_goal"][0]
    ax.plot(start_pt[1], start_pt[0], 'o', color=[0.2, 0.8, 0.2], markersize=12, zorder=5)
    ax.plot(goal_pt[1], goal_pt[0], 'o', color=[1.0, 0.6, 0.0], markersize=12, zorder=5)

    # Build frames: groups of edges, then path segments
    edges_per_frame = max(1, len(all_edges) // 60)
    frames = []
    for i in range(0, len(all_edges), edges_per_frame):
        frames.append(("tree", all_edges[:i + edges_per_frame]))
    frames.append(("tree", all_edges))

    if path:
        for i in range(len(path) - 1):
            frames.append(("path", i + 1))

    drawn_lines = []

    def update(frame_idx):
        frame_type, data = frames[frame_idx]

        if frame_type == "tree":
            # Clear previous tree lines
            for line in drawn_lines:
                line.remove()
            drawn_lines.clear()

            for tree_type, (par, node) in data:
                color = [0.68, 0.85, 0.95] if tree_type == "start" else [0.7, 0.95, 0.7]
                line, = ax.plot([par[1], node[1]], [par[0], node[0]],
                                color=color, linewidth=0.8, alpha=0.7)
                drawn_lines.append(line)

        elif frame_type == "path" and path:
            idx = data
            line, = ax.plot([path[idx-1][1], path[idx][1]],
                            [path[idx-1][0], path[idx][0]],
                            color=[0.9, 0.2, 0.2], linewidth=2.5, zorder=4)
            drawn_lines.append(line)

        return drawn_lines

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=80, blit=False, repeat=False)
    ani.save(f"../../visualizations/{filename}", writer='pillow', fps=12)
    plt.show()
    print(f"Saved to visualizations/{filename}")


if __name__ == "__main__":
    g = Grid(20, 20, obstacle_density=0.2, seed=42)
    start = (0, 0)
    goal = (19, 19)

    np.random.seed(42)
    rrt = RRTConnect(g, step_size=3, max_iterations=5000)
    result = rrt.search(start, goal)

    if result["path"]:
        print(f"Path found! Waypoints: {len(result['path'])}")
        print(f"Cost: {result['cost']:.2f}")
        visualize_static(g, result)
        visualize_animated(g, result)
    else:
        print("No path found.")
