import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from grid import Grid
from dijkstra import Dijkstra
from astar import AStar
from rrt_connect_kd import RRTConnectKD
from rrt_star_kd import RRTStarKD


def draw_grid_base(ax, grid):
    """Draw the obstacle grid as background."""
    display = np.ones((grid.height, grid.width, 3))
    for r in range(grid.height):
        for c in range(grid.width):
            if grid.grid[r, c] == 1:
                display[r, c] = [0.15, 0.15, 0.15]
    ax.imshow(display, interpolation='nearest', extent=[0, grid.width, grid.height, 0])


def draw_explored_cells(ax, explored, color):
    """Draw explored cells for grid-based algorithms."""
    for (r, c) in explored:
        rect = plt.Rectangle((c, r), 1, 1, facecolor=color, alpha=0.3)
        ax.add_patch(rect)


def draw_grid_path(ax, path, color, linewidth=2.5):
    """Draw path for grid-based algorithms (cell centers)."""
    if path:
        cols = [c + 0.5 for r, c in path]
        rows = [r + 0.5 for r, c in path]
        ax.plot(cols, rows, color=color, linewidth=linewidth, zorder=4)


def draw_tree_edges(ax, parent_map, color, linewidth=0.4, alpha=0.4):
    """Draw RRT tree edges."""
    for node, par in parent_map.items():
        if par is not None:
            ax.plot([par[1], node[1]], [par[0], node[0]],
                    color=color, linewidth=linewidth, alpha=alpha)


def draw_rrt_path(ax, path, color, linewidth=2.5):
    """Draw path for RRT algorithms (continuous coordinates)."""
    if path:
        for i in range(len(path) - 1):
            ax.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]],
                    color=color, linewidth=linewidth, zorder=4)


def draw_markers(ax, start, goal, grid_based=True):
    """Draw start and goal markers."""
    if grid_based:
        ax.plot(start[1] + 0.5, start[0] + 0.5, 'o', color='#2ECC40',
                markersize=10, zorder=5, markeredgecolor='white', markeredgewidth=1.5)
        ax.plot(goal[1] + 0.5, goal[0] + 0.5, 'o', color='#FF851B',
                markersize=10, zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    else:
        ax.plot(start[1], start[0], 'o', color='#2ECC40',
                markersize=10, zorder=5, markeredgecolor='white', markeredgewidth=1.5)
        ax.plot(goal[1], goal[0], 'o', color='#FF851B',
                markersize=10, zorder=5, markeredgecolor='white', markeredgewidth=1.5)


def add_metrics(ax, result, algo_name, is_rrt=False):
    """Add metrics text below the plot."""
    if result["path"]:
        path_len = len(result["path"])
        cost = result["cost"]
        nodes = result["nodes_explored"]
        time_ms = result["planning_time_ms"]

        text = (f"Cost: {cost:.1f}  |  "
                f"Nodes: {nodes}  |  "
                f"Time: {time_ms:.1f}ms")
    else:
        text = "No path found"

    ax.set_xlabel(text, fontsize=8, labelpad=8)


def main():
    # Use a grid where all algorithms can find a path
    np.random.seed(42)
    g = Grid(30, 30, obstacle_density=0.15, seed=42)
    start = (0, 0)
    goal = (29, 29)

    # Run all algorithms
    dijkstra = Dijkstra(g, connectivity=4)
    result_dijkstra = dijkstra.search(start, goal)

    astar = AStar(g, connectivity=4, heuristic="euclidean")
    result_astar = astar.search(start, goal)

    np.random.seed(42)
    rrt_conn = RRTConnectKD(g, step_size=4, max_iterations=5000, connect_threshold=2.0)
    result_rrt_conn = rrt_conn.search(start, goal)

    np.random.seed(42)
    rrt_star = RRTStarKD(g, step_size=6, max_iterations=500,
                         goal_bias=0.2, goal_threshold=6.0, rewire_radius=12.0)
    result_rrt_star = rrt_star.search(start, goal)

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle('Path Planning Algorithm Comparison — 30×30 Grid, 15% Obstacles',
                 fontsize=16, fontweight='bold', y=1.02)

    path_color = '#E74C3C'
    explored_color = '#85C1E9'
    tree_color = '#85C1E9'

    # 1. Dijkstra
    ax = axes[0]
    draw_grid_base(ax, g)
    draw_explored_cells(ax, result_dijkstra["explored_cells"], explored_color)
    draw_grid_path(ax, result_dijkstra["path"], path_color)
    draw_markers(ax, start, goal, grid_based=True)
    add_metrics(ax, result_dijkstra, "Dijkstra")
    ax.set_title("Dijkstra", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlim(0, g.width)
    ax.set_ylim(g.height, 0)
    ax.set_aspect('equal')
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    # 2. A*
    ax = axes[1]
    draw_grid_base(ax, g)
    draw_explored_cells(ax, result_astar["explored_cells"], explored_color)
    draw_grid_path(ax, result_astar["path"], path_color)
    draw_markers(ax, start, goal, grid_based=True)
    add_metrics(ax, result_astar, "A*")
    ax.set_title("A*", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlim(0, g.width)
    ax.set_ylim(g.height, 0)
    ax.set_aspect('equal')
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    # 3. RRT-Connect
    ax = axes[2]
    draw_grid_base(ax, g)
    if result_rrt_conn["path"]:
        draw_tree_edges(ax, result_rrt_conn["parent_start"], '#85C1E9')
        draw_tree_edges(ax, result_rrt_conn["parent_goal"], '#ABEBC6')
        draw_rrt_path(ax, result_rrt_conn["path"], path_color)
        draw_markers(ax, (float(start[0]), float(start[1])),
                     (float(goal[0]), float(goal[1])), grid_based=False)
    add_metrics(ax, result_rrt_conn, "RRT-Connect", is_rrt=True)
    ax.set_title("RRT-Connect", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlim(0, g.width)
    ax.set_ylim(g.height, 0)
    ax.set_aspect('equal')
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    # 4. RRT*
    ax = axes[3]
    draw_grid_base(ax, g)
    if result_rrt_star["path"]:
        draw_tree_edges(ax, result_rrt_star["parent_map"], tree_color)
        draw_rrt_path(ax, result_rrt_star["path"], path_color)
        draw_markers(ax, (float(start[0]), float(start[1])),
                     (float(goal[0]), float(goal[1])), grid_based=False)
    else:
        ax.text(g.width/2, g.height/2, "No path\nfound",
                ha='center', va='center', fontsize=14, color='red', fontweight='bold')
        draw_markers(ax, (float(start[0]), float(start[1])),
                     (float(goal[0]), float(goal[1])), grid_based=False)
    add_metrics(ax, result_rrt_star, "RRT*", is_rrt=True)
    ax.set_title("RRT*", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlim(0, g.width)
    ax.set_ylim(g.height, 0)
    ax.set_aspect('equal')
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    plt.tight_layout()
    plt.savefig("visualizations/side_by_side_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to visualizations/side_by_side_comparison.png")


if __name__ == "__main__":
    main()
