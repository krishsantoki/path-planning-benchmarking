import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import time
from grid import Grid
from rrt_connect_kd import RRTConnectKD
from rrt_star_kd import RRTStarKD


class DijkstraRecorder:
    """Dijkstra that records exploration order."""
    def __init__(self, grid, connectivity=4):
        self.grid = grid
        self.connectivity = connectivity

    def search(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0.0, start[0], start[1]))
        g_cost = {start: 0.0}
        parent = {start: None}
        explored = set()
        exploration_order = []

        while open_list:
            cost, row, col = heapq.heappop(open_list)
            current = (row, col)
            if current in explored:
                continue
            explored.add(current)
            exploration_order.append(current)
            if current == goal:
                path = []
                c = goal
                while c is not None:
                    path.append(c)
                    c = parent[c]
                path.reverse()
                return {"path": path, "cost": cost, "exploration_order": exploration_order,
                        "nodes_explored": len(explored)}
            for nr, nc, mc in self.grid.get_neighbors(row, col, self.connectivity):
                neighbor = (nr, nc)
                new_cost = cost + mc
                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    parent[neighbor] = current
                    heapq.heappush(open_list, (new_cost, nr, nc))
        return {"path": None, "cost": None, "exploration_order": exploration_order,
                "nodes_explored": len(explored)}


class AStarRecorder:
    """A* that records exploration order."""
    def __init__(self, grid, connectivity=4):
        self.grid = grid
        self.connectivity = connectivity

    def _h(self, node, goal):
        return np.sqrt((node[0]-goal[0])**2 + (node[1]-goal[1])**2)

    def search(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (self._h(start, goal), 0.0, start[0], start[1]))
        g_cost = {start: 0.0}
        parent = {start: None}
        explored = set()
        exploration_order = []

        while open_list:
            f, g, row, col = heapq.heappop(open_list)
            current = (row, col)
            if current in explored:
                continue
            explored.add(current)
            exploration_order.append(current)
            if current == goal:
                path = []
                c = goal
                while c is not None:
                    path.append(c)
                    c = parent[c]
                path.reverse()
                return {"path": path, "cost": g, "exploration_order": exploration_order,
                        "nodes_explored": len(explored)}
            for nr, nc, mc in self.grid.get_neighbors(row, col, self.connectivity):
                neighbor = (nr, nc)
                new_g = g + mc
                if neighbor not in g_cost or new_g < g_cost[neighbor]:
                    g_cost[neighbor] = new_g
                    parent[neighbor] = current
                    heapq.heappush(open_list, (new_g + self._h(neighbor, goal), new_g, nr, nc))
        return {"path": None, "cost": None, "exploration_order": exploration_order,
                "nodes_explored": len(explored)}


def build_grid_display(grid):
    """Create base RGB array for grid."""
    display = np.ones((grid.height, grid.width, 3))
    for r in range(grid.height):
        for c in range(grid.width):
            if grid.grid[r, c] == 1:
                display[r, c] = [0.15, 0.15, 0.15]
    return display


def main():
    print("Running algorithms...")
    g = Grid(30, 30, obstacle_density=0.15, seed=42)
    start = (0, 0)
    goal = (29, 29)

    # Run grid-based algorithms
    dijk = DijkstraRecorder(g, 4)
    r_dijk = dijk.search(start, goal)

    astar = AStarRecorder(g, 4)
    r_astar = astar.search(start, goal)

    # Run RRT-Connect
    np.random.seed(42)
    rrt_c = RRTConnectKD(g, step_size=4, max_iterations=5000, connect_threshold=2.0)
    r_rrt_c = rrt_c.search(start, goal)

    # Run RRT* with minimal iterations
    np.random.seed(42)
    rrt_s = RRTStarKD(g, step_size=6, max_iterations=500,
                      goal_bias=0.2, goal_threshold=6.0, rewire_radius=12.0)
    r_rrt_s = rrt_s.search(start, goal)

    print(f"Dijkstra: {'found' if r_dijk['path'] else 'failed'}")
    print(f"A*: {'found' if r_astar['path'] else 'failed'}")
    print(f"RRT-Connect: {'found' if r_rrt_c['path'] else 'failed'}")
    print(f"RRT*: {'found' if r_rrt_s['path'] else 'failed'}")

    # Normalize frame count — all algorithms animate over same number of frames
    total_frames = 100
    path_frames = 30

    # Prepare exploration data
    dijk_order = r_dijk["exploration_order"]
    astar_order = r_astar["exploration_order"]

    # RRT edges
    rrt_c_edges = [(par, node) for node, par in r_rrt_c.get("parent_start", {}).items() if par is not None]
    rrt_c_edges += [(par, node) for node, par in r_rrt_c.get("parent_goal", {}).items() if par is not None]

    rrt_s_edges = [(par, node) for node, par in r_rrt_s.get("parent_map", {}).items() if par is not None]

    # Build figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Path Planning — Algorithm Comparison',
                 fontsize=14, fontweight='bold')

    base = build_grid_display(g)
    imgs = []
    for ax in axes:
        img = ax.imshow(base.copy(), interpolation='nearest')
        ax.set_xlim(-0.5, g.width - 0.5)
        ax.set_ylim(g.height - 0.5, -0.5)
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        imgs.append(img)

    axes[0].set_title("Dijkstra", fontsize=12, fontweight='bold')
    axes[1].set_title("A*", fontsize=12, fontweight='bold')
    axes[2].set_title("RRT-Connect", fontsize=12, fontweight='bold')
    axes[3].set_title("RRT*", fontsize=12, fontweight='bold')

    # Store drawn lines for RRT panels
    rrt_c_lines = []
    rrt_s_lines = []
    path_lines = [[], [], [], []]

    def update(frame):
        # Phase 1: Exploration (frames 0 to total_frames-1)
        if frame < total_frames:
            progress = frame / total_frames

            # Dijkstra exploration
            d = base.copy()
            n_dijk = int(progress * len(dijk_order))
            for r, c in dijk_order[:n_dijk]:
                if g.grid[r, c] == 0:
                    d[r, c] = [0.68, 0.85, 0.95]
            d[start[0], start[1]] = [0.2, 0.8, 0.2]
            d[goal[0], goal[1]] = [1.0, 0.6, 0.0]
            imgs[0].set_data(d)

            # A* exploration
            d2 = base.copy()
            n_astar = int(progress * len(astar_order))
            for r, c in astar_order[:n_astar]:
                if g.grid[r, c] == 0:
                    d2[r, c] = [0.68, 0.85, 0.95]
            d2[start[0], start[1]] = [0.2, 0.8, 0.2]
            d2[goal[0], goal[1]] = [1.0, 0.6, 0.0]
            imgs[1].set_data(d2)

            # RRT-Connect tree growth
            n_rrt_c = int(progress * len(rrt_c_edges))
            for line in rrt_c_lines:
                line.remove()
            rrt_c_lines.clear()
            for par, node in rrt_c_edges[:n_rrt_c]:
                # Color based on which tree
                line, = axes[2].plot([par[1], node[1]], [par[0], node[0]],
                                     color='#85C1E9', linewidth=0.6, alpha=0.5)
                rrt_c_lines.append(line)

            # RRT* tree growth
            n_rrt_s = int(progress * len(rrt_s_edges))
            for line in rrt_s_lines:
                line.remove()
            rrt_s_lines.clear()
            for par, node in rrt_s_edges[:n_rrt_s]:
                line, = axes[3].plot([par[1], node[1]], [par[0], node[0]],
                                     color='#85C1E9', linewidth=0.4, alpha=0.4)
                rrt_s_lines.append(line)

        # Phase 2: Path drawing (frames total_frames to total_frames+path_frames)
        elif frame < total_frames + path_frames:
            path_progress = (frame - total_frames) / path_frames

            # Draw final exploration for grid-based
            d = base.copy()
            for r, c in dijk_order:
                if g.grid[r, c] == 0:
                    d[r, c] = [0.68, 0.85, 0.95]
            d[start[0], start[1]] = [0.2, 0.8, 0.2]
            d[goal[0], goal[1]] = [1.0, 0.6, 0.0]
            imgs[0].set_data(d)

            d2 = base.copy()
            for r, c in astar_order:
                if g.grid[r, c] == 0:
                    d2[r, c] = [0.68, 0.85, 0.95]
            d2[start[0], start[1]] = [0.2, 0.8, 0.2]
            d2[goal[0], goal[1]] = [1.0, 0.6, 0.0]
            imgs[1].set_data(d2)

            # Dijkstra path
            for line in path_lines[0]:
                line.remove()
            path_lines[0].clear()
            if r_dijk["path"]:
                n = int(path_progress * (len(r_dijk["path"]) - 1))
                for i in range(n):
                    p = r_dijk["path"]
                    line, = axes[0].plot([p[i][1], p[i+1][1]],
                                         [p[i][0], p[i+1][0]],
                                         color='#E74C3C', linewidth=2.5, zorder=4)
                    path_lines[0].append(line)

            # A* path
            for line in path_lines[1]:
                line.remove()
            path_lines[1].clear()
            if r_astar["path"]:
                n = int(path_progress * (len(r_astar["path"]) - 1))
                for i in range(n):
                    p = r_astar["path"]
                    line, = axes[1].plot([p[i][1], p[i+1][1]],
                                         [p[i][0], p[i+1][0]],
                                         color='#E74C3C', linewidth=2.5, zorder=4)
                    path_lines[1].append(line)

            # RRT-Connect path
            for line in path_lines[2]:
                line.remove()
            path_lines[2].clear()
            if r_rrt_c["path"]:
                n = int(path_progress * (len(r_rrt_c["path"]) - 1))
                for i in range(n):
                    p = r_rrt_c["path"]
                    line, = axes[2].plot([p[i][1], p[i+1][1]],
                                         [p[i][0], p[i+1][0]],
                                         color='#E74C3C', linewidth=2.5, zorder=4)
                    path_lines[2].append(line)

            # RRT* path
            for line in path_lines[3]:
                line.remove()
            path_lines[3].clear()
            if r_rrt_s["path"]:
                n = int(path_progress * (len(r_rrt_s["path"]) - 1))
                for i in range(n):
                    p = r_rrt_s["path"]
                    line, = axes[3].plot([p[i][1], p[i+1][1]],
                                         [p[i][0], p[i+1][0]],
                                         color='#E74C3C', linewidth=2.5, zorder=4)
                    path_lines[3].append(line)
            elif frame == total_frames:
                txt = axes[3].text(15, 15, "No path\nfound", ha='center', va='center',
                                    fontsize=12, color='red', fontweight='bold', zorder=5)
                path_lines[3].append(txt)

        return imgs + rrt_c_lines + rrt_s_lines

    total = total_frames + path_frames + 10  # 10 extra frames to hold final image
    print("Building animation...")
    ani = animation.FuncAnimation(fig, update, frames=total,
                                  interval=60, blit=False, repeat=False)
    ani.save("../../visualizations/combined_animation.gif", writer='pillow', fps=15)
    plt.close()
    print("Saved to visualizations/combined_animation.gif")


if __name__ == "__main__":
    main()
