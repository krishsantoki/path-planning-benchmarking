import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase_a"))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import time
import copy
from grid import Grid
from dynamic_grid import DynamicGrid, ObstacleAgent
from dstar_lite import DStarLite
from astar import AStar


class ReplanningSimulator:
    """
    Simulate robot navigation with moving obstacles.
    Compare D* Lite incremental replan vs A* full replan.
    """

    def __init__(self, grid_size=20, obstacle_density=0.15, seed=42,
                 num_agents=3, agent_path_length=8, max_steps=100):
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.seed = seed
        self.num_agents = num_agents
        self.agent_path_length = agent_path_length
        self.max_steps = max_steps

    def run(self):
        """Run simulation and return frame-by-frame data for both algorithms."""
        start = (0, 0)
        goal = (self.grid_size - 1, self.grid_size - 1)

        # --- D* Lite simulation ---
        grid_dstar = DynamicGrid(self.grid_size, self.grid_size,
                                  self.obstacle_density, self.seed)
        agents_dstar = self._create_agents(grid_dstar)

        dstar = DStarLite(grid_dstar, connectivity=4)
        dstar.initialize(start, goal)
        dstar.initial_plan()

        dstar_frames = []
        dstar_replan_times = []
        dstar_robot_pos = start
        dstar_path = dstar.get_path()
        dstar_path_idx = 0
        dstar_success = False

        for step in range(self.max_steps):
            grid_dstar.step()

            # Move obstacles
            all_changes = []
            for agent in agents_dstar:
                changes = agent.step(grid_dstar.timestep)
                for r, c, blocked in changes:
                    all_changes.append((r, c))

            # Replan
            replan_start = time.perf_counter()
            if all_changes:
                dstar.replan(all_changes, new_start=dstar_robot_pos)
            replan_end = time.perf_counter()
            replan_time = (replan_end - replan_start) * 1000
            dstar_replan_times.append(replan_time)

            # Get updated path
            dstar_path = dstar.get_path()

            # Save frame
            dstar_frames.append({
                "grid": grid_dstar.grid.copy(),
                "robot_pos": dstar_robot_pos,
                "path": list(dstar_path) if dstar_path else None,
                "agents": [a.current_pos for a in agents_dstar],
                "replan_time_ms": replan_time,
            })

            # Move robot one step along path
            if dstar_path and len(dstar_path) > 1:
                dstar_robot_pos = dstar_path[1]
                if dstar_robot_pos == goal:
                    dstar_success = True
                    break

        # --- A* full replan simulation ---
        grid_astar = DynamicGrid(self.grid_size, self.grid_size,
                                  self.obstacle_density, self.seed)
        agents_astar = self._create_agents(grid_astar)

        astar_frames = []
        astar_replan_times = []
        astar_robot_pos = start
        astar_success = False

        for step in range(self.max_steps):
            grid_astar.step()

            # Move obstacles
            for agent in agents_astar:
                agent.step(grid_astar.timestep)

            # Full replan from current position every step
            replan_start = time.perf_counter()
            astar_algo = AStar(grid_astar, connectivity=4, heuristic="euclidean")
            result = astar_algo.search(astar_robot_pos, goal)
            replan_end = time.perf_counter()
            replan_time = (replan_end - replan_start) * 1000
            astar_replan_times.append(replan_time)

            astar_path = result["path"]

            # Save frame
            astar_frames.append({
                "grid": grid_astar.grid.copy(),
                "robot_pos": astar_robot_pos,
                "path": list(astar_path) if astar_path else None,
                "agents": [a.current_pos for a in agents_astar],
                "replan_time_ms": replan_time,
            })

            # Move robot one step along path
            if astar_path and len(astar_path) > 1:
                astar_robot_pos = astar_path[1]
                if astar_robot_pos == goal:
                    astar_success = True
                    break

        return {
            "dstar_frames": dstar_frames,
            "astar_frames": astar_frames,
            "dstar_replan_times": dstar_replan_times,
            "astar_replan_times": astar_replan_times,
            "dstar_success": dstar_success,
            "astar_success": astar_success,
            "dstar_steps": len(dstar_frames),
            "astar_steps": len(astar_frames),
        }

    def _create_agents(self, grid):
        """Create obstacle agents with consistent paths."""
        rng = np.random.RandomState(self.seed + 100)
        agents = []

        for i in range(self.num_agents):
            while True:
                start_r = rng.randint(3, grid.height - 3)
                start_c = rng.randint(3, grid.width - 3)
                if grid.is_free(start_r, start_c):
                    break

            path = [(start_r, start_c)]
            r, c = start_r, start_c
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            for _ in range(self.agent_path_length - 1):
                rng.shuffle(directions)
                moved = False
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < grid.height and 0 <= nc < grid.width
                            and grid.is_free(nr, nc)
                            and (nr, nc) not in path):
                        path.append((nr, nc))
                        r, c = nr, nc
                        moved = True
                        break
                if not moved:
                    break

            if len(path) >= 3:
                agents.append(ObstacleAgent(path, grid, loop=True))

        return agents


def animate_comparison(sim_results, grid_size, save_path="visualizations/phase_b"):
    """Create side-by-side animation of D* Lite vs A* full replan."""
    os.makedirs(save_path, exist_ok=True)

    dstar_frames = sim_results["dstar_frames"]
    astar_frames = sim_results["astar_frames"]
    max_frames = max(len(dstar_frames), len(astar_frames))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Dynamic Replanning: D* Lite vs A* Full Replan",
                 fontsize=14, fontweight='bold')

    def build_display(frame_data, grid_size):
        grid = frame_data["grid"]
        d = np.ones((grid_size, grid_size, 3))

        # Obstacles
        for r in range(grid_size):
            for c in range(grid_size):
                if grid[r, c] == 1:
                    d[r, c] = [0.15, 0.15, 0.15]

        # Moving obstacle agents (yellow)
        for (ar, ac) in frame_data["agents"]:
            if 0 <= ar < grid_size and 0 <= ac < grid_size:
                d[ar, ac] = [1.0, 0.95, 0.0]

        # Path (light blue)
        if frame_data["path"]:
            for (pr, pc) in frame_data["path"]:
                if 0 <= pr < grid_size and 0 <= pc < grid_size and grid[pr, pc] == 0:
                    d[pr, pc] = [0.68, 0.85, 0.95]

        # Robot (green)
        rr, rc = frame_data["robot_pos"]
        d[rr, rc] = [0.2, 0.8, 0.2]

        # Goal (orange)
        d[grid_size - 1, grid_size - 1] = [1.0, 0.6, 0.0]

        return d

    img1 = ax1.imshow(np.ones((grid_size, grid_size, 3)), interpolation='nearest')
    img2 = ax2.imshow(np.ones((grid_size, grid_size, 3)), interpolation='nearest')

    ax1.set_title("D* Lite (Incremental)", fontsize=12, fontweight='bold')
    ax2.set_title("A* (Full Replan)", fontsize=12, fontweight='bold')

    for ax in [ax1, ax2]:
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    text1 = ax1.text(0.02, 0.98, "", transform=ax1.transAxes, fontsize=9,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    text2 = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, fontsize=9,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def update(frame):
        # D* Lite frame
        if frame < len(dstar_frames):
            df = dstar_frames[frame]
            img1.set_data(build_display(df, grid_size))
            text1.set_text(f"Step {frame+1}\nReplan: {df['replan_time_ms']:.2f}ms")
        
        # A* frame
        if frame < len(astar_frames):
            af = astar_frames[frame]
            img2.set_data(build_display(af, grid_size))
            text2.set_text(f"Step {frame+1}\nReplan: {af['replan_time_ms']:.2f}ms")

        return [img1, img2, text1, text2]

    ani = animation.FuncAnimation(fig, update, frames=max_frames,
                                  interval=200, blit=False, repeat=False)
    gif_path = os.path.join(save_path, "dstar_vs_astar_replan.gif")
    ani.save(gif_path, writer='pillow', fps=5)
    plt.close()
    print(f"Saved animation to {gif_path}")


def plot_replan_times(sim_results, save_path="visualizations/phase_b"):
    """Bar chart comparing replan times."""
    os.makedirs(save_path, exist_ok=True)

    dstar_times = sim_results["dstar_replan_times"]
    astar_times = sim_results["astar_replan_times"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Replanning Time Comparison", fontsize=14, fontweight='bold')

    # Per-step replan time
    steps = range(1, min(len(dstar_times), len(astar_times)) + 1)
    d_times = dstar_times[:len(steps)]
    a_times = astar_times[:len(steps)]

    ax1.plot(steps, d_times, color='#2196F3', linewidth=1.5, label='D* Lite', alpha=0.8)
    ax1.plot(steps, a_times, color='#E91E63', linewidth=1.5, label='A* Full Replan', alpha=0.8)
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Replan Time (ms)')
    ax1.set_title('Per-Step Replanning Time')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Summary bar chart
    labels = ['D* Lite', 'A* Full Replan']
    means = [np.mean(d_times), np.mean(a_times)]
    maxes = [np.max(d_times), np.max(a_times)]
    colors = ['#2196F3', '#E91E63']

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax2.bar(x - width/2, means, width, label='Mean', color=colors, alpha=0.7)
    bars2 = ax2.bar(x + width/2, maxes, width, label='Max', color=colors, alpha=1.0)

    ax2.set_ylabel('Replan Time (ms)')
    ax2.set_title('Average vs Max Replanning Time')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "replan_time_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved replan time comparison chart")


def print_summary(sim_results):
    """Print simulation summary."""
    print(f"\n{'='*60}")
    print(f"{'DYNAMIC REPLANNING SIMULATION SUMMARY':^60}")
    print(f"{'='*60}")

    d_times = sim_results["dstar_replan_times"]
    a_times = sim_results["astar_replan_times"]

    print(f"\nD* Lite:")
    print(f"  Steps taken: {sim_results['dstar_steps']}")
    print(f"  Reached goal: {sim_results['dstar_success']}")
    print(f"  Mean replan time: {np.mean(d_times):.2f} ms")
    print(f"  Max replan time: {np.max(d_times):.2f} ms")
    print(f"  Total replan time: {np.sum(d_times):.2f} ms")

    print(f"\nA* Full Replan:")
    print(f"  Steps taken: {sim_results['astar_steps']}")
    print(f"  Reached goal: {sim_results['astar_success']}")
    print(f"  Mean replan time: {np.mean(a_times):.2f} ms")
    print(f"  Max replan time: {np.max(a_times):.2f} ms")
    print(f"  Total replan time: {np.sum(a_times):.2f} ms")

    if np.mean(a_times) > 0:
        speedup = np.mean(a_times) / max(np.mean(d_times), 0.001)
        print(f"\nD* Lite speedup: {speedup:.1f}x faster on average")


if __name__ == "__main__":
    print("Running dynamic replanning simulation...")

    sim = ReplanningSimulator(
        grid_size=20,
        obstacle_density=0.15,
        seed=42,
        num_agents=3,
        agent_path_length=8,
        max_steps=60
    )

    results = sim.run()
    print_summary(results)
    animate_comparison(results, grid_size=20)
    plot_replan_times(results)

    print("\nDone!")
