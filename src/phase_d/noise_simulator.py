import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase_a"))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import time
from grid import Grid
from dijkstra import Dijkstra
from astar import AStar
from rrt_connect_kd import RRTConnectKD
from mdp_solver import StochasticActionModel, MDP, ValueIteration


class NoiseSimulator:
    """
    Execute any planner's path under stochastic noise.
    At each step, the intended action is sampled through the noise model
    instead of executed deterministically.
    """

    def __init__(self, grid, goal, noise_model, max_steps=300):
        self.grid = grid
        self.goal = goal
        self.noise = noise_model
        self.max_steps = max_steps

    def _intended_action(self, current, next_pos):
        """Determine intended action from current to next position."""
        dr = next_pos[0] - current[0]
        dc = next_pos[1] - current[1]

        # Map to closest cardinal direction
        if abs(dr) >= abs(dc):
            return "down" if dr > 0 else "up"
        else:
            return "right" if dc > 0 else "left"

    def _apply_noise(self, pos, intended_action, rng):
        """Apply noise to intended action, return new position."""
        dr, dc = self.noise.sample_action(intended_action, rng)
        nr, nc = pos[0] + dr, pos[1] + dc

        # Boundary/obstacle check — stay in place if blocked
        if not (0 <= nr < self.grid.height and 0 <= nc < self.grid.width):
            return pos
        if self.grid.grid[nr, nc] == 1:
            return pos

        return (nr, nc)

    def run_deterministic_planner(self, path, rng_seed=None):
        """
        Execute a pre-planned path under noise — OPEN LOOP.
        The planner follows its fixed action sequence regardless of actual position.
        It does NOT know where it is — no replanning, no re-targeting.

        This models a robot executing pre-computed motor commands without localization.

        Returns:
            dict with success, steps, trajectory, total_drift.
        """
        if path is None or len(path) < 2:
            return {"success": False, "steps": 0, "trajectory": [], "total_drift": 0}

        rng = np.random.RandomState(rng_seed)
        pos = path[0]
        trajectory = [pos]
        total_drift = 0

        # Pre-compute the action sequence from the path
        action_sequence = []
        for i in range(len(path) - 1):
            action_sequence.append(self._intended_action(path[i], path[i + 1]))

        # Execute actions blindly in order
        for step_idx in range(len(action_sequence)):
            if pos == self.goal:
                return {"success": True, "steps": step_idx, "trajectory": trajectory,
                        "total_drift": total_drift}

            intended = action_sequence[step_idx]
            new_pos = self._apply_noise(pos, intended, rng)

            if new_pos != pos:
                intended_dr, intended_dc = StochasticActionModel.ACTIONS[intended]
                intended_pos = (pos[0] + intended_dr, pos[1] + intended_dc)
                if new_pos != intended_pos:
                    total_drift += 1

            pos = new_pos
            trajectory.append(pos)

        # After exhausting action sequence, check if at goal
        if pos == self.goal:
            return {"success": True, "steps": len(action_sequence), "trajectory": trajectory,
                    "total_drift": total_drift}

        # Extra steps: robot is lost, wander toward goal blindly for remaining budget
        remaining = self.max_steps - len(action_sequence)
        for step in range(remaining):
            if pos == self.goal:
                return {"success": True, "steps": len(action_sequence) + step,
                        "trajectory": trajectory, "total_drift": total_drift}

            # Try to move toward goal (but still open-loop — just aim at goal)
            intended = self._intended_action(pos, self.goal)
            new_pos = self._apply_noise(pos, intended, rng)

            if new_pos != pos:
                intended_dr, intended_dc = StochasticActionModel.ACTIONS[intended]
                intended_pos = (pos[0] + intended_dr, pos[1] + intended_dc)
                if new_pos != intended_pos:
                    total_drift += 1

            pos = new_pos
            trajectory.append(pos)

        return {"success": pos == self.goal, "steps": self.max_steps,
                "trajectory": trajectory, "total_drift": total_drift}

    def run_mdp_policy(self, policy_func, rng_seed=None):
        """
        Execute MDP policy under noise.
        Policy reacts to current state — inherently handles drift.

        Returns:
            dict with success, steps, trajectory, total_drift.
        """
        rng = np.random.RandomState(rng_seed)
        pos = (0, 0)  # Start
        trajectory = [pos]
        total_drift = 0

        for step in range(self.max_steps):
            if pos == self.goal:
                return {"success": True, "steps": step, "trajectory": trajectory,
                        "total_drift": total_drift}

            action = policy_func(pos)
            if action is None:
                return {"success": False, "steps": step, "trajectory": trajectory,
                        "total_drift": total_drift}

            new_pos = self._apply_noise(pos, action, rng)

            # Track drift
            intended_dr, intended_dc = StochasticActionModel.ACTIONS[action]
            intended_pos = (pos[0] + intended_dr, pos[1] + intended_dc)
            if new_pos != intended_pos and new_pos != pos:
                total_drift += 1

            pos = new_pos
            trajectory.append(pos)

        return {"success": False, "steps": self.max_steps, "trajectory": trajectory,
                "total_drift": total_drift}


def run_full_comparison():
    """Run all planners at 3 noise levels, 200 trials each."""
    os.makedirs("visualizations/phase_d", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    grid_size = 20
    density = 0.15
    seed = 42
    trials = 200
    start = (0, 0)
    goal = (grid_size - 1, grid_size - 1)

    noise_levels = {
        "Low (95/2.5/2.5)": (0.95, 0.025),
        "Medium (80/10/10)": (0.80, 0.10),
        "High (60/20/20)": (0.60, 0.20),
    }

    g = Grid(grid_size, grid_size, obstacle_density=density, seed=seed)

    # Pre-compute deterministic paths
    print("Computing deterministic paths...")
    dijk_path = Dijkstra(g, 4).search(start, goal)["path"]
    astar_path = AStar(g, 4, "euclidean").search(start, goal)["path"]

    np.random.seed(seed)
    rrt_path = RRTConnectKD(g, step_size=3, max_iterations=5000,
                             connect_threshold=2.0).search(start, goal)["path"]

    # Convert RRT path to grid cells for noise simulation
    if rrt_path:
        rrt_grid_path = [(int(round(r)), int(round(c))) for r, c in rrt_path]
        # Ensure all points are valid
        rrt_grid_path = [(r, c) for r, c in rrt_grid_path
                          if 0 <= r < grid_size and 0 <= c < grid_size and g.is_free(r, c)]
    else:
        rrt_grid_path = None

    print(f"  Dijkstra path: {len(dijk_path) if dijk_path else 'None'} steps")
    print(f"  A* path: {len(astar_path) if astar_path else 'None'} steps")
    print(f"  RRT-Connect path: {len(rrt_grid_path) if rrt_grid_path else 'None'} steps")

    planners = {
        "Dijkstra": ("deterministic", dijk_path),
        "A*": ("deterministic", astar_path),
        "RRT-Connect": ("deterministic", rrt_grid_path),
        "MDP Policy": ("mdp", None),
    }

    all_results = []

    for noise_name, (fwd_prob, drift_prob) in noise_levels.items():
        print(f"\n{'='*60}")
        print(f"Noise Level: {noise_name}")
        print(f"{'='*60}")

        noise_model = StochasticActionModel(fwd_prob, drift_prob)

        # Solve MDP for this noise level
        print("  Solving MDP...")
        mdp = MDP(g, goal, noise_model, gamma=0.95)
        vi = ValueIteration(mdp, convergence_threshold=0.001)
        vi_result = vi.solve()
        print(f"  MDP solved in {vi_result['iterations']} iterations, "
              f"{vi_result['convergence_time_ms']:.0f}ms")

        sim = NoiseSimulator(g, goal, noise_model, max_steps=300)

        for planner_name, (ptype, path) in planners.items():
            print(f"\n  {planner_name}...", end=" ")
            successes = 0
            total_steps = []
            total_drifts = []

            for trial in range(trials):
                if ptype == "deterministic":
                    result = sim.run_deterministic_planner(path, rng_seed=trial)
                else:
                    result = sim.run_mdp_policy(vi.get_action, rng_seed=trial)

                if result["success"]:
                    successes += 1
                    total_steps.append(result["steps"])
                total_drifts.append(result["total_drift"])

                all_results.append({
                    "noise_level": noise_name,
                    "planner": planner_name,
                    "trial": trial,
                    "success": result["success"],
                    "steps": result["steps"],
                    "drift_events": result["total_drift"],
                })

            success_rate = successes / trials * 100
            avg_steps = np.mean(total_steps) if total_steps else 0
            avg_drift = np.mean(total_drifts)
            print(f"{success_rate:.0f}% success, avg steps: {avg_steps:.0f}, avg drift: {avg_drift:.1f}")

    # Save CSV
    with open("results/phase_d_results.csv", 'w', newline='', encoding='utf-8') as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)
    print("\nSaved results/phase_d_results.csv")

    # Generate visualizations
    generate_plots(all_results, noise_levels)
    generate_policy_viz(g, goal, noise_levels)

    return all_results


def generate_plots(all_results, noise_levels):
    """Generate comparison charts."""
    planners = ["Dijkstra", "A*", "RRT-Connect", "MDP Policy"]
    colors = {"Dijkstra": "#2196F3", "A*": "#4CAF50",
              "RRT-Connect": "#FF9800", "MDP Policy": "#E91E63"}
    noise_names = list(noise_levels.keys())

    # --- 1. Success Rate vs Noise Level ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for planner in planners:
        rates = []
        for nn in noise_names:
            runs = [r for r in all_results if r["planner"] == planner and r["noise_level"] == nn]
            rate = sum(1 for r in runs if r["success"]) / len(runs) * 100
            rates.append(rate)
        ax.plot(noise_names, rates, 'o-', color=colors[planner], linewidth=2.5,
                markersize=10, label=planner)

    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Goal-Reaching Success Rate vs Motion Noise', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig("visualizations/phase_d/success_vs_noise.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved success_vs_noise.png")

    # --- 2. Average Steps vs Noise ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for planner in planners:
        avg_steps = []
        for nn in noise_names:
            runs = [r for r in all_results
                    if r["planner"] == planner and r["noise_level"] == nn and r["success"]]
            avg_steps.append(np.mean([r["steps"] for r in runs]) if runs else 0)
        ax.plot(noise_names, avg_steps, 'o-', color=colors[planner], linewidth=2.5,
                markersize=10, label=planner)

    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Average Steps to Goal', fontsize=12)
    ax.set_title('Path Efficiency Under Noise', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/phase_d/steps_vs_noise.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved steps_vs_noise.png")

    # --- 3. Dashboard ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase D — Probabilistic Planning Under Motion Uncertainty',
                 fontsize=14, fontweight='bold')

    # Success rate bars
    ax = axes[0]
    x = np.arange(len(noise_names))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for i, p in enumerate(planners):
        rates = []
        for nn in noise_names:
            runs = [r for r in all_results if r["planner"] == p and r["noise_level"] == nn]
            rates.append(sum(1 for r in runs if r["success"]) / len(runs) * 100)
        ax.bar(x + offsets[i]*width, rates, width, label=p, color=colors[p], edgecolor='white')
    ax.set_title('Success Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_xlabel('Noise Level')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Avg steps bars
    ax = axes[1]
    for i, p in enumerate(planners):
        steps = []
        for nn in noise_names:
            runs = [r for r in all_results if r["planner"] == p and r["noise_level"] == nn and r["success"]]
            steps.append(np.mean([r["steps"] for r in runs]) if runs else 0)
        ax.bar(x + offsets[i]*width, steps, width, label=p, color=colors[p], edgecolor='white')
    ax.set_title('Avg Steps to Goal')
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_xlabel('Noise Level')
    ax.grid(axis='y', alpha=0.3)

    # Drift events
    ax = axes[2]
    for i, p in enumerate(planners):
        drifts = []
        for nn in noise_names:
            runs = [r for r in all_results if r["planner"] == p and r["noise_level"] == nn]
            drifts.append(np.mean([r["drift_events"] for r in runs]))
        ax.bar(x + offsets[i]*width, drifts, width, label=p, color=colors[p], edgecolor='white')
    ax.set_title('Avg Drift Events')
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_xlabel('Noise Level')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("visualizations/phase_d/phase_d_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved phase_d_dashboard.png")


def generate_policy_viz(grid, goal, noise_levels):
    """Generate policy vector field visualization for medium noise."""
    noise = StochasticActionModel(0.8, 0.1)
    mdp = MDP(grid, goal, noise, gamma=0.95)
    vi = ValueIteration(mdp, convergence_threshold=0.001)
    vi.solve()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Value function heatmap ---
    value_map = np.full((grid.height, grid.width), np.nan)
    for (r, c), v in vi.V.items():
        value_map[r, c] = v
    for r in range(grid.height):
        for c in range(grid.width):
            if grid.grid[r, c] == 1:
                value_map[r, c] = np.nan

    im = ax1.imshow(value_map, cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(im, ax=ax1, shrink=0.8)
    ax1.set_title('Value Function V*(s)', fontsize=12, fontweight='bold')
    ax1.plot(goal[1], goal[0], '*', color='gold', markersize=15, markeredgecolor='black', zorder=5)

    # --- Policy vector field ---
    display = np.ones((grid.height, grid.width, 3))
    for r in range(grid.height):
        for c in range(grid.width):
            if grid.grid[r, c] == 1:
                display[r, c] = [0.15, 0.15, 0.15]

    ax2.imshow(display, interpolation='nearest')

    arrow_map = {"up": (0, -0.4), "down": (0, 0.4), "left": (-0.4, 0), "right": (0.4, 0)}

    for (r, c), action in vi.policy.items():
        if (r, c) == goal:
            continue
        if action in arrow_map:
            dx, dy = arrow_map[action]
            ax2.arrow(c, r, dx, dy, head_width=0.25, head_length=0.15,
                      fc='#2196F3', ec='#1565C0', linewidth=0.5, alpha=0.8)

    ax2.plot(goal[1], goal[0], '*', color='gold', markersize=15, markeredgecolor='black', zorder=5)
    ax2.plot(0, 0, 'o', color='#2ECC40', markersize=10, markeredgecolor='white', zorder=5)
    ax2.set_title('Optimal Policy pi*(s) — Medium Noise', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig("visualizations/phase_d/policy_and_values.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved policy_and_values.png")


if __name__ == "__main__":
    print("Phase D — Probabilistic Planning Comparison")
    print("Running 4 planners x 3 noise levels x 200 trials = 2400 runs\n")
    run_full_comparison()
    print("\nPhase D complete!")
