import numpy as np
import matplotlib.pyplot as plt
import csv
import os


def load_results(filepath="results/benchmark_results.csv"):
    """Load benchmark results from CSV."""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["density"] = float(row["density"])
            row["trial"] = int(row["trial"])
            row["success"] = row["success"] == "True"
            row["planning_time_ms"] = float(row["planning_time_ms"]) if row["planning_time_ms"] else None
            row["path_length"] = int(row["path_length"]) if row["path_length"] else None
            row["path_cost"] = float(row["path_cost"]) if row["path_cost"] else None
            row["nodes_explored"] = int(row["nodes_explored"]) if row["nodes_explored"] else None
            row["memory_mb"] = float(row["memory_mb"]) if row["memory_mb"] else None
            results.append(row)
    return results


def get_stats(results, algo, density, metric):
    """Get mean and std for a metric, only from successful runs."""
    values = [r[metric] for r in results
              if r["algorithm"] == algo
              and r["density"] == density
              and r["success"]
              and r[metric] is not None]
    if values:
        return np.mean(values), np.std(values)
    return None, None


def get_success_rate(results, algo, density):
    """Get success rate percentage."""
    all_runs = [r for r in results if r["algorithm"] == algo and r["density"] == density]
    if not all_runs:
        return 0.0
    successes = sum(1 for r in all_runs if r["success"])
    return successes / len(all_runs) * 100


def plot_benchmark(results):
    """Generate all benchmark charts."""
    algorithms = ["Dijkstra", "A*", "RRT-Connect", "RRT*"]
    densities = sorted(set(r["density"] for r in results))
    colors = {
        "Dijkstra": "#2196F3",
        "A*": "#4CAF50",
        "RRT-Connect": "#FF9800",
        "RRT*": "#E91E63"
    }

    density_labels = [f"{int(d*100)}%" for d in densities]
    x = np.arange(len(densities))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    os.makedirs("visualizations", exist_ok=True)

    # ---- 1. Success Rate ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, algo in enumerate(algorithms):
        rates = [get_success_rate(results, algo, d) for d in densities]
        bars = ax.bar(x + offsets[i] * width, rates, width,
                      label=algo, color=colors[algo], edgecolor='white')
        for bar, val in zip(bars, rates):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Obstacle Density', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Navigation Success Rate by Algorithm and Obstacle Density', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(density_labels)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/bench_success_rate.png", dpi=150)
    plt.close()
    print("Saved bench_success_rate.png")

    # ---- 2. Planning Time ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, algo in enumerate(algorithms):
        means = []
        stds = []
        for d in densities:
            m, s = get_stats(results, algo, d, "planning_time_ms")
            means.append(m if m else 0)
            stds.append(s if s else 0)

        bars = ax.bar(x + offsets[i] * width, means, width, yerr=stds,
                      label=algo, color=colors[algo], edgecolor='white',
                      capsize=3, error_kw={'linewidth': 0.8})

    ax.set_xlabel('Obstacle Density', fontsize=12)
    ax.set_ylabel('Planning Time (ms)', fontsize=12)
    ax.set_title('Planning Time by Algorithm and Obstacle Density', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(density_labels)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/bench_planning_time.png", dpi=150)
    plt.close()
    print("Saved bench_planning_time.png")

    # ---- 3. Path Cost ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, algo in enumerate(algorithms):
        means = []
        stds = []
        for d in densities:
            m, s = get_stats(results, algo, d, "path_cost")
            means.append(m if m else 0)
            stds.append(s if s else 0)

        bars = ax.bar(x + offsets[i] * width, means, width, yerr=stds,
                      label=algo, color=colors[algo], edgecolor='white',
                      capsize=3, error_kw={'linewidth': 0.8})

    ax.set_xlabel('Obstacle Density', fontsize=12)
    ax.set_ylabel('Path Cost', fontsize=12)
    ax.set_title('Path Cost by Algorithm and Obstacle Density', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(density_labels)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/bench_path_cost.png", dpi=150)
    plt.close()
    print("Saved bench_path_cost.png")

    # ---- 4. Nodes Explored ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, algo in enumerate(algorithms):
        means = []
        stds = []
        for d in densities:
            m, s = get_stats(results, algo, d, "nodes_explored")
            means.append(m if m else 0)
            stds.append(s if s else 0)

        bars = ax.bar(x + offsets[i] * width, means, width, yerr=stds,
                      label=algo, color=colors[algo], edgecolor='white',
                      capsize=3, error_kw={'linewidth': 0.8})

    ax.set_xlabel('Obstacle Density', fontsize=12)
    ax.set_ylabel('Nodes Explored', fontsize=12)
    ax.set_title('Nodes Explored by Algorithm and Obstacle Density', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(density_labels)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/bench_nodes_explored.png", dpi=150)
    plt.close()
    print("Saved bench_nodes_explored.png")

    # ---- 5. Memory Usage ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, algo in enumerate(algorithms):
        means = []
        stds = []
        for d in densities:
            m, s = get_stats(results, algo, d, "memory_mb")
            means.append(m if m else 0)
            stds.append(s if s else 0)

        bars = ax.bar(x + offsets[i] * width, means, width, yerr=stds,
                      label=algo, color=colors[algo], edgecolor='white',
                      capsize=3, error_kw={'linewidth': 0.8})

    ax.set_xlabel('Obstacle Density', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('Peak Memory Usage by Algorithm and Obstacle Density', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(density_labels)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/bench_memory.png", dpi=150)
    plt.close()
    print("Saved bench_memory.png")

    # ---- 6. Path Optimality Ratio (vs Dijkstra) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, algo in enumerate(algorithms):
        ratios = []
        for d in densities:
            # Get Dijkstra costs per trial
            dijkstra_costs = {r["trial"]: r["path_cost"] for r in results
                              if r["algorithm"] == "Dijkstra"
                              and r["density"] == d
                              and r["success"]
                              and r["path_cost"] is not None}

            algo_costs = {r["trial"]: r["path_cost"] for r in results
                          if r["algorithm"] == algo
                          and r["density"] == d
                          and r["success"]
                          and r["path_cost"] is not None}

            # Compute ratio for trials where both succeeded
            trial_ratios = []
            for trial in dijkstra_costs:
                if trial in algo_costs and dijkstra_costs[trial] > 0:
                    trial_ratios.append(algo_costs[trial] / dijkstra_costs[trial])

            ratios.append(np.mean(trial_ratios) if trial_ratios else 0)

        bars = ax.bar(x + offsets[i] * width, ratios, width,
                      label=algo, color=colors[algo], edgecolor='white')
        for bar, val in zip(bars, ratios):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Optimal (1.0)')
    ax.set_xlabel('Obstacle Density', fontsize=12)
    ax.set_ylabel('Path Cost Ratio (vs Dijkstra Optimal)', fontsize=12)
    ax.set_title('Path Optimality Ratio by Algorithm', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(density_labels)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/bench_optimality_ratio.png", dpi=150)
    plt.close()
    print("Saved bench_optimality_ratio.png")

    # ---- 7. Combined Dashboard ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Path Planning Benchmark Dashboard — 50×50 Grid, 50 Trials',
                 fontsize=16, fontweight='bold', y=1.02)

    metrics = [
        ("Success Rate (%)", "success_rate"),
        ("Planning Time (ms)", "planning_time_ms"),
        ("Path Cost", "path_cost"),
        ("Nodes Explored", "nodes_explored"),
        ("Memory (MB)", "memory_mb"),
        ("Optimality Ratio", "optimality"),
    ]

    for idx, (title, metric) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]

        for i, algo in enumerate(algorithms):
            values = []
            for d in densities:
                if metric == "success_rate":
                    values.append(get_success_rate(results, algo, d))
                elif metric == "optimality":
                    dijkstra_costs = {r["trial"]: r["path_cost"] for r in results
                                      if r["algorithm"] == "Dijkstra" and r["density"] == d
                                      and r["success"] and r["path_cost"] is not None}
                    algo_costs = {r["trial"]: r["path_cost"] for r in results
                                  if r["algorithm"] == algo and r["density"] == d
                                  and r["success"] and r["path_cost"] is not None}
                    trial_ratios = [algo_costs[t] / dijkstra_costs[t]
                                    for t in dijkstra_costs
                                    if t in algo_costs and dijkstra_costs[t] > 0]
                    values.append(np.mean(trial_ratios) if trial_ratios else 0)
                else:
                    m, _ = get_stats(results, algo, d, metric)
                    values.append(m if m else 0)

            ax.bar(x + offsets[i] * width, values, width,
                   label=algo, color=colors[algo], edgecolor='white')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(density_labels)
        ax.grid(axis='y', alpha=0.3)

        if metric == "optimality":
            ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    axes[0][0].legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig("visualizations/bench_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved bench_dashboard.png")

    print("\nAll benchmark visualizations saved to visualizations/")


if __name__ == "__main__":
    results = load_results("results/benchmark_results.csv")
    plot_benchmark(results)
