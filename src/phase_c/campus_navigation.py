import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase_a"))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import csv
from dijkstra import Dijkstra
from astar import AStar
from rrt_connect_kd import RRTConnectKD
from weighted_grid import WeightedGrid
from neu_campus_processor import process_neu_campus


# Building locations on the 150x150 grid (approximate centers)
# Format: "Name": (row, col)
BUILDINGS = {
    "ISEC": (240, 136),
    "Snell Library": (194, 136),
    "Curry Student Center": (176, 190),
    "Egan Research": (200, 100),
    "Dodge Hall": (144, 190),
    "Hayden Hall": (160, 150),
    "Ell Hall": (156, 176),
    "Mugar Life Sciences": (164, 200),
    "Richards Hall": (136, 170),
    "West Village H": (110, 36),
    "Shillman Hall": (184, 94),
    "Snell Engineering": (194, 116),
    "Forsyth Building": (164, 104),
    "Churchill Hall": (164, 124),
    "Ryder Hall": (224, 44),
    "Behrakis": (192, 30),
    "Marino Center": (96, 104),
    "Cabot Center": (130, 140),
    "Ruggles Station": (210, 80),
    "Krentzman Quad": (136, 104),
}

# Predefined routes (interesting paths across campus)
ROUTES = [
    ("ISEC", "Snell Library"),
    ("West Village H", "Dodge Hall"),
    ("Ryder Hall", "Curry Student Center"),
    ("Behrakis", "ISEC"),
    ("Marino Center", "Ell Hall"),
    ("Ruggles Station", "Richards Hall"),
    ("Snell Engineering", "Mugar Life Sciences"),
    ("Krentzman Quad", "ISEC"),
    ("Shillman Hall", "Curry Student Center"),
    ("West Village H", "ISEC"),
]


def find_nearest_free(grid, row, col, max_search=15):
    """Find nearest free cell to a given position."""
    if grid.is_free(row, col):
        return (row, col)
    for r in range(1, max_search):
        for dr in range(-r, r+1):
            for dc in range(-r, r+1):
                nr, nc = row+dr, col+dc
                if 0 <= nr < grid.height and 0 <= nc < grid.width and grid.is_free(nr, nc):
                    return (nr, nc)
    return None


def run_algo(name, grid, start, goal):
    """Run one algorithm, return result."""
    try:
        if name == "Dijkstra":
            r = Dijkstra(grid, 4).search(start, goal)
        elif name == "A*":
            r = AStar(grid, 4, "euclidean").search(start, goal)
        elif name == "RRT-Connect":
            np.random.seed(hash((start, goal)) % 2**31)
            r = RRTConnectKD(grid, step_size=6, max_iterations=8000,
                              connect_threshold=3.0).search(start, goal)
        else:
            return None
        return {
            "success": r["path"] is not None,
            "time_ms": r["planning_time_ms"],
            "cost": r["cost"],
            "nodes": r["nodes_explored"],
            "path": r["path"],
        }
    except:
        return {"success": False, "time_ms": None, "cost": None,
                "nodes": None, "path": None}


def draw_path_on_map(ax, path, color, linewidth=2.5, label=None, is_rrt=False):
    """Draw a path on a map axis."""
    if not path:
        return
    if is_rrt:
        for i in range(len(path)-1):
            ax.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]],
                    color=color, linewidth=linewidth,
                    label=label if i == 0 else None)
    else:
        cols = [c for _, c in path]
        rows = [r for r, _ in path]
        ax.plot(cols, rows, color=color, linewidth=linewidth, label=label)


def visualize_single_route(grid, original_img, start_name, goal_name,
                            start_pos, goal_pos, results, save_dir):
    """Show one route with all 3 algorithms on the actual map."""
    fig, ax = plt.subplots(figsize=(10, 12))

    # Show actual campus map as background
    h, w = grid.height, grid.width
    img_resized = cv2.resize(original_img, (w, h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb, alpha=0.85)

    # Draw paths
    algo_colors = {"Dijkstra": "#FF4444", "A*": "#44FF44", "RRT-Connect": "#4444FF"}
    algo_lw = {"Dijkstra": 3, "A*": 2.5, "RRT-Connect": 2}

    for algo_name, r in results.items():
        if r["success"]:
            is_rrt = algo_name == "RRT-Connect"
            draw_path_on_map(ax, r["path"], algo_colors[algo_name],
                              algo_lw[algo_name], label=algo_name, is_rrt=is_rrt)

    # Mark start and goal
    ax.plot(start_pos[1], start_pos[0], 'o', color='#00FF00', markersize=14,
            markeredgecolor='white', markeredgewidth=2, zorder=10)
    ax.plot(goal_pos[1], goal_pos[0], '*', color='#FF0000', markersize=18,
            markeredgecolor='white', markeredgewidth=2, zorder=10)

    # Labels
    ax.annotate(start_name, (start_pos[1], start_pos[0]-5),
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='green', alpha=0.8))
    ax.annotate(goal_name, (goal_pos[1], goal_pos[0]-5),
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='red', alpha=0.8))

    # Metrics box
    metrics_text = ""
    for algo_name, r in results.items():
        if r["success"]:
            metrics_text += f"{algo_name}: cost={r['cost']:.1f}, time={r['time_ms']:.1f}ms\n"
        else:
            metrics_text += f"{algo_name}: No path found\n"

    ax.text(0.02, 0.02, metrics_text.strip(), transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_title(f'{start_name}  →  {goal_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    fname = f"{start_name}_{goal_name}".replace(' ', '_').lower()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"route_{fname}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_all_routes_overview(grid, original_img, all_route_results, save_dir):
    """Show all routes on one map with Dijkstra paths only."""
    fig, ax = plt.subplots(figsize=(12, 14))

    h, w = grid.height, grid.width
    img_resized = cv2.resize(original_img, (w, h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb, alpha=0.8)

    route_colors = plt.cm.tab10(np.linspace(0, 1, len(all_route_results)))

    for i, (route_name, results) in enumerate(all_route_results.items()):
        dijk = results.get("Dijkstra")
        if dijk and dijk["success"]:
            path = dijk["path"]
            cols = [c for _, c in path]
            rows = [r for r, _ in path]
            ax.plot(cols, rows, color=route_colors[i], linewidth=2, label=route_name, alpha=0.8)

    # Mark all buildings
    for name, (r, c) in BUILDINGS.items():
        ax.plot(c, r, 's', color='white', markersize=4, markeredgecolor='black', markeredgewidth=0.5)

    ax.set_title('NEU Campus — All Dijkstra Routes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_routes_overview.png"), dpi=150, bbox_inches='tight')
    plt.close()


def run_campus_benchmark(map_path):
    """Run full campus navigation benchmark."""
    os.makedirs("visualizations/phase_c", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Processing NEU campus map...")
    grid = process_neu_campus(map_path, target_size=300)
    stats = grid.get_terrain_stats()
    print(f"  Grid: {grid.width}x{grid.height}")
    print(f"  Paths: {stats['path_pct']:.1f}% | Buildings: {stats['impassable_pct']:.1f}% | "
          f"Grass: {stats['grass_pct']:.1f}% | Roads: {stats['road_pct']:.1f}%")

    # Load original image for overlay
    original_img = cv2.imread(map_path)

    algos = ["Dijkstra", "A*", "RRT-Connect"]
    all_results = []
    all_route_results = {}

    print(f"\nRunning {len(ROUTES)} building-to-building routes...")

    for start_name, goal_name in ROUTES:
        start_raw = BUILDINGS.get(start_name)
        goal_raw = BUILDINGS.get(goal_name)

        if not start_raw or not goal_raw:
            print(f"  SKIP: {start_name} or {goal_name} not in building list")
            continue

        start = find_nearest_free(grid, start_raw[0], start_raw[1])
        goal = find_nearest_free(grid, goal_raw[0], goal_raw[1])

        if not start or not goal:
            print(f"  SKIP: No free cell near {start_name} or {goal_name}")
            continue

        route_name = f"{start_name} → {goal_name}"
        print(f"\n  {route_name}")

        route_results = {}
        for algo in algos:
            r = run_algo(algo, grid, start, goal)
            route_results[algo] = r

            status = f"cost={r['cost']:.1f}, time={r['time_ms']:.1f}ms" if r["success"] else "FAILED"
            print(f"    {algo}: {status}")

            all_results.append({
                "route": route_name, "algorithm": algo,
                "success": r["success"], "time_ms": r["time_ms"],
                "cost": r["cost"], "nodes": r["nodes"],
            })

        # Visualize this route
        visualize_single_route(grid, original_img, start_name, goal_name,
                                start, goal, route_results, "visualizations/phase_c")
        all_route_results[route_name] = route_results

    # Overview map
    visualize_all_routes_overview(grid, original_img, all_route_results, "visualizations/phase_c")

    # Save CSV
    with open("results/phase_c_campus_results.csv", 'w', newline='', encoding='utf-8') as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)

    # Summary dashboard
    gen_campus_dashboard(all_results)

    print(f"\n{'='*60}")
    print("Phase C complete!")
    print(f"  {len(ROUTES)} routes benchmarked")
    print(f"  Results: results/phase_c_campus_results.csv")
    print(f"  Visuals: visualizations/phase_c/")


def gen_campus_dashboard(all_results):
    """Generate comparison dashboard for campus routes."""
    algos = ["Dijkstra", "A*", "RRT-Connect"]
    colors = {"Dijkstra": "#2196F3", "A*": "#4CAF50", "RRT-Connect": "#FF9800"}

    # Aggregate metrics
    algo_metrics = {}
    for a in algos:
        runs = [r for r in all_results if r["algorithm"] == a and r["success"]]
        algo_metrics[a] = {
            "success_rate": len(runs) / len([r for r in all_results if r["algorithm"] == a]) * 100,
            "avg_time": np.mean([r["time_ms"] for r in runs]) if runs else 0,
            "avg_cost": np.mean([r["cost"] for r in runs]) if runs else 0,
            "avg_nodes": np.mean([r["nodes"] for r in runs]) if runs else 0,
        }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('NEU Campus Navigation — Algorithm Comparison',
                 fontsize=14, fontweight='bold')

    metrics = [
        ("Success Rate (%)", "success_rate"),
        ("Avg Planning Time (ms)", "avg_time"),
        ("Avg Path Cost", "avg_cost"),
        ("Avg Nodes Explored", "avg_nodes"),
    ]

    for idx, (title, key) in enumerate(metrics):
        ax = axes[idx]
        vals = [algo_metrics[a][key] for a in algos]
        bars = ax.bar(algos, vals, color=[colors[a] for a in algos], edgecolor='white')

        for bar, val in zip(bars, vals):
            fmt = f'{val:.0f}' if key in ['success_rate', 'avg_nodes'] else f'{val:.1f}'
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    fmt, ha='center', fontsize=10, fontweight='bold')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("visualizations/phase_c/campus_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved campus dashboard")


if __name__ == "__main__":
    # Find campus map
    candidates = ["maps/Campus_Map_2d.jpg", "maps/Campus_Map_2d.png",
                   "maps/neu_campus.jpg", "maps/neu_campus.png"]
    map_path = None
    for p in candidates:
        if os.path.exists(p):
            map_path = p
            break

    if map_path:
        run_campus_benchmark(map_path)
    else:
        print("Campus map not found!")
        print("Place your campus map image in maps/ as one of:")
        for p in candidates:
            print(f"  {p}")
