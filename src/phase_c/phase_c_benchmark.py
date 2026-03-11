import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase_a"))
import numpy as np
import matplotlib.pyplot as plt
import csv
from grid import Grid
from dijkstra import Dijkstra
from astar import AStar
from rrt_connect_kd import RRTConnectKD
from weighted_grid import WeightedGrid
from neu_campus_processor import process_neu_campus, save_terrain_visualization, save_occupancy_image
from map_loader import ProceduralMapGenerator


def compute_smoothness(path):
    if path is None or len(path) < 3:
        return 0.0
    angles = []
    for i in range(1, len(path) - 1):
        v1 = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        v2 = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        m1 = np.sqrt(v1[0]**2+v1[1]**2)
        m2 = np.sqrt(v2[0]**2+v2[1]**2)
        if m1 == 0 or m2 == 0:
            continue
        angles.append(np.degrees(np.arccos(np.clip(dot/(m1*m2), -1, 1))))
    return np.mean(angles) if angles else 0.0


def gen_pairs(grid, n=10, seed=42):
    rng = np.random.RandomState(seed)
    free = [(r, c) for r in range(grid.height) for c in range(grid.width) if grid.is_free(r, c)]
    pairs = []
    if len(free) < 2:
        return pairs
    corners = [(2,2), (2,grid.width-3), (grid.height-3,2), (grid.height-3,grid.width-3)]
    fc = [c for c in corners if grid.is_free(c[0],c[1])]
    if len(fc) >= 2:
        pairs.append((fc[0], fc[-1]))
    att = 0
    while len(pairs) < n and att < 1000:
        i1, i2 = rng.choice(len(free), 2, replace=False)
        s, g = free[i1], free[i2]
        if abs(s[0]-g[0]) + abs(s[1]-g[1]) > grid.width // 4:
            pairs.append((s, g))
        att += 1
    return pairs[:n]


def run_one(algo, grid, start, goal):
    try:
        if algo == "Dijkstra":
            r = Dijkstra(grid, 4).search(start, goal)
        elif algo == "A*":
            r = AStar(grid, 4, "euclidean").search(start, goal)
        elif algo == "RRT-Connect":
            np.random.seed(hash((start, goal)) % 2**31)
            r = RRTConnectKD(grid, step_size=8, max_iterations=8000,
                              connect_threshold=4.0).search(start, goal)
        else:
            return None
        return {
            "success": r["path"] is not None,
            "time_ms": r["planning_time_ms"],
            "cost": r["cost"],
            "length": len(r["path"]) if r["path"] else None,
            "nodes": r["nodes_explored"],
            "memory_mb": r["memory_mb"],
            "smoothness": compute_smoothness(r["path"]),
            "path": r["path"],
        }
    except:
        return {"success": False, "time_ms": None, "cost": None,
                "length": None, "nodes": None, "memory_mb": None,
                "smoothness": None, "path": None}


def viz_paths_on_map(grid, results, map_name, save_dir):
    """Visualize algorithm paths on a map. Supports WeightedGrid terrain coloring."""
    algos = ["Dijkstra", "A*", "RRT-Connect"]
    colors = {"Dijkstra": "#2196F3", "A*": "#4CAF50", "RRT-Connect": "#FF9800"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Path Planning on {map_name}', fontsize=14, fontweight='bold')

    for idx, algo in enumerate(algos):
        ax = axes[idx]
        h, w = grid.height, grid.width
        display = np.ones((h, w, 3))

        # Terrain coloring if WeightedGrid
        if hasattr(grid, 'terrain'):
            for r in range(h):
                for c in range(w):
                    t = grid.terrain[r, c]
                    if t == 1:
                        display[r, c] = [0.15, 0.15, 0.15]
                    elif t == 2:
                        display[r, c] = [0.56, 0.93, 0.56]
                    elif t == 3:
                        display[r, c] = [0.75, 0.75, 0.75]
        else:
            for r in range(h):
                for c in range(w):
                    if grid.grid[r, c] == 1:
                        display[r, c] = [0.15, 0.15, 0.15]

        ax.imshow(display, interpolation='nearest')

        r = results.get(algo)
        if r and r["path"]:
            path = r["path"]
            if algo == "RRT-Connect":
                for i in range(len(path)-1):
                    ax.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]],
                            color=colors[algo], linewidth=2)
            else:
                ax.plot([c for _, c in path], [r for r, _ in path],
                        color=colors[algo], linewidth=2)
            ax.plot(path[0][1], path[0][0], 'o', color='#2ECC40', markersize=8, zorder=5)
            ax.plot(path[-1][1], path[-1][0], 'o', color='#E74C3C', markersize=8, zorder=5)
            sub = f"Cost: {r['cost']:.1f} | Time: {r['time_ms']:.1f}ms | Smooth: {r['smoothness']:.1f}°"
        else:
            sub = "No path found"

        ax.set_title(f"{algo}\n{sub}", fontsize=10)
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    plt.tight_layout()
    fname = map_name.lower().replace(' ', '_')
    plt.savefig(os.path.join(save_dir, f"{fname}_paths.png"), dpi=150, bbox_inches='tight')
    plt.close()


def run_phase_c():
    os.makedirs("visualizations/phase_c", exist_ok=True)
    os.makedirs("maps", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    algos = ["Dijkstra", "A*", "RRT-Connect"]
    maps = {}

    # NEU Campus (weighted)
    for p in ["maps/neu_campus.jpg", "maps/Campus_Map_2d.jpg", "maps/neu_campus.png"]:
        if os.path.exists(p):
            maps["NEU Campus"] = process_neu_campus(p, target_size=150)
            save_terrain_visualization(maps["NEU Campus"], "maps/neu_campus_terrain.png")
            save_occupancy_image(maps["NEU Campus"], "maps/neu_campus_occupancy.png")
            print(f"Loaded NEU campus from {p}")
            stats = maps["NEU Campus"].get_terrain_stats()
            print(f"  Paths: {stats['path_pct']:.1f}% | Buildings: {stats['impassable_pct']:.1f}% | "
                  f"Grass: {stats['grass_pct']:.1f}% | Roads: {stats['road_pct']:.1f}%")
            break

    if "NEU Campus" not in maps:
        print("WARNING: Campus map not found. Place image in maps/")

    # Procedural maps (standard grids, not weighted)
    maps["Office Building"] = ProceduralMapGenerator.office_building(size=150, seed=42)
    maps["Warehouse"] = ProceduralMapGenerator.warehouse(size=150, seed=42)
    maps["Hospital"] = ProceduralMapGenerator.hospital(size=150, seed=42)

    for name, g in maps.items():
        obs = np.sum(g.grid == 1) / (g.width * g.height) * 100
        print(f"  {name}: {g.width}x{g.height}, obstacles={obs:.1f}%")

    all_results = []
    n_pairs = 10

    for map_name, grid in maps.items():
        print(f"\n{'='*60}")
        print(f"Map: {map_name}")
        print(f"{'='*60}")

        pairs = gen_pairs(grid, n_pairs, seed=42)
        print(f"  {len(pairs)} start-goal pairs")

        # Visualize first pair
        if pairs:
            vr = {}
            for a in algos:
                vr[a] = run_one(a, grid, pairs[0][0], pairs[0][1])
            viz_paths_on_map(grid, vr, map_name, "visualizations/phase_c")

        for a in algos:
            print(f"  {a}...", end=" ")
            ok = 0
            for i, (s, g) in enumerate(pairs):
                r = run_one(a, grid, s, g)
                if r["success"]:
                    ok += 1
                all_results.append({
                    "map": map_name, "algorithm": a, "pair": i,
                    "success": r["success"], "time_ms": r["time_ms"],
                    "cost": r["cost"], "length": r["length"],
                    "nodes": r["nodes"], "memory_mb": r["memory_mb"],
                    "smoothness": r["smoothness"],
                })
            print(f"{ok}/{len(pairs)} ({ok/len(pairs)*100:.0f}%)")

    # Save CSV
    with open("results/phase_c_results.csv", 'w', newline='') as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)
    print("\nSaved results/phase_c_results.csv")

    # Dashboard
    gen_dashboard(all_results, maps)
    return all_results


def gen_dashboard(all_results, maps):
    algos = ["Dijkstra", "A*", "RRT-Connect"]
    map_names = list(maps.keys())
    colors = {"Dijkstra": "#2196F3", "A*": "#4CAF50", "RRT-Connect": "#FF9800"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Phase C — NEU Campus & Structured Map Benchmark',
                 fontsize=16, fontweight='bold', y=1.02)

    x = np.arange(len(map_names))
    width = 0.25
    offsets = [-1, 0, 1]

    metrics = [
        ("Success Rate (%)", "success_rate"),
        ("Planning Time (ms)", "time_ms"),
        ("Path Cost", "cost"),
        ("Nodes Explored", "nodes"),
        ("Memory (MB)", "memory_mb"),
        ("Smoothness (deg)", "smoothness"),
    ]

    for idx, (title, metric) in enumerate(metrics):
        ax = axes[idx//3][idx%3]
        for i, a in enumerate(algos):
            vals = []
            for mn in map_names:
                runs = [r for r in all_results if r["map"]==mn and r["algorithm"]==a]
                sr = [r for r in runs if r["success"]]
                if metric == "success_rate":
                    vals.append(len(sr)/len(runs)*100 if runs else 0)
                elif sr:
                    v = [r[metric] for r in sr if r[metric] is not None]
                    vals.append(np.mean(v) if v else 0)
                else:
                    vals.append(0)
            ax.bar(x+offsets[i]*width, vals, width, label=a, color=colors[a], edgecolor='white')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(' ', '\n') for n in map_names], fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    axes[0][0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("visualizations/phase_c/phase_c_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved Phase C dashboard")


if __name__ == "__main__":
    run_phase_c()
