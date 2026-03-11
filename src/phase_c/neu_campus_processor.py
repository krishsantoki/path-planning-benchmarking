import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import numpy as np
import cv2
from weighted_grid import WeightedGrid


def process_neu_campus(filepath, target_size=200):
    """
    Process NEU campus map into a WeightedGrid with terrain types.

    Terrain classification:
        Buildings (dark gray) → impassable
        Green areas (parks, grass) → high cost (5x)
        Water (blue) → impassable
        Paths/sidewalks (light/white) → low cost (1x)
        Roads (medium gray) → medium cost (2x)

    Args:
        filepath: Path to campus map image (JPG/PNG).
        target_size: Output grid resolution.

    Returns:
        WeightedGrid with terrain costs.
    """
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Could not load: {filepath}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Classify terrain
    terrain = np.zeros(h.shape, dtype=np.int8)  # Default: path (0)

    # Buildings: dark, low saturation
    buildings = (s < 60) & (v < 120) & (v > 30)
    terrain[buildings] = WeightedGrid.TERRAIN_IMPASSABLE

    # Water: blue hue, some saturation
    water = (h > 90) & (h < 130) & (s > 40)
    terrain[water] = WeightedGrid.TERRAIN_IMPASSABLE

    # Green areas: green hue, moderate saturation
    green = (h > 30) & (h < 85) & (s > 30) & (v > 80)
    terrain[green] = WeightedGrid.TERRAIN_GRASS

    # Roads: medium brightness, low saturation, not already classified
    roads = (v > 150) & (v < 210) & (s < 30) & (terrain == 0)
    terrain[roads] = WeightedGrid.TERRAIN_ROAD

    # Dilate buildings slightly to ensure they're solid
    building_mask = (terrain == WeightedGrid.TERRAIN_IMPASSABLE).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    building_mask = cv2.dilate(building_mask, kernel, iterations=1)
    terrain[building_mask == 1] = WeightedGrid.TERRAIN_IMPASSABLE

    # Resize terrain to target
    terrain_resized = cv2.resize(terrain.astype(np.float32),
                                  (target_size, target_size),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int8)

    # Create WeightedGrid
    grid = WeightedGrid(target_size, target_size)
    grid.set_terrain(terrain_resized)

    # Ensure start and goal corners are free paths
    margin = 3
    grid.terrain[0:margin, 0:margin] = WeightedGrid.TERRAIN_PATH
    grid.terrain[-margin:, -margin:] = WeightedGrid.TERRAIN_PATH
    grid.grid[0:margin, 0:margin] = 0
    grid.grid[-margin:, -margin:] = 0

    return grid


def save_terrain_visualization(grid, filepath):
    """
    Save terrain map as a colored PNG.
    White=path, Dark=building, Green=grass, Gray=road.
    """
    h, w = grid.height, grid.width
    viz = np.ones((h, w, 3), dtype=np.uint8) * 200

    viz[grid.terrain == WeightedGrid.TERRAIN_PATH] = [255, 255, 255]
    viz[grid.terrain == WeightedGrid.TERRAIN_IMPASSABLE] = [40, 40, 40]
    viz[grid.terrain == WeightedGrid.TERRAIN_GRASS] = [144, 238, 144]
    viz[grid.terrain == WeightedGrid.TERRAIN_ROAD] = [180, 180, 180]

    cv2.imwrite(filepath, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))


def save_occupancy_image(grid, filepath):
    """Save binary occupancy (black=obstacle, white=free)."""
    img = np.where(grid.grid == 1, 0, 255).astype(np.uint8)
    cv2.imwrite(filepath, img)


if __name__ == "__main__":
    os.makedirs("maps", exist_ok=True)

    # Try loading campus map
    candidates = ["maps/neu_campus.jpg", "maps/Campus_Map_2d.jpg",
                   "maps/neu_campus.png"]
    map_path = None
    for p in candidates:
        if os.path.exists(p):
            map_path = p
            break

    if map_path:
        print(f"Processing: {map_path}")
        grid = process_neu_campus(map_path, target_size=200)
        stats = grid.get_terrain_stats()

        print(f"\nNEU Campus WeightedGrid: {grid.width}x{grid.height}")
        print(f"  Paths:      {stats['path_pct']:.1f}% (cost 1.0)")
        print(f"  Buildings:  {stats['impassable_pct']:.1f}% (impassable)")
        print(f"  Grass:      {stats['grass_pct']:.1f}% (cost 5.0)")
        print(f"  Roads:      {stats['road_pct']:.1f}% (cost 2.0)")

        save_terrain_visualization(grid, "maps/neu_campus_terrain.png")
        save_occupancy_image(grid, "maps/neu_campus_occupancy.png")
        print("\nSaved terrain and occupancy maps")
    else:
        print("Campus map not found. Place image in maps/ folder.")
