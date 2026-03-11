import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import numpy as np
from grid import Grid


class WeightedGrid(Grid):
    """
    Grid with terrain-based movement costs.

    Terrain types:
        0 = path/sidewalk (cost 1.0)
        1 = impassable (buildings, water)
        2 = grass/green area (cost 5.0)
        3 = road (cost 2.0)

    Algorithms using get_neighbors() automatically get weighted costs.
    """

    TERRAIN_PATH = 0       # Sidewalks, walkways
    TERRAIN_IMPASSABLE = 1 # Buildings, water
    TERRAIN_GRASS = 2      # Green areas, parks
    TERRAIN_ROAD = 3       # Roads, streets

    COST_MAP = {
        0: 1.0,    # Path: preferred, lowest cost
        1: float('inf'),  # Building: impassable
        2: 5.0,    # Grass: walkable but discouraged
        3: 2.0,    # Road: walkable but less preferred
    }

    def __init__(self, width=100, height=100, obstacle_density=0.0, seed=None):
        super().__init__(width, height, obstacle_density=0.0, seed=seed)
        # Terrain map: separate from binary grid
        self.terrain = np.zeros((height, width), dtype=np.int8)

    def set_terrain(self, terrain_array):
        """Set terrain map and sync binary grid."""
        self.terrain = terrain_array.astype(np.int8)
        # Binary grid: 1 = impassable, 0 = passable
        self.grid = (self.terrain == self.TERRAIN_IMPASSABLE).astype(np.int8)

    def get_terrain_cost(self, row, col):
        """Get movement cost for a cell based on terrain."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.COST_MAP.get(self.terrain[row, col], float('inf'))
        return float('inf')

    def get_neighbors(self, row, col, connectivity=4):
        """
        Return neighbors with terrain-weighted costs.
        Overrides Grid.get_neighbors() to add terrain costs.
        """
        neighbors = []
        cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in cardinal:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                cost = self.get_terrain_cost(nr, nc)
                if cost < float('inf'):
                    neighbors.append((nr, nc, cost))

        if connectivity == 8:
            for dr, dc in diagonal:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    cost = self.get_terrain_cost(nr, nc)
                    if cost < float('inf'):
                        neighbors.append((nr, nc, cost * np.sqrt(2)))

        return neighbors

    def get_terrain_stats(self):
        """Return terrain distribution stats."""
        total = self.width * self.height
        return {
            "path_cells": int(np.sum(self.terrain == 0)),
            "impassable_cells": int(np.sum(self.terrain == 1)),
            "grass_cells": int(np.sum(self.terrain == 2)),
            "road_cells": int(np.sum(self.terrain == 3)),
            "path_pct": np.sum(self.terrain == 0) / total * 100,
            "impassable_pct": np.sum(self.terrain == 1) / total * 100,
            "grass_pct": np.sum(self.terrain == 2) / total * 100,
            "road_pct": np.sum(self.terrain == 3) / total * 100,
        }


if __name__ == "__main__":
    # Quick test
    wg = WeightedGrid(10, 10)
    terrain = np.zeros((10, 10), dtype=np.int8)
    terrain[3:7, 3:7] = 1  # Building in center
    terrain[0:3, :] = 2    # Grass at top
    terrain[8, :] = 3      # Road at bottom

    wg.set_terrain(terrain)
    stats = wg.get_terrain_stats()
    print(f"Terrain stats: {stats}")

    # Test neighbor costs
    print(f"\nNeighbors of (2,2) - grass area:")
    for n in wg.get_neighbors(2, 2, 4):
        print(f"  {n[0:2]} cost={n[2]}")

    print(f"\nNeighbors of (8,5) - road:")
    for n in wg.get_neighbors(8, 5, 4):
        print(f"  {n[0:2]} cost={n[2]}")

    print(f"\nNeighbors of (5,2) - near building:")
    for n in wg.get_neighbors(5, 2, 4):
        print(f"  {n[0:2]} cost={n[2]}")
