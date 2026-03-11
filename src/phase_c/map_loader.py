import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import numpy as np
import cv2
from grid import Grid


class MapLoader:
    """Load real floorplan maps from image files and convert to Grid objects."""

    @staticmethod
    def from_image(filepath, target_size=200, threshold=128):
        """
        Load a PNG/PGM occupancy map and convert to a Grid.

        Args:
            filepath: Path to image file.
            target_size: Downsample to this resolution (width and height).
            threshold: Pixel value below this = obstacle. Above = free.

        Returns:
            Grid object with obstacles loaded from the image.
        """
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {filepath}")

        # Resize to target resolution
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # Threshold: dark pixels = obstacles, light = free
        binary = (img < threshold).astype(np.int8)

        # Create Grid and inject the map
        grid = Grid(target_size, target_size, obstacle_density=0.0, seed=None)
        grid.grid = binary

        # Ensure start and goal are free
        grid.grid[0, 0] = 0
        grid.grid[target_size - 1, target_size - 1] = 0

        return grid

    @staticmethod
    def get_map_info(grid):
        """Return statistics about a loaded map."""
        total = grid.width * grid.height
        obstacles = np.sum(grid.grid == 1)
        free = total - obstacles
        density = obstacles / total

        return {
            "width": grid.width,
            "height": grid.height,
            "total_cells": total,
            "obstacle_cells": int(obstacles),
            "free_cells": int(free),
            "obstacle_density": density,
        }


class ProceduralMapGenerator:
    """Generate realistic building floorplan maps for benchmarking."""

    @staticmethod
    def office_building(size=200, seed=42):
        """
        Generate an office building with rooms, corridors, and doorways.

        Layout:
        - Outer walls
        - Horizontal and vertical corridors
        - Rooms of varying sizes
        - Doorways connecting rooms to corridors
        """
        rng = np.random.RandomState(seed)
        grid = Grid(size, size, obstacle_density=0.0, seed=None)
        grid.grid = np.zeros((size, size), dtype=np.int8)  # Start all free

        # Outer walls (3 cells thick)
        wall_thickness = 3
        grid.grid[:wall_thickness, :] = 1
        grid.grid[-wall_thickness:, :] = 1
        grid.grid[:, :wall_thickness] = 1
        grid.grid[:, -wall_thickness:] = 1

        # Main corridors
        corridor_width = 5
        # Horizontal corridor at 1/3 and 2/3 height
        h1 = size // 3
        h2 = 2 * size // 3
        # Vertical corridor at 1/3 and 2/3 width
        v1 = size // 3
        v2 = 2 * size // 3

        # Draw room walls (everything is wall, then carve corridors and rooms)
        # Actually easier: start with walls, carve out spaces
        grid.grid[:, :] = 1  # All walls

        # Carve corridors
        grid.grid[h1:h1+corridor_width, wall_thickness:-wall_thickness] = 0
        grid.grid[h2:h2+corridor_width, wall_thickness:-wall_thickness] = 0
        grid.grid[wall_thickness:-wall_thickness, v1:v1+corridor_width] = 0
        grid.grid[wall_thickness:-wall_thickness, v2:v2+corridor_width] = 0

        # Carve rooms in each quadrant
        room_margin = 8
        room_door_width = 3

        # Define quadrants
        quadrants = [
            (wall_thickness + 1, wall_thickness + 1, h1 - 1, v1 - 1),        # Top-left
            (wall_thickness + 1, v1 + corridor_width + 1, h1 - 1, v2 - 1),   # Top-middle
            (wall_thickness + 1, v2 + corridor_width + 1, h1 - 1, size - wall_thickness - 1),  # Top-right
            (h1 + corridor_width + 1, wall_thickness + 1, h2 - 1, v1 - 1),   # Middle-left
            (h1 + corridor_width + 1, v1 + corridor_width + 1, h2 - 1, v2 - 1),  # Center
            (h1 + corridor_width + 1, v2 + corridor_width + 1, h2 - 1, size - wall_thickness - 1),  # Middle-right
            (h2 + corridor_width + 1, wall_thickness + 1, size - wall_thickness - 1, v1 - 1),  # Bottom-left
            (h2 + corridor_width + 1, v1 + corridor_width + 1, size - wall_thickness - 1, v2 - 1),  # Bottom-middle
            (h2 + corridor_width + 1, v2 + corridor_width + 1, size - wall_thickness - 1, size - wall_thickness - 1),  # Bottom-right
        ]

        for r1, c1, r2, c2 in quadrants:
            if r2 - r1 > room_margin and c2 - c1 > room_margin:
                # Carve room interior (leave walls around edges)
                inner_margin = 2
                grid.grid[r1+inner_margin:r2-inner_margin, c1+inner_margin:c2-inner_margin] = 0

                # Add doorways to adjacent corridors
                mid_r = (r1 + r2) // 2
                mid_c = (c1 + c2) // 2

                # Door on each wall if adjacent to corridor
                # Top door
                if r1 > wall_thickness + 2:
                    grid.grid[r1:r1+inner_margin+1, mid_c-1:mid_c+2] = 0
                # Bottom door
                if r2 < size - wall_thickness - 2:
                    grid.grid[r2-inner_margin:r2+1, mid_c-1:mid_c+2] = 0
                # Left door
                if c1 > wall_thickness + 2:
                    grid.grid[mid_r-1:mid_r+2, c1:c1+inner_margin+1] = 0
                # Right door
                if c2 < size - wall_thickness - 2:
                    grid.grid[mid_r-1:mid_r+2, c2-inner_margin:c2+1] = 0

        # Ensure start and goal are free
        grid.grid[wall_thickness + 1, wall_thickness + 1] = 0
        grid.grid[0, 0] = 0
        grid.grid[size - 1, size - 1] = 0
        # Clear path from corners to nearest corridor
        grid.grid[0:h1+1, 0:wall_thickness+2] = 0
        grid.grid[size-wall_thickness-2:size, size-wall_thickness-2:size] = 0

        return grid

    @staticmethod
    def warehouse(size=200, seed=42):
        """
        Generate a warehouse with shelving aisles and open loading areas.

        Layout:
        - Regular shelving rows
        - Cross aisles
        - Open area at one end
        """
        rng = np.random.RandomState(seed)
        grid = Grid(size, size, obstacle_density=0.0, seed=None)
        grid.grid = np.zeros((size, size), dtype=np.int8)

        # Outer walls
        wall = 2
        grid.grid[:wall, :] = 1
        grid.grid[-wall:, :] = 1
        grid.grid[:, :wall] = 1
        grid.grid[:, -wall:] = 1

        # Shelving rows (vertical)
        shelf_width = 4
        aisle_width = 5
        shelf_start = wall + aisle_width
        open_area_end = size // 5  # Open area at top

        col = shelf_start
        while col + shelf_width < size - wall - aisle_width:
            # Leave cross aisles at 1/3 and 2/3
            cross1 = size // 3
            cross2 = 2 * size // 3
            cross_width = 4

            for row in range(open_area_end, size - wall):
                if not (cross1 <= row <= cross1 + cross_width or
                        cross2 <= row <= cross2 + cross_width):
                    grid.grid[row, col:col+shelf_width] = 1

            col += shelf_width + aisle_width

        # Ensure start and goal
        grid.grid[0, 0] = 0
        grid.grid[size-1, size-1] = 0
        grid.grid[wall:wall+3, wall:wall+3] = 0
        grid.grid[size-wall-3:size-wall, size-wall-3:size-wall] = 0

        return grid

    @staticmethod
    def hospital(size=200, seed=42):
        """
        Generate a hospital floor with long corridors, T-junctions, and narrow passages.
        """
        rng = np.random.RandomState(seed)
        grid = Grid(size, size, obstacle_density=0.0, seed=None)
        grid.grid = np.ones((size, size), dtype=np.int8)  # All walls

        corridor = 4
        wall = 2

        # Main horizontal corridor through center
        center_r = size // 2
        grid.grid[center_r-corridor//2:center_r+corridor//2, wall:size-wall] = 0

        # Vertical corridors branching off
        num_branches = 5
        for i in range(num_branches):
            col = wall + (i + 1) * (size - 2*wall) // (num_branches + 1)
            grid.grid[wall:size-wall, col-corridor//2:col+corridor//2] = 0

        # Rooms along corridors
        room_size = size // (num_branches + 1) // 2 - 2
        for i in range(num_branches):
            col = wall + (i + 1) * (size - 2*wall) // (num_branches + 1)

            # Rooms above main corridor
            for side in [-1, 1]:
                room_r = center_r + side * (corridor + 4)
                room_c = col + corridor

                if (wall < room_r < size - wall - room_size and
                        wall < room_c < size - wall - room_size):
                    grid.grid[room_r:room_r+room_size, room_c:room_c+room_size] = 0
                    # Doorway to corridor
                    door_r = room_r if side > 0 else room_r + room_size
                    grid.grid[min(door_r, room_r):max(door_r, room_r+room_size),
                              col-corridor//2:col+corridor//2] = 0

        # Narrow passage connecting two corridors
        narrow = 2
        narrow_r = size // 4
        grid.grid[narrow_r:narrow_r+narrow, wall:size-wall] = 0

        # Ensure start and goal
        grid.grid[0, 0] = 0
        grid.grid[0:wall+2, 0:wall+2] = 0
        grid.grid[size-1, size-1] = 0
        grid.grid[size-wall-2:size, size-wall-2:size] = 0

        return grid


def save_map_as_image(grid, filepath):
    """Save a grid as a PNG image for visualization."""
    img = np.where(grid.grid == 1, 0, 255).astype(np.uint8)
    cv2.imwrite(filepath, img)


if __name__ == "__main__":
    os.makedirs("maps", exist_ok=True)

    print("Generating procedural maps...")

    # Office building
    office = ProceduralMapGenerator.office_building(size=200, seed=42)
    info = MapLoader.get_map_info(office)
    print(f"\nOffice Building: {info['width']}x{info['height']}, "
          f"density={info['obstacle_density']:.2f}")
    save_map_as_image(office, "maps/office_building.png")

    # Warehouse
    warehouse = ProceduralMapGenerator.warehouse(size=200, seed=42)
    info = MapLoader.get_map_info(warehouse)
    print(f"Warehouse: {info['width']}x{info['height']}, "
          f"density={info['obstacle_density']:.2f}")
    save_map_as_image(warehouse, "maps/warehouse.png")

    # Hospital
    hospital = ProceduralMapGenerator.hospital(size=200, seed=42)
    info = MapLoader.get_map_info(hospital)
    print(f"Hospital: {info['width']}x{info['height']}, "
          f"density={info['obstacle_density']:.2f}")
    save_map_as_image(hospital, "maps/hospital.png")

    print("\nMaps saved to maps/ folder")
    print("\nTo load a real map later:")
    print("  grid = MapLoader.from_image('maps/intel_lab.png', target_size=200)")
