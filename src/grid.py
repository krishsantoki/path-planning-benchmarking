import numpy as np


class Grid:
    """2D occupancy grid for path planning benchmarking."""

    def __init__(self, width=100, height=100, obstacle_density=0.3, seed=None):
        """
        Args:
            width: Grid width in cells.
            height: Grid height in cells.
            obstacle_density: Fraction of cells that are obstacles (0.0 to 1.0).
            seed: Random seed for reproducibility.
        """
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density

        if seed is not None:
            np.random.seed(seed)

        # 0 = free, 1 = obstacle
        self.grid = (np.random.random((height, width)) < obstacle_density).astype(np.int8)

        # Ensure start (top-left) and goal (bottom-right) are free
        self.grid[0, 0] = 0
        self.grid[height - 1, width - 1] = 0

    def is_free(self, row, col):
        """Check if a cell is within bounds and not an obstacle."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row, col] == 0
        return False

    def get_neighbors(self, row, col, connectivity=4):
        """
        Return free neighboring cells.

        Args:
            row: Current cell row.
            col: Current cell column.
            connectivity: 4 (cardinal) or 8 (cardinal + diagonal).

        Returns:
            List of (row, col, cost) tuples.
        """
        neighbors = []

        # Cardinal directions: up, down, left, right
        cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Diagonal directions
        diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in cardinal:
            nr, nc = row + dr, col + dc
            if self.is_free(nr, nc):
                neighbors.append((nr, nc, 1.0))

        if connectivity == 8:
            for dr, dc in diagonal:
                nr, nc = row + dr, col + dc
                if self.is_free(nr, nc):
                    neighbors.append((nr, nc, np.sqrt(2)))

        return neighbors

    def set_obstacle(self, row, col):
        """Set a cell as an obstacle."""
        if 0 <= row < self.height and 0 <= col < self.width:
            self.grid[row, col] = 1

    def clear_obstacle(self, row, col):
        """Set a cell as free."""
        if 0 <= row < self.height and 0 <= col < self.width:
            self.grid[row, col] = 0

    def __repr__(self):
        return f"Grid({self.width}x{self.height}, density={self.obstacle_density})"


if __name__ == "__main__":
    # Quick test
    g = Grid(10, 10, obstacle_density=0.3, seed=42)
    print(g)
    print(g.grid)
    print(f"Start free: {g.is_free(0, 0)}")
    print(f"Goal free: {g.is_free(9, 9)}")
    print(f"Neighbors of (0,0) 4-conn: {g.get_neighbors(0, 0, 4)}")
    print(f"Neighbors of (0,0) 8-conn: {g.get_neighbors(0, 0, 8)}")
