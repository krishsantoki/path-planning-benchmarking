import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import numpy as np
from grid import Grid


class DynamicGrid(Grid):
    """Grid with dynamic obstacle support for D* Lite replanning."""

    def __init__(self, width=100, height=100, obstacle_density=0.3, seed=None):
        super().__init__(width, height, obstacle_density, seed)
        self.change_log = []
        self.timestep = 0

    def update_obstacle(self, row, col, blocked, timestep=None):
        """
        Change a cell's state and log the change.

        Args:
            row, col: Cell position.
            blocked: True to set obstacle, False to clear.
            timestep: Current simulation timestep.
        """
        if 0 <= row < self.height and 0 <= col < self.width:
            old_state = self.grid[row, col]
            new_state = 1 if blocked else 0

            if old_state != new_state:
                self.grid[row, col] = new_state
                ts = timestep if timestep is not None else self.timestep
                self.change_log.append({
                    "timestep": ts,
                    "row": row,
                    "col": col,
                    "old": old_state,
                    "new": new_state
                })
                return True
        return False

    def get_changes_since(self, timestep):
        """Return all changes that occurred at or after the given timestep."""
        return [c for c in self.change_log if c["timestep"] >= timestep]

    def clear_log(self):
        """Clear the change log."""
        self.change_log = []

    def step(self):
        """Advance timestep."""
        self.timestep += 1


class ObstacleAgent:
    """A moving obstacle that follows a preset path on the grid."""

    def __init__(self, path, grid, loop=True):
        """
        Args:
            path: List of (row, col) positions the obstacle visits.
            grid: DynamicGrid reference.
            loop: If True, obstacle loops back to start after reaching end.
        """
        self.path = path
        self.grid = grid
        self.loop = loop
        self.index = 0
        self.current_pos = path[0]

        # Place initial obstacle
        self.grid.update_obstacle(self.current_pos[0], self.current_pos[1], True)

    def step(self, timestep):
        """
        Move obstacle to next position in path.
        Clears old position, sets new position.

        Returns:
            List of changed cells: [(row, col, blocked), ...]
        """
        changes = []

        # Clear current position
        old_r, old_c = self.current_pos
        self.grid.update_obstacle(old_r, old_c, False, timestep)
        changes.append((old_r, old_c, False))

        # Advance index
        self.index += 1
        if self.index >= len(self.path):
            if self.loop:
                self.index = 0
            else:
                self.index = len(self.path) - 1

        # Set new position
        self.current_pos = self.path[self.index]
        new_r, new_c = self.current_pos
        self.grid.update_obstacle(new_r, new_c, True, timestep)
        changes.append((new_r, new_c, True))

        return changes


def create_moving_obstacles(grid, num_agents=3, path_length=10, seed=42):
    """
    Generate random moving obstacle agents on the grid.

    Args:
        grid: DynamicGrid.
        num_agents: Number of moving obstacles.
        path_length: Steps in each obstacle's patrol path.
        seed: Random seed.

    Returns:
        List of ObstacleAgent objects.
    """
    rng = np.random.RandomState(seed)
    agents = []

    for _ in range(num_agents):
        # Find a random free starting cell away from corners
        while True:
            start_r = rng.randint(3, grid.height - 3)
            start_c = rng.randint(3, grid.width - 3)
            if grid.is_free(start_r, start_c):
                break

        # Generate a random walk path
        path = [(start_r, start_c)]
        r, c = start_r, start_c
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for _ in range(path_length - 1):
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


if __name__ == "__main__":
    g = DynamicGrid(20, 20, obstacle_density=0.15, seed=42)
    agents = create_moving_obstacles(g, num_agents=3, path_length=8, seed=42)

    print(f"Grid: {g}")
    print(f"Moving obstacles: {len(agents)}")
    for i, agent in enumerate(agents):
        print(f"  Agent {i}: path length = {len(agent.path)}, start = {agent.path[0]}")

    # Simulate 5 steps
    for t in range(5):
        g.step()
        for agent in agents:
            agent.step(g.timestep)
        changes = g.get_changes_since(g.timestep)
        print(f"\nTimestep {g.timestep}: {len(changes)} cell changes")
