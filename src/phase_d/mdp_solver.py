import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import numpy as np
import time
from grid import Grid


class StochasticActionModel:
    """
    Models actuator noise for robot movement.
    
    When the robot intends to move in a direction:
        - forward_prob: chance it moves as intended
        - drift_prob: chance it drifts left or right (perpendicular)
    
    Example: forward=0.8, drift=0.1 means 80% forward, 10% left, 10% right.
    """

    # Action definitions: (dr, dc)
    ACTIONS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    # Perpendicular directions for each action
    PERPENDICULAR = {
        "up": ["left", "right"],
        "down": ["left", "right"],
        "left": ["up", "down"],
        "right": ["up", "down"],
    }

    def __init__(self, forward_prob=0.8, drift_prob=0.1):
        """
        Args:
            forward_prob: Probability of moving as intended.
            drift_prob: Probability of drifting to each perpendicular side.
                        Total = forward_prob + 2*drift_prob should equal 1.0.
        """
        self.forward_prob = forward_prob
        self.drift_prob = drift_prob

    def get_transition_probs(self, action):
        """
        Return list of (probability, resulting_action_direction) tuples.
        
        Args:
            action: Intended action string ("up", "down", "left", "right").
        
        Returns:
            List of (prob, (dr, dc)) tuples.
        """
        perps = self.PERPENDICULAR[action]
        return [
            (self.forward_prob, self.ACTIONS[action]),
            (self.drift_prob, self.ACTIONS[perps[0]]),
            (self.drift_prob, self.ACTIONS[perps[1]]),
        ]

    def sample_action(self, action, rng=None):
        """
        Sample an actual movement given intended action.
        
        Returns:
            (dr, dc) tuple of actual movement.
        """
        if rng is None:
            rng = np.random
        
        r = rng.random()
        if r < self.forward_prob:
            return self.ACTIONS[action]
        elif r < self.forward_prob + self.drift_prob:
            return self.ACTIONS[self.PERPENDICULAR[action][0]]
        else:
            return self.ACTIONS[self.PERPENDICULAR[action][1]]


class MDP:
    """
    Markov Decision Process defined over a Grid.
    
    State space: all free cells (row, col)
    Action space: up, down, left, right
    Transition: StochasticActionModel
    Rewards:
        +100 at goal
        -1 per step (encourages short paths)
        -50 for hitting obstacle (bounces back)
    """

    def __init__(self, grid, goal, action_model, gamma=0.95):
        """
        Args:
            grid: Grid object.
            goal: (row, col) goal position.
            action_model: StochasticActionModel instance.
            gamma: Discount factor.
        """
        self.grid = grid
        self.goal = goal
        self.action_model = action_model
        self.gamma = gamma
        self.actions = ["up", "down", "left", "right"]

        # Rewards
        self.goal_reward = 100.0
        self.step_cost = -1.0
        self.obstacle_penalty = -50.0

    def get_next_state(self, state, dr, dc):
        """
        Compute next state given movement. If hitting wall/obstacle, stay in place.
        
        Returns:
            (next_state, reward)
        """
        nr, nc = state[0] + dr, state[1] + dc

        # Hit boundary or obstacle — stay in place with penalty
        if not (0 <= nr < self.grid.height and 0 <= nc < self.grid.width):
            return state, self.step_cost + self.obstacle_penalty
        if self.grid.grid[nr, nc] == 1:
            return state, self.step_cost + self.obstacle_penalty

        next_state = (nr, nc)

        # Goal reached
        if next_state == self.goal:
            return next_state, self.goal_reward

        return next_state, self.step_cost

    def get_transition(self, state, action):
        """
        Get all possible transitions from state given action.
        
        Returns:
            List of (probability, next_state, reward) tuples.
        """
        transitions = []
        for prob, (dr, dc) in self.action_model.get_transition_probs(action):
            next_state, reward = self.get_next_state(state, dr, dc)
            transitions.append((prob, next_state, reward))
        return transitions


class ValueIteration:
    """
    Value Iteration solver for MDP.
    
    Computes optimal value function V*(s) and policy pi*(s)
    using the Bellman Optimality Equation:
    
    V*(s) = max_a sum_s' T(s,a,s') [R(s,a,s') + gamma * V*(s')]
    """

    def __init__(self, mdp, convergence_threshold=0.001):
        self.mdp = mdp
        self.threshold = convergence_threshold
        self.V = {}  # Value function
        self.policy = {}  # Optimal policy

        # Initialize values to 0 for all free cells
        for r in range(mdp.grid.height):
            for c in range(mdp.grid.width):
                if mdp.grid.is_free(r, c):
                    self.V[(r, c)] = 0.0

        # Goal value
        self.V[mdp.goal] = 0.0

    def solve(self):
        """
        Run value iteration until convergence.
        
        Returns:
            dict with: iterations, convergence_time_ms, final_max_delta.
        """
        start_time = time.perf_counter()
        iterations = 0

        while True:
            max_delta = 0.0
            iterations += 1

            for state in list(self.V.keys()):
                # Goal is terminal — value stays 0
                if state == self.mdp.goal:
                    continue

                old_v = self.V[state]

                # Bellman update: V(s) = max_a Q(s,a)
                best_value = float('-inf')
                best_action = None

                for action in self.mdp.actions:
                    q_value = 0.0
                    for prob, next_state, reward in self.mdp.get_transition(state, action):
                        next_v = self.V.get(next_state, 0.0)
                        q_value += prob * (reward + self.mdp.gamma * next_v)

                    if q_value > best_value:
                        best_value = q_value
                        best_action = action

                self.V[state] = best_value
                self.policy[state] = best_action

                delta = abs(old_v - best_value)
                if delta > max_delta:
                    max_delta = delta

            if max_delta < self.threshold:
                break

        end_time = time.perf_counter()
        convergence_time = (end_time - start_time) * 1000

        return {
            "iterations": iterations,
            "convergence_time_ms": convergence_time,
            "final_max_delta": max_delta,
        }

    def get_action(self, state):
        """Get optimal action for a state."""
        return self.policy.get(state, None)

    def get_value(self, state):
        """Get optimal value for a state."""
        return self.V.get(state, 0.0)


class PolicyIteration:
    """
    Policy Iteration solver for MDP.

    Alternates between:
    1. Policy Evaluation: compute V^pi(s) for current policy
    2. Policy Improvement: update policy greedily from V^pi

    Repeat until policy is stable.
    """

    def __init__(self, mdp, eval_threshold=0.001):
        self.mdp = mdp
        self.eval_threshold = eval_threshold
        self.V = {}
        self.policy = {}

        # Initialize random policy and zero values
        for r in range(mdp.grid.height):
            for c in range(mdp.grid.width):
                if mdp.grid.is_free(r, c):
                    self.V[(r, c)] = 0.0
                    self.policy[(r, c)] = "right"  # Arbitrary initial policy

        self.V[mdp.goal] = 0.0

    def _policy_evaluation(self):
        """Evaluate current policy until values converge."""
        iterations = 0
        while True:
            max_delta = 0.0
            iterations += 1

            for state in list(self.V.keys()):
                if state == self.mdp.goal:
                    continue

                action = self.policy.get(state)
                if action is None:
                    continue

                # V^pi(s) = sum_s' T(s, pi(s), s') [R + gamma * V^pi(s')]
                new_v = 0.0
                for prob, next_state, reward in self.mdp.get_transition(state, action):
                    next_v = self.V.get(next_state, 0.0)
                    new_v += prob * (reward + self.mdp.gamma * next_v)

                delta = abs(self.V[state] - new_v)
                if delta > max_delta:
                    max_delta = delta
                self.V[state] = new_v

            if max_delta < self.eval_threshold:
                break

        return iterations

    def _policy_improvement(self):
        """Improve policy greedily. Returns True if policy changed."""
        policy_stable = True

        for state in list(self.policy.keys()):
            if state == self.mdp.goal:
                continue

            old_action = self.policy[state]
            best_value = float('-inf')
            best_action = old_action

            for action in self.mdp.actions:
                q_value = 0.0
                for prob, next_state, reward in self.mdp.get_transition(state, action):
                    next_v = self.V.get(next_state, 0.0)
                    q_value += prob * (reward + self.mdp.gamma * next_v)

                if q_value > best_value:
                    best_value = q_value
                    best_action = action

            self.policy[state] = best_action
            if best_action != old_action:
                policy_stable = False

        return policy_stable

    def solve(self):
        """Run policy iteration until policy is stable."""
        start_time = time.perf_counter()
        policy_iterations = 0
        total_eval_iterations = 0

        while True:
            policy_iterations += 1

            # Evaluate
            eval_iters = self._policy_evaluation()
            total_eval_iterations += eval_iters

            # Improve
            stable = self._policy_improvement()

            if stable:
                break

        end_time = time.perf_counter()
        convergence_time = (end_time - start_time) * 1000

        return {
            "policy_iterations": policy_iterations,
            "total_eval_iterations": total_eval_iterations,
            "convergence_time_ms": convergence_time,
        }

    def get_action(self, state):
        return self.policy.get(state, None)

    def get_value(self, state):
        return self.V.get(state, 0.0)


if __name__ == "__main__":
    g = Grid(20, 20, obstacle_density=0.15, seed=42)
    start = (0, 0)
    goal = (19, 19)

    noise = StochasticActionModel(forward_prob=0.8, drift_prob=0.1)
    mdp = MDP(g, goal, noise, gamma=0.95)

    # Value Iteration
    print("Value Iteration:")
    vi = ValueIteration(mdp, convergence_threshold=0.001)
    vi_result = vi.solve()
    print(f"  Iterations: {vi_result['iterations']}")
    print(f"  Time: {vi_result['convergence_time_ms']:.1f} ms")
    print(f"  Start value: {vi.get_value(start):.2f}")

    # Policy Iteration
    print("\nPolicy Iteration:")
    pi = PolicyIteration(mdp, eval_threshold=0.001)
    pi_result = pi.solve()
    print(f"  Policy iterations: {pi_result['policy_iterations']}")
    print(f"  Total eval iterations: {pi_result['total_eval_iterations']}")
    print(f"  Time: {pi_result['convergence_time_ms']:.1f} ms")
    print(f"  Start value: {pi.get_value(start):.2f}")

    # Compare policies
    mismatches = 0
    total = 0
    for state in vi.policy:
        if state == goal:
            continue
        total += 1
        if vi.get_action(state) != pi.get_action(state):
            mismatches += 1
    print(f"\nPolicy agreement: {(total-mismatches)/total*100:.1f}% ({mismatches} mismatches out of {total})")

    # Simulate episode with VI policy
    print("\nEpisode with Value Iteration policy:")
    rng = np.random.RandomState(42)
    pos = start
    steps = 0
    total_reward = 0
    for _ in range(200):
        action = vi.get_action(pos)
        if action is None:
            break
        dr, dc = noise.sample_action(action, rng)
        next_state, reward = mdp.get_next_state(pos, dr, dc)
        total_reward += reward
        pos = next_state
        steps += 1
        if pos == goal:
            print(f"  Goal in {steps} steps, reward: {total_reward:.1f}")
            break
    else:
        print(f"  Failed after 200 steps")
