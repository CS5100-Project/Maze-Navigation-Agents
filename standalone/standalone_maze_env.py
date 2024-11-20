import numpy as np
import random
from typing import Tuple, Dict


class MazeEnv:
    def __init__(
            self,
            maze_size: int = 8,
            num_obstacles: int = 5,
            max_steps: int = 200
    ):
        self.maze_size = maze_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps

        self.action_space = 4  # Up, Right, Down, Left
        self.observation_space_shape = (maze_size, maze_size, 3)

        self.agent_pos = None
        self.goal_pos = None
        self.walls = None
        self.steps_taken = 0

        self.reset()

    def _create_maze(self) -> None:
        self.walls = np.zeros((self.maze_size, self.maze_size), dtype=bool)
        obstacle_positions = random.sample(
            [(i, j) for i in range(1, self.maze_size - 1) for j in range(1, self.maze_size - 1)],
            self.num_obstacles
        )
        for pos in obstacle_positions:
            self.walls[pos] = True

    def reset(self) -> np.ndarray:
        self._create_maze()
        self.steps_taken = 0

        available_positions = [
            (i, j) for i in range(self.maze_size) for j in range(self.maze_size)
            if not self.walls[i, j]
        ]
        self.agent_pos, self.goal_pos = random.sample(available_positions, 2)

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space_shape, dtype=np.float32)
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 1  # Agent position
        obs[:, :, 1] = self.walls  # Walls
        obs[self.goal_pos[0], self.goal_pos[1], 2] = 1  # Goal position
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps_taken += 1

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = directions[action]
        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        reward = -0.1  # Small negative reward for each step
        done = False
        info = {}

        if (0 <= new_pos[0] < self.maze_size and
                0 <= new_pos[1] < self.maze_size and
                not self.walls[new_pos]):
            self.agent_pos = new_pos

            if new_pos == self.goal_pos:
                reward = 10.0
                done = True
                info["success"] = True
        else:
            reward -= 1.0  # Penalty for hitting a wall
            info["wall_collision"] = True

        if self.steps_taken >= self.max_steps:
            done = True
            info["timeout"] = True

        return self._get_observation(), reward, done, info