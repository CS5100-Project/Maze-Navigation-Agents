import random
import time
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
import pygame
from gym import spaces


class MazeEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    Features:
    - Configurable maze size
    - Moving walls (optional)
    - Dymanic goal location (optional)
    - Time constraints
    - Random hazards (optional)
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        maze_size: int = 8,
        num_obstacles: int = 5,
        max_steps: int = 200,
        enable_moving_walls: bool = False,
        dynamic_goal: bool = False,
        random_hazards: bool = False,
    ):
        super(MazeEnv, self).__init__()

        # Set Maze Params
        self.maze_size = maze_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.enable_moving_walls = enable_moving_walls
        self.dynamic_goal = dynamic_goal
        self.random_hazards = random_hazards

        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
        low=-self.maze_size,
        high=self.maze_size,
        shape=(2,),
        dtype=np.float32
        )

        # Rendering
        self.cell_size = 50
        self.window_size = self.maze_size * self.cell_size
        self.window = None
        self.clock = None

        # Episode variables
        self.agent_pos = None
        self.goal_pos = None
        self.walls = None
        self.hazards = None
        self.steps_taken = 0
        self.visited_states = set()

        # Initialize the game
        self.reset()

    def _init_display(self):
        """
        Initialize the display
        """
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Maze Environment")
            self.clock = pygame.time.Clock()

    def _create_maze(self) -> None:
        """
        Create the maze
        """
        self.walls = np.zeros((self.maze_size, self.maze_size), dtype=bool)

        # Randomly place obstacles
        obstacle_positions = random.sample(
            [
                (i, j)
                for i in range(1, self.maze_size - 1)
                for j in range(1, self.maze_size - 1)
            ],
            self.num_obstacles,
        )
        for pos in obstacle_positions:
            self.walls[pos] = True

        # Initialize the hazards if enabled
        if self.random_hazards:
            self.hazards = np.zeros((self.maze_size, self.maze_size), dtype=bool)
            hazard_positions = random.sample(
                [
                    (i, j)
                    for i in range(1, self.maze_size - 1)
                    for j in range(1, self.maze_size - 1)
                    if not self.walls[i, j]
                ],
                self.num_obstacles // 2,
            )
            for pos in hazard_positions:
                self.hazards[pos] = True

    def _move_walls(self) -> None:
        """
        Move the random walls if enabled
        """
        if not self.enable_moving_walls:
            return

        for _ in range(self.num_obstacles // 2):
            wall_positions = np.where(self.walls)
            if len(wall_positions[0]) > 0:
                idx = random.randint(0, len(wall_positions[0]) - 1)
                old_pos = (wall_positions[0][idx], wall_positions[1][idx])

                # Try to move the wall in a random direction
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                random.shuffle(directions)

                for dx, dy in directions:
                    new_x, new_y = old_pos[0] + dx, old_pos[1] + dy
                    if (
                        0 < new_x < self.maze_size - 1
                        and 0 < new_y < self.maze_size - 1
                        and not self.walls[new_x, new_y]
                    ):
                        self.walls[old_pos] = False
                        self.walls[new_x, new_y] = True
                        break

    def _update_goal(self) -> None:
        """
        Update goal position if dynamic goal is enabled.
        """
        if not self.dynamic_goal:
            return

        if random.random() < 0.1:  # 10% chance to move the goal
            available_positions = [
                (i, j)
                for i in range(self.maze_size)
                for j in range(self.maze_size)
                if not self.walls[i, j] and (i, j) != self.agent_pos
            ]
            if available_positions:
                self.goal_pos = random.choice(available_positions)

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self._create_maze()
        self.steps_taken = 0
        self.visited_states = set()

        # Place agent and goal in random, non-overlapping positions
        available_positions = [
            (i, j)
            for i in range(self.maze_size)
            for j in range(self.maze_size)
            if not self.walls[i, j]
        ]

        if len(available_positions) < 2:
            raise ValueError("Maze configuration has insufficient free spaces")

        pos1, pos2 = random.sample(available_positions, 2)
        self.agent_pos = pos1
        self.goal_pos = pos2

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        Convert the environment state to the observation space format.
        """
        obs = np.zeros((self.maze_size, self.maze_size, 4), dtype=np.float32)

        # Channel 0: Agent position
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 1

        # Channel 1: Walls
        obs[:, :, 1] = self.walls

        # Channel 2: Goal
        obs[self.goal_pos[0], self.goal_pos[1], 2] = 1

        # Channel 3: Hazards (if enabled)
        if self.random_hazards:
            obs[:, :, 3] = self.hazards

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step within the environment.

        Args:
            action (int): 0: Up, 1: Right, 2: Down, 3: Left

        Returns:
            observation (np.ndarray): Environment observation
            reward (float): Reward for the action
            done (bool): Whether the episode has ended
            info (dict): Additional information
        """
        self.steps_taken += 1

        # Movement directions
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = directions[action]
        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        # Initialize reward
        reward = -0.01  # Small negative reward for each step
        done = False
        info = {}

        # Calculate distance to goal before and after move
        old_distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(
            self.agent_pos[1] - self.goal_pos[1]
        )
        new_distance = abs(new_pos[0] - self.goal_pos[0]) + abs(
            new_pos[1] - self.goal_pos[1]
        )

        # Check if the move is valid
        if (
            0 <= new_pos[0] < self.maze_size
            and 0 <= new_pos[1] < self.maze_size
            and not self.walls[new_pos]
        ):
            self.agent_pos = new_pos

            # Reward for moving closer to goal
            if new_distance < old_distance:
                reward += 1.0
            elif new_distance > old_distance:
                reward -= 0.5

            self.agent_pos = new_pos

            # Check if agent reached the goal
            if new_pos == self.goal_pos:
                reward = 50.0  # Much higher goal reward
                done = True
                info["success"] = True

        else:
            reward = -1.0  # Small penalty for invalid moves
            info["wall_collision"] = True

        # Timeout penalty
        if self.steps_taken >= self.max_steps:
            done = True
            reward = -10.0  # Penalty for not reaching goal
            info["timeout"] = True

        return self._get_observation(), reward, done, info

    def render(self, mode="human"):
        """
        Render the environment.
        """
        if mode == "human":
            self._init_display()

            # Fill background
            self.window.fill((255, 255, 255))

            # Draw walls
            for i in range(self.maze_size):
                for j in range(self.maze_size):
                    if self.walls[i, j]:
                        pygame.draw.rect(
                            self.window,
                            (0, 0, 0),
                            (
                                j * self.cell_size,
                                i * self.cell_size,
                                self.cell_size,
                                self.cell_size,
                            ),
                        )

            # Draw hazards
            if self.random_hazards:
                for i in range(self.maze_size):
                    for j in range(self.maze_size):
                        if self.hazards[i, j]:
                            pygame.draw.rect(
                                self.window,
                                (255, 0, 0),
                                (
                                    j * self.cell_size,
                                    i * self.cell_size,
                                    self.cell_size,
                                    self.cell_size,
                                ),
                                2,
                            )

            # Draw goal
            pygame.draw.rect(
                self.window,
                (0, 255, 0),
                (
                    self.goal_pos[1] * self.cell_size,
                    self.goal_pos[0] * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
            )

            # Draw agent
            pygame.draw.circle(
                self.window,
                (0, 0, 255),
                (
                    self.agent_pos[1] * self.cell_size + self.cell_size // 2,
                    self.agent_pos[0] * self.cell_size + self.cell_size // 2,
                ),
                self.cell_size // 3,
            )

            pygame.display.flip()
            self.clock.tick(10)

            return None
        elif mode == "rgb_array":
            # Return the game surface as a numpy array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Clean up resources.
        """
        if self.window is not None:
            pygame.quit()
            self.window = None
