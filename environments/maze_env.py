import gym
from gym import spaces
import numpy as np
import random
import pygame
import time
from typing import Optional, List, Tuple, Dict

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
            random_hazards: bool = False
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
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.maze_size, self.maze_size, 4), dtype=np.float32)
        