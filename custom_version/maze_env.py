import numpy as np
import random
import pygame
from typing import Tuple, Dict

class MazeEnv:
    def __init__(
        self,
        maze_size: int = 8,
        num_obstacles: int = 5,
        max_steps: int = 200
    ):
        # Initialize pygame
        pygame.init()
        
        # Environment parameters
        self.maze_size = maze_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.action_space = 4  # Up, Right, Down, Left
        self.observation_space_shape = (maze_size, maze_size, 3)
        
        # Display parameters
        self.cell_size = 40
        self.window_size = maze_size * self.cell_size
        self.screen = None
        self.clock = pygame.time.Clock()
        
        # Colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'gray': (128, 128, 128)
        }
        
        # State variables
        self.agent_pos = None
        self.goal_pos = None
        self.walls = None
        self.steps_taken = 0
        
        # Seed random for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        self.reset()

    def _create_maze(self) -> None:
        """Initialize maze layout with walls and spaces."""
        self.walls = np.zeros((self.maze_size, self.maze_size), dtype=bool)
        
        # Place random obstacles
        obstacle_positions = random.sample(
            [(i, j) for i in range(1, self.maze_size - 1) 
             for j in range(1, self.maze_size - 1)],
            self.num_obstacles
        )
        for pos in obstacle_positions:
            self.walls[pos] = True

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self._create_maze()
        self.steps_taken = 0

        # Place agent and goal in non-overlapping positions
        available_positions = [
            (i, j) for i in range(self.maze_size)
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
        """Convert the environment state to observation."""
        obs = np.zeros(self.observation_space_shape, dtype=np.float32)
        
        # Channel 0: Agent position
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 1
        
        # Channel 1: Walls
        obs[:, :, 1] = self.walls
        
        # Channel 2: Goal position
        obs[self.goal_pos[0], self.goal_pos[1], 2] = 1
        
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step."""
        self.steps_taken += 1
        
        # Movement directions (Up, Right, Down, Left)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = directions[action]
        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        
        # Initialize reward and info
        reward = -0.1  # Small negative reward for each step
        done = False
        info = {}
        
        # Check if move is valid
        if (0 <= new_pos[0] < self.maze_size and 
            0 <= new_pos[1] < self.maze_size and 
            not self.walls[new_pos]):
            self.agent_pos = new_pos
            
            # Check if agent reached goal
            if new_pos == self.goal_pos:
                reward = 10.0
                done = True
                info['success'] = True
        else:
            reward = -1.0  # Penalty for hitting wall
            info['wall_collision'] = True
        
        # Check for timeout
        if self.steps_taken >= self.max_steps:
            done = True
            info['timeout'] = True
            
        return self._get_observation(), reward, done, info

    def render(self) -> None:
        """Render the environment using PyGame."""
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Custom Maze Environment")
        
        # Fill background
        self.screen.fill(self.colors['white'])
        
        # Draw grid lines
        for i in range(self.maze_size + 1):
            pygame.draw.line(
                self.screen,
                self.colors['black'],
                (0, i * self.cell_size),
                (self.window_size, i * self.cell_size)
            )
            pygame.draw.line(
                self.screen,
                self.colors['black'],
                (i * self.cell_size, 0),
                (i * self.cell_size, self.window_size)
            )
        
        # Draw walls
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if self.walls[i, j]:
                    pygame.draw.rect(
                        self.screen,
                        self.colors['gray'],
                        (j * self.cell_size, i * self.cell_size,
                         self.cell_size, self.cell_size)
                    )
        
        # Draw goal
        pygame.draw.rect(
            self.screen,
            self.colors['green'],
            (self.goal_pos[1] * self.cell_size,
             self.goal_pos[0] * self.cell_size,
             self.cell_size, self.cell_size)
        )
        
        # Draw agent
        pygame.draw.circle(
            self.screen,
            self.colors['blue'],
            (self.agent_pos[1] * self.cell_size + self.cell_size // 2,
             self.agent_pos[0] * self.cell_size + self.cell_size // 2),
            self.cell_size // 3
        )
        
        pygame.display.flip()
        self.clock.tick(10)

    def close(self) -> None:
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def save_state(self) -> Dict:
        """Save the current state of the environment."""
        return {
            'walls': self.walls.copy(),
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
            'steps_taken': self.steps_taken
        }

    def load_state(self, state: Dict) -> None:
        """Load a saved environment state."""
        self.walls = state['walls'].copy()
        self.agent_pos = state['agent_pos']
        self.goal_pos = state['goal_pos']
        self.steps_taken = state['steps_taken']