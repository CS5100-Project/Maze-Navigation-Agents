import numpy as np
from collections import defaultdict, deque
import pickle
from typing import Dict, Tuple, List
import random
import os


class QLearningAgent:
    """
    Q-Learning agent for maze navigation.

    Features:
    - Epsilon-greedy exploration
    - Decaying learning rate
    - Experience replay buffer
    - Saving/loading Q-table
    - Training metrics tracking

    Args:
        action_space (int): Number of possible actions
        learning_rate (float): Initial learning rate
        discount_factor (float): Reward discount factor
        epsilon (float): Initial exploration rate
        epsilon_min (float): Minimum exploration rate
        epsilon_decay (float): Decay rate for epsilon
        buffer_size (int): Size of experience replay buffer
        batch_size (int): Number of experiences to sample in each learning step
    """

    def __init__(
        self,
        action_space: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
    ):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Initialize Q-table as a defaultdict to handle new states automatically
        self.q_table = defaultdict(lambda: np.zeros(self.action_space))

        # Experience replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=buffer_size)

        # Metrics for tracking learning progress
        self.training_rewards = []
        self.training_steps = []
        self.episode_rewards = 0
        self.total_steps = 0
        self.episodes_completed = 0

        # Best score tracking for model saving
        self.best_average_reward = float("-inf")

    def _state_to_key(self, state: np.ndarray) -> tuple:
        """
        Convert state array to hashable tuple for Q-table lookup.

        Args:
            state (np.ndarray): The state to convert

        Returns:
            tuple: Hashable representation of the state
        """
        # Extract agent and goal positions from the state channels
        agent_pos = tuple(np.argwhere(state[:, :, 0])[0])
        goal_pos = tuple(np.argwhere(state[:, :, 2])[0])

        # Convert walls to tuple for hashability
        walls = tuple(map(tuple, state[:, :, 1]))

        # Include hazards if present (channel 3)
        if state.shape[2] > 3:
            hazards = tuple(map(tuple, state[:, :, 3]))
            return (agent_pos, goal_pos, walls, hazards)

        return (agent_pos, goal_pos, walls)

    def choose_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Choose action using epsilon-greedy policy.

        Args:
            state (np.ndarray): Current environment state
            eval_mode (bool): If True, always choose best action

        Returns:
            int: Chosen action
        """
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)

        state_key = self._state_to_key(state)
        return int(np.argmax(self.q_table[state_key]))

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode is done
        """
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.episode_rewards += reward
        self.total_steps += 1

    def learn(self) -> None:
        """
        Update Q-values based on a batch of experiences from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch of experiences
        batch = random.sample(self.replay_buffer, self.batch_size)

        for state, action, reward, next_state, done in batch:
            state_key = self._state_to_key(state)
            next_state_key = self._state_to_key(next_state)

            # Q-learning update
            best_next_action = np.argmax(self.q_table[next_state_key])
            td_target = reward + (
                0
                if done
                else self.gamma * self.q_table[next_state_key][best_next_action]
            )
            td_error = td_target - self.q_table[state_key][action]

            self.q_table[state_key][action] += self.lr * td_error

        # Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def end_episode(self) -> Tuple[float, int]:
        """
        End the current episode and return metrics.

        Returns:
            Tuple[float, int]: Episode total reward and number of steps
        """
        self.episodes_completed += 1
        total_reward = self.episode_rewards
        self.training_rewards.append(total_reward)
        self.training_steps.append(self.total_steps)

        # Update best score if necessary
        if len(self.training_rewards) >= 100:  # Use 100-episode moving average
            avg_reward = np.mean(self.training_rewards[-100:])
            if avg_reward > self.best_average_reward:
                self.best_average_reward = avg_reward

        # Reset episode-specific metrics
        self.episode_rewards = 0

        return total_reward, self.total_steps

    def save(self, filepath: str) -> None:
        """
        Save the Q-table and training metrics.

        Args:
            filepath (str): Path to save the agent's state
        """
        save_dict = {
            "q_table": dict(self.q_table),  # Convert defaultdict to regular dict
            "epsilon": self.epsilon,
            "training_rewards": self.training_rewards,
            "training_steps": self.training_steps,
            "total_steps": self.total_steps,
            "episodes_completed": self.episodes_completed,
            "best_average_reward": self.best_average_reward,
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)

    def load(self, filepath: str) -> None:
        """
        Load the Q-table and training metrics.

        Args:
            filepath (str): Path to load the agent's state from
        """
        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)

        # Convert regular dict back to defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.action_space))
        self.q_table.update(save_dict["q_table"])

        self.epsilon = save_dict["epsilon"]
        self.training_rewards = save_dict["training_rewards"]
        self.training_steps = save_dict["training_steps"]
        self.total_steps = save_dict["total_steps"]
        self.episodes_completed = save_dict["episodes_completed"]
        self.best_average_reward = save_dict["best_average_reward"]

    def get_metrics(self) -> Dict:
        """
        Get current training metrics.

        Returns:
            Dict: Dictionary containing various training metrics
        """
        return {
            "total_episodes": self.episodes_completed,
            "total_steps": self.total_steps,
            "current_epsilon": self.epsilon,
            "average_reward_last_100": (
                np.mean(self.training_rewards[-100:]) if self.training_rewards else 0
            ),
            "best_average_reward": self.best_average_reward,
            "q_table_size": len(self.q_table),
            "buffer_size": len(self.replay_buffer),
        }

    def decay_learning_rate(self, decay_factor: float = 0.995) -> None:
        """
        Decay the learning rate.

        Args:
            decay_factor (float): Factor to multiply learning rate by
        """
        self.lr *= decay_factor

    def reset_metrics(self) -> None:
        """Reset all training metrics."""
        self.training_rewards = []
        self.training_steps = []
        self.episode_rewards = 0
        self.total_steps = 0
        self.episodes_completed = 0
        self.best_average_reward = float("-inf")
