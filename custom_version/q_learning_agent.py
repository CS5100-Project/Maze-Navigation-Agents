import pickle
import random
from collections import defaultdict, deque
from typing import Dict, Tuple

import numpy as np


class QLearningAgent:
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

        self.q_table = defaultdict(lambda: np.zeros(self.action_space))
        self.replay_buffer = deque(maxlen=buffer_size)

        self.training_rewards = []
        self.training_steps = []
        self.episode_rewards = 0
        self.total_steps = 0
        self.episodes_completed = 0

    def _state_to_key(self, state: np.ndarray) -> tuple:
        agent_pos = tuple(np.argwhere(state[:, :, 0])[0])
        goal_pos = tuple(np.argwhere(state[:, :, 2])[0])
        walls = tuple(map(tuple, state[:, :, 1]))
        return (agent_pos, goal_pos, walls)

    def choose_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
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
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.episode_rewards += reward
        self.total_steps += 1

    def learn(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)

        for state, action, reward, next_state, done in batch:
            state_key = self._state_to_key(state)
            next_state_key = self._state_to_key(next_state)

            best_next_action = np.argmax(self.q_table[next_state_key])
            td_target = reward + (
                0
                if done
                else self.gamma * self.q_table[next_state_key][best_next_action]
            )
            td_error = td_target - self.q_table[state_key][action]
            self.q_table[state_key][action] += self.lr * td_error

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def end_episode(self) -> Tuple[float, int]:
        self.episodes_completed += 1
        total_reward = self.episode_rewards
        self.training_rewards.append(total_reward)
        self.training_steps.append(self.total_steps)
        self.episode_rewards = 0
        return total_reward, self.total_steps

    def get_metrics(self) -> Dict:
        return {
            "total_episodes": self.episodes_completed,
            "total_steps": self.total_steps,
            "current_epsilon": self.epsilon,
            "average_reward_last_100": (
                np.mean(self.training_rewards[-100:]) if self.training_rewards else 0
            ),
            "q_table_size": len(self.q_table),
            "buffer_size": len(self.replay_buffer),
        }

    def save(self, filepath: str) -> None:
        """Save the agent's state."""
        save_dict = {
            "q_table": dict(self.q_table),  # Convert defaultdict to regular dict
            "epsilon": self.epsilon,
            "training_rewards": self.training_rewards,
            "training_steps": self.training_steps,
            "total_steps": self.total_steps,
            "episodes_completed": self.episodes_completed,
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)

    def load(self, filepath: str) -> None:
        """Load the agent's state."""
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
