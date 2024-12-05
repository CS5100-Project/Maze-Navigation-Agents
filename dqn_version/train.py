# train_dqn.py
import os
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import torch

from gym_version.maze_env import MazeEnv
from dqn_agent import DQNAgent


class DQNExperimentManager:
    def __init__(
        self,
        env_config: dict,
        agent_config: dict,
        training_config: dict,
        exp_name: str = None,
    ):
        self.setup_directories()
        self.exp_name = exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create environment and agent
        self.env = MazeEnv(**env_config)
        state_dim = self.env.observation_space.shape[0]  # Should be 2 for a 2D state vector

        # Obtain a sample observation
        sample_state = self.env.reset()
        print("Sample state shape:", sample_state.shape)
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space)  # Should display the action space details

        # Define state_shape based on sample_state
        # Rearrange to (channels, height, width)
        state_shape = (sample_state.shape[2], sample_state.shape[0], sample_state.shape[1])  # (4, 8, 8)
        print(f"state_shape after rearrangement: {state_shape}")

        self.agent = DQNAgent(
            state_shape=state_shape,
            action_size=self.env.action_space.n,
            **agent_config,
        )

        # Store configurations
        self.env_config = env_config
        self.agent_config = agent_config
        self.training_config = training_config

        # Metrics
        self.training_rewards = []
        self.training_steps = []
        self.success_rates = []

    def setup_directories(self):
        self.dirs = {
            "checkpoints": os.path.join("checkpoints"),
            "plots": os.path.join("plots"),
            "logs": os.path.join("logs"),
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def train(self):
        max_episodes = self.training_config["max_episodes"]
        eval_interval = self.training_config.get("eval_interval", 20)
        save_interval = self.training_config.get("save_interval", 100)

        print("\nStarting DQN training...")
        print(f"Training for {max_episodes} episodes")

        start_time = time.time()

        for episode in range(1, max_episodes + 1):
            episode_reward, episode_length, success = self.run_episode(training=True)
            self.training_rewards.append(episode_reward)
            self.training_steps.append(episode_length)

            if episode % 10 == 0:
                avg_reward = np.mean(self.training_rewards[-10:])
                print(
                    f"Episode {episode}/{max_episodes} - "
                    f"Average Reward: {avg_reward:.2f} - "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )

            if episode % eval_interval == 0:
                success_rate = self.evaluate()
                self.success_rates.append(success_rate)

            if episode % save_interval == 0:
                self.save_checkpoint(episode)

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")

        self.evaluate()
        self.save_checkpoint("final")

    def run_episode(self, training=True) -> Tuple[float, int, bool]:
        state = self.env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            # Ensure state is in (channels, height, width) format
            if state.shape != self.agent.state_shape:
                state_processed = np.transpose(state, (2, 0, 1))  # From (H, W, C) to (C, H, W)
            else:
                state_processed = state

            action = self.agent.select_action(state_processed, eval_mode=not training)
            next_state, reward, done, info = self.env.step(action)

            # Ensure next_state is in (channels, height, width) format
            if next_state.shape != self.agent.state_shape:
                next_state_processed = np.transpose(next_state, (2, 0, 1))  # From (H, W, C) to (C, H, W)
            else:
                next_state_processed = next_state

            # Rearrange state to (channels, height, width)
            #state_processed = np.transpose(state, (2, 0, 1))
            # Normalize if necessary (e.g., if state values are between 0 and 1, skip normalization)
            # If state values are between 0 and 255 (typical for images), normalize by dividing by 255.0
            #state_processed = state_processed / 255.0
            # Convert to float32 if necessary
            #state_processed = state_processed.astype(np.float32)

            
            #next_state_processed = np.transpose(next_state, (2, 0, 1))
            #next_state_processed = next_state_processed / 255.0

            if training:
                self.agent.remember(state_processed, action, reward, next_state_processed, done)
                self.agent.learn()

            episode_reward += reward
            state = next_state
            step += 1

        return episode_reward, step, info.get("success", False)

    def evaluate(self, num_episodes=10):
        print("\nEvaluating...")
        total_rewards = []
        successes = 0

        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                state_processed = np.transpose(state, (2, 0, 1))
                action = self.agent.select_action(state_processed, eval_mode=True)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)
            if info.get("success", False):
                successes += 1

        avg_reward = np.mean(total_rewards)
        success_rate = successes / num_episodes
        print(f"Average Reward: {avg_reward:.2f} - Success Rate: {success_rate:.2%}")
        return success_rate

    def save_checkpoint(self, episode):
        checkpoint_path = os.path.join(
            self.dirs["checkpoints"], f"{self.exp_name}_episode_{episode}.pt"
        )
        self.agent.save(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def main():
    env_config = {
        "maze_size": 8,
        "num_obstacles": 10,
        "max_steps": 200,
        "enable_moving_walls": False,
        "dynamic_goal": False,
        "random_hazards": False,
    }

    agent_config = {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9999,
        "buffer_size": 10000,
        "batch_size": 64,
        "target_update_freq": 1000,
    }

    training_config = {
        "max_episodes": 1000,
        "eval_interval": 20,
        "save_interval": 100,
    }

    experiment = DQNExperimentManager(
        env_config=env_config,
        agent_config=agent_config,
        training_config=training_config,
    )

    experiment.train()


if __name__ == "__main__":
    main()