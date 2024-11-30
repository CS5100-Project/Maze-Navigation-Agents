import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from custom_version.maze_env import MazeEnv
from custom_version.q_learning_agent import QLearningAgent
from utils.visualizer import TrainingVisualizer


class ExperimentManager:
    def __init__(
        self,
        env_config: Dict,
        agent_config: Dict,
        training_config: Dict,
        exp_name: str = None,
    ):
        # Create directories
        self.setup_directories()

        # Set experiment name
        self.exp_name = exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create environment and agent
        self.env = MazeEnv(**env_config)
        self.agent = QLearningAgent(action_space=self.env.action_space, **agent_config)

        # Store configurations
        self.env_config = env_config
        self.agent_config = agent_config
        self.training_config = training_config

        # Initialize visualizer
        self.visualizer = TrainingVisualizer()

    def setup_directories(self) -> None:
        """Create necessary directories for saving results."""
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Get version name from class location
        version = (
            "gym_version"
            if "gym_version" in os.path.abspath(__file__)
            else "custom_version"
        )

        # Define directories relative to project root
        self.dirs = {
            "checkpoints": os.path.join(
                project_root, "experiments", version, "checkpoints"
            ),
            "plots": os.path.join(project_root, "experiments", version, "plots"),
            "configs": os.path.join(project_root, "experiments", version, "configs"),
            "logs": os.path.join(project_root, "experiments", version, "logs"),
        }

        # Create each directory
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def train(self) -> None:
        max_episodes = self.training_config["max_episodes"]
        eval_interval = self.training_config["eval_interval"]
        save_interval = self.training_config["save_interval"]

        print("\nStarting training (Custom Version)...")
        print(f"Training for {max_episodes} episodes")

        start_time = time.time()

        for episode in range(1, max_episodes + 1):
            episode_reward, episode_length, success = self.run_episode(training=True)

            # Update visualizer and get metrics
            metrics = self.visualizer.update(episode_reward, episode_length, success)

            if episode % 10 == 0:
                print(
                    f"Episode {episode}/{max_episodes} - "
                    f"Reward: {episode_reward:.2f} - "
                    f"Steps: {episode_length} - "
                    f"Success Rate: {metrics['success_rate']:.2%} - "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )

            if episode % eval_interval == 0:
                self.evaluate()

            if episode % save_interval == 0:
                self.save_checkpoint(episode)
                self.visualizer.plot_training_curves(
                    save_path=os.path.join(
                        self.dirs["plots"], f"{self.exp_name}_training_curves.png"
                    ),
                    implementation="custom",
                )

            if metrics["success_rate"] >= 0.95 and episode >= 100:
                print("\nEarly stopping: 95% success rate achieved!")
                break

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")

        self.evaluate()
        self.save_checkpoint("final")
        self.visualizer.plot_training_curves(
            save_path=os.path.join(
                self.dirs["plots"], f"{self.exp_name}_training_curves.png"
            ),
            implementation="custom",
        )
        self.visualizer.save_metrics(
            save_path=os.path.join(self.dirs["logs"], f"{self.exp_name}_metrics.npy"),
            implementation="custom",
        )

    def run_episode(self, training: bool = True) -> Tuple[float, int, bool]:
        state = self.env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            action = self.agent.choose_action(state, eval_mode=not training)
            next_state, reward, done, info = self.env.step(action)

            if training:
                self.agent.store_experience(state, action, reward, next_state, done)
                self.agent.learn()

            episode_reward += reward
            state = next_state
            step += 1

            if not training:
                self.env.render()  # This will need rendering support in custom maze_env

        success = info.get("success", False)
        return episode_reward, step, success

    def evaluate(self, num_episodes: int = 10) -> None:
        print("\nStarting evaluation...")
        eval_rewards = []
        eval_lengths = []
        success_count = 0

        for episode in range(num_episodes):
            reward, length, success = self.run_episode(training=False)
            eval_rewards.append(reward)
            eval_lengths.append(length)
            if success:
                success_count += 1

        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        success_rate = success_count / num_episodes

        print(f"\nEvaluation results over {num_episodes} episodes:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Length: {avg_length:.2f}")
        print(f"Success Rate: {success_rate:.2%}\n")

    def save_checkpoint(self, episode) -> None:
        checkpoint_path = os.path.join(
            self.dirs["checkpoints"], f"{self.exp_name}_custom_episode_{episode}.pkl"
        )
        self.agent.save(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def main():
    env_config = {
        "maze_size": 4,
        "num_obstacles": 1,
        "max_steps": 50,
    }

    agent_config = {
        "learning_rate": 0.6,
        "discount_factor": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.97,
        "buffer_size": 5000,
        "batch_size": 32,
    }

    training_config = {"max_episodes": 1000, "eval_interval": 20, "save_interval": 100}

    experiment = ExperimentManager(
        env_config=env_config,
        agent_config=agent_config,
        training_config=training_config,
    )

    experiment.train()


if __name__ == "__main__":
    main()
