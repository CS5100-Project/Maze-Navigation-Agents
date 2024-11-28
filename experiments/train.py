import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
from typing import Dict, List, Tuple
import logging
import time

from environments.maze_env import MazeEnv
from agents.q_learning_agent import QLearningAgent


class ExperimentManager:
    """
    Manages the training process, logging, and visualization.
    """

    def __init__(
        self,
        env_config: Dict,
        agent_config: Dict,
        training_config: Dict,
        exp_name: str = None,
    ):
        # Create directories first
        self.setup_directories()

        # Then initialize logging
        self.setup_logging(exp_name)

        # Create environment and agent
        self.env = MazeEnv(**env_config)
        self.agent = QLearningAgent(
            action_space=self.env.action_space.n, **agent_config
        )

        # Store configurations
        self.env_config = env_config
        self.agent_config = agent_config
        self.training_config = training_config

        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_rate: List[float] = []

    def setup_directories(self) -> None:
        """Create necessary directories for saving results."""
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Define directories relative to project root
        self.dirs = {
            "logs": os.path.join(project_root, "experiments", "logs"),
            "checkpoints": os.path.join(project_root, "experiments", "checkpoints"),
            "plots": os.path.join(project_root, "experiments", "plots"),
            "configs": os.path.join(project_root, "experiments", "configs"),
        }

        # Create each directory
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def setup_logging(self, exp_name: str = None) -> None:
        """Setup logging configuration."""
        if exp_name is None:
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = exp_name

        # Create log file path using the logs directory
        log_file = os.path.join(self.dirs["logs"], f"{exp_name}.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
        )
        self.logger = logging.getLogger(__name__)

    def save_configs(self) -> None:
        """Save all configurations to a file."""
        config = {
            "environment": self.env_config,
            "agent": self.agent_config,
            "training": self.training_config,
        }
        config_file = os.path.join(self.dirs["configs"], f"{self.exp_name}.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

    def save_checkpoint(self, episode) -> None:
        """Save a checkpoint of the agent and training progress."""
        checkpoint_path = os.path.join(
            self.dirs["checkpoints"], f"{self.exp_name}_episode_{episode}.pkl"
        )
        self.agent.save(checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def plot_training_curves(self) -> None:
        """Plot and save training curves."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        # Plot rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")

        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title("Episode Lengths")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")

        # Plot success rate
        ax3.plot(self.success_rate)
        ax3.set_title("Success Rate (100-episode moving average)")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Success Rate")
        ax3.set_ylim([0, 1])

        plt.tight_layout()
        plot_file = os.path.join(
            self.dirs["plots"], f"{self.exp_name}_training_curves.png"
        )
        plt.savefig(plot_file)
        plt.close()

    def train(self) -> None:
        """Main training loop."""
        max_episodes = self.training_config["max_episodes"]
        eval_interval = self.training_config["eval_interval"]
        save_interval = self.training_config["save_interval"]

        # Save initial configurations
        self.save_configs()

        self.logger.info("Starting training...")
        self.logger.info(f"Training for {max_episodes} episodes")

        start_time = time.time()
        success_window = []

        for episode in range(1, max_episodes + 1):
            episode_reward, episode_length, success = self.run_episode(training=True)

            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            success_window.append(success)

            # Calculate success rate over last 100 episodes
            if len(success_window) > 100:
                success_window.pop(0)
            success_rate = sum(success_window) / len(success_window)
            self.success_rate.append(success_rate)

            # Log progress
            if episode % 10 == 0:
                self.logger.info(
                    f"Episode {episode}/{max_episodes} - "
                    f"Reward: {episode_reward:.2f} - "
                    f"Steps: {episode_length} - "
                    f"Success Rate: {success_rate:.2%} - "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )

            # Evaluation
            if episode % eval_interval == 0:
                self.evaluate()

            # Save checkpoint
            if episode % save_interval == 0:
                self.save_checkpoint(episode)

            # Early stopping if success rate is high enough
            if success_rate >= 0.95 and len(success_window) == 100:
                self.logger.info("Early stopping: 95% success rate achieved!")
                break

        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")

        # Final evaluation and saving
        self.evaluate()
        self.save_checkpoint("final")
        self.plot_training_curves()

    def run_episode(self, training: bool = True) -> Tuple[float, int, bool]:
        """Run a single episode."""
        state = self.env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            # Choose action
            action = self.agent.choose_action(state, eval_mode=not training)

            # Take action
            next_state, reward, done, info = self.env.step(action)

            if training:
                # Store experience and learn
                self.agent.store_experience(state, action, reward, next_state, done)
                self.agent.learn()

            episode_reward += reward
            state = next_state
            step += 1

            # Render if in evaluation mode
            if not training:
                self.env.render()

        success = info.get("success", False)
        return episode_reward, step, success

    def evaluate(self, num_episodes: int = 10) -> None:
        """Evaluate the agent's performance."""
        eval_rewards = []
        eval_lengths = []
        success_count = 0

        self.logger.info("\nStarting evaluation...")

        for episode in range(num_episodes):
            reward, length, success = self.run_episode(training=False)
            eval_rewards.append(reward)
            eval_lengths.append(length)
            if success:
                success_count += 1

        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        success_rate = success_count / num_episodes

        self.logger.info(
            f"Evaluation results over {num_episodes} episodes:\n"
            f"Average Reward: {avg_reward:.2f}\n"
            f"Average Length: {avg_length:.2f}\n"
            f"Success Rate: {success_rate:.2%}\n"
        )

    def save_checkpoint(self, episode) -> None:
        """Save a checkpoint of the agent and training progress."""
        checkpoint_path = (
            f"experiments/checkpoints/{self.exp_name}_episode_{episode}.pkl"
        )
        self.agent.save(checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def plot_training_curves(self) -> None:
        """Plot and save training curves."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        # Plot rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")

        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title("Episode Lengths")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")

        # Plot success rate
        ax3.plot(self.success_rate)
        ax3.set_title("Success Rate (100-episode moving average)")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Success Rate")
        ax3.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(f"experiments/plots/{self.exp_name}_training_curves.png")
        plt.close()


def main():
    # Configuration
    env_config = {
        "maze_size": 4,
        "num_obstacles": 1,
        "max_steps": 50,
        "enable_moving_walls": False,  # Start with simpler environment
        "dynamic_goal": False,
        "random_hazards": False,
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

    training_config = {
        "max_episodes": 1000, 
        "eval_interval": 20, 
        "save_interval": 100
    }

    # Create and run experiment
    experiment = ExperimentManager(
        env_config=env_config,
        agent_config=agent_config,
        training_config=training_config,
    )

    experiment.train()


if __name__ == "__main__":
    main()
