import os
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class TrainingVisualizer:
    """
    Handles visualization of training metrics for both Gym and Custom implementations.
    """

    def __init__(self, save_dir: str = "experiments/plots"):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_rates: List[float] = []
        self.success_window = []

        # Create plots directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # Timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def update(self, reward: float, length: int, success: bool) -> Dict[str, float]:
        """
        Update training metrics after each episode.

        Args:
            reward: Episode total reward
            length: Episode length
            success: Whether episode was successful

        Returns:
            Dict containing current metrics
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        # Update success rate calculation
        self.success_window.append(success)
        if len(self.success_window) > 100:
            self.success_window.pop(0)
        success_rate = sum(self.success_window) / len(self.success_window)
        self.success_rates.append(success_rate)

        return {"reward": reward, "length": length, "success_rate": success_rate}

    def plot_training_curves(self, implementation: str = "gym") -> None:
        """
        Plot and save training curves for rewards, lengths, and success rates.

        Args:
            implementation: String indicating which implementation ("gym" or "custom")
        """
        plt.style.use("default")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Plot episode rewards
        ax1.plot(self.episode_rewards, label="Episode Reward", color="blue")
        ax1.set_title(f"{implementation.capitalize()} Implementation - Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.grid(True)

        # Add moving average for rewards
        window_size = 50
        rewards_ma = np.convolve(
            self.episode_rewards, np.ones(window_size) / window_size, mode="valid"
        )
        ax1.plot(
            range(window_size - 1, len(self.episode_rewards)),
            rewards_ma,
            label=f"{window_size}-Episode Moving Average",
            color="red",
            linestyle="--",
        )
        ax1.legend()

        # Plot episode lengths
        ax2.plot(self.episode_lengths, label="Episode Length")
        ax2.set_title(f"{implementation.capitalize()} Implementation - Episode Lengths")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.grid(True)

        # Add moving average for lengths
        lengths_ma = np.convolve(
            self.episode_lengths, np.ones(window_size) / window_size, mode="valid"
        )
        ax2.plot(
            range(window_size - 1, len(self.episode_lengths)),
            lengths_ma,
            label=f"{window_size}-Episode Moving Average",
            color="red",
            linestyle="--",
        )
        ax2.legend()

        # Plot success rate
        ax3.plot(self.success_rates, label="Success Rate")
        ax3.set_title(f"{implementation.capitalize()} Implementation - Success Rate")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Success Rate")
        ax3.set_ylim([0, 1])
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{self.timestamp}_{implementation}_training_curves.png"
        )
        plt.close()

    def plot_comparison(self, custom_metrics: Dict, gym_metrics: Dict) -> None:
        """
        Plot comparison between Custom and Gym implementations.

        Args:
            custom_metrics: Dictionary containing metrics from custom implementation
            gym_metrics: Dictionary containing metrics from gym implementation
        """
        plt.style.use("default")
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Success Rate Comparison
        axes[0, 0].plot(custom_metrics["success_rates"], label="Custom")
        axes[0, 0].plot(gym_metrics["success_rates"], label="Gym")
        axes[0, 0].set_title("Success Rate Comparison")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Success Rate")
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Average Reward Comparison
        axes[0, 1].plot(custom_metrics["rewards"], label="Custom")
        axes[0, 1].plot(gym_metrics["rewards"], label="Gym")
        axes[0, 1].set_title("Average Reward Comparison")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Average Reward")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Episode Length Comparison
        axes[1, 0].plot(custom_metrics["lengths"], label="Custom")
        axes[1, 0].plot(gym_metrics["lengths"], label="Gym")
        axes[1, 0].set_title("Episode Length Comparison")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Steps")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Training Time Comparison (if available)
        if "training_times" in custom_metrics and "training_times" in gym_metrics:
            axes[1, 1].bar(
                ["Custom", "Gym"],
                [custom_metrics["training_times"], gym_metrics["training_times"]],
            )
            axes[1, 1].set_title("Training Time Comparison")
            axes[1, 1].set_ylabel("Time (seconds)")
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{self.timestamp}_implementation_comparison.png")
        plt.close()

    def save_metrics(self, implementation: str) -> None:
        """
        Save metrics to a file.

        Args:
            implementation: String indicating which implementation ("gym" or "custom")
        """
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "success_rates": self.success_rates,
            "final_success_rate": self.success_rates[-1] if self.success_rates else 0,
            "average_reward": np.mean(self.episode_rewards),
            "average_length": np.mean(self.episode_lengths),
        }

        np.save(
            f"{self.save_dir}/{self.timestamp}_{implementation}_metrics.npy", metrics
        )
