import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


class TrainingVisualizer:
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_rates: List[float] = []
        self.success_window = []

    def update(self, reward: float, length: int, success: bool) -> Dict[str, float]:
        """Update training metrics after each episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        # Update success rate calculation
        self.success_window.append(success)
        if len(self.success_window) > 100:
            self.success_window.pop(0)
        success_rate = sum(self.success_window) / len(self.success_window)
        self.success_rates.append(success_rate)

        return {"reward": reward, "length": length, "success_rate": success_rate}

    def plot_training_curves(self, save_path: str, implementation: str = "gym") -> None:
        """Plot and save training curves."""
        plt.style.use("default")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        # Plot episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title(f"{implementation.capitalize()} Implementation - Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.grid(True)

        # Add moving average for rewards
        window_size = 50
        if len(self.episode_rewards) >= window_size:
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
        ax2.plot(self.episode_lengths)
        ax2.set_title(f"{implementation.capitalize()} Implementation - Episode Lengths")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.grid(True)

        # Add moving average for lengths
        if len(self.episode_lengths) >= window_size:
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
        ax3.plot(self.success_rates)
        ax3.set_title(f"{implementation.capitalize()} Implementation - Success Rate")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Success Rate")
        ax3.set_ylim([0, 1])
        ax3.grid(True)

        plt.tight_layout()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    def save_metrics(self, save_path: str, implementation: str = "gym") -> None:
        """Save metrics to a file."""
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "success_rates": self.success_rates,
            "final_success_rate": self.success_rates[-1] if self.success_rates else 0,
            "average_reward": np.mean(self.episode_rewards),
            "average_length": np.mean(self.episode_lengths),
            "implementation": implementation,
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, metrics)
