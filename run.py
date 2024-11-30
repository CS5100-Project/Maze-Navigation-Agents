import multiprocessing
import os
import subprocess
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def run_implementation(version):
    """Run a specific implementation."""
    print(f"Starting {version} training...")
    subprocess.run(["python", f"{version}/train.py"])
    print(f"Completed {version} training")


def run_training_parallel():
    """Run both implementations in parallel."""
    # Create processes
    gym_process = multiprocessing.Process(
        target=run_implementation, args=("gym_version",)
    )
    custom_process = multiprocessing.Process(
        target=run_implementation, args=("custom_version",)
    )

    # Start both processes
    start_time = time.time()
    gym_process.start()
    custom_process.start()

    # Wait for both to complete
    gym_process.join()
    custom_process.join()

    training_time = time.time() - start_time
    print(f"\nBoth implementations completed in {training_time:.2f} seconds")

    # Get most recent results for each version
    gym_timestamp = sorted(os.listdir("experiments/gym_version/logs"))[-1].split(
        "_metrics"
    )[0]
    custom_timestamp = sorted(os.listdir("experiments/custom_version/logs"))[-1].split(
        "_metrics"
    )[0]

    return gym_timestamp, custom_timestamp


def load_metrics(timestamp, version):
    """Load metrics from numpy files."""
    metrics_path = f"experiments/{version}/logs/{timestamp}_metrics.npy"
    return np.load(metrics_path, allow_pickle=True).item()


def plot_comparison(gym_metrics, custom_metrics, save_path="experiments/comparison"):
    """Create comparison plots."""
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot settings
    plt.style.use("default")
    colors = {"gym": "#1f77b4", "custom": "#2ca02c"}

    # 1. Average Reward Comparison
    plt.figure(figsize=(12, 6))
    window = 50  # Rolling average window

    gym_rewards = np.array(gym_metrics["episode_rewards"])
    custom_rewards = np.array(custom_metrics["episode_rewards"])

    # Calculate rolling averages
    gym_avg = np.convolve(gym_rewards, np.ones(window) / window, mode="valid")
    custom_avg = np.convolve(custom_rewards, np.ones(window) / window, mode="valid")

    plt.plot(gym_avg, label="Gym Version", color=colors["gym"])
    plt.plot(custom_avg, label="Custom Version", color=colors["custom"])
    plt.title("Average Reward Comparison")
    plt.xlabel("Episode")
    plt.ylabel(f"Average Reward (over {window} episodes)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{timestamp}_reward_comparison.png")
    plt.close()

    # 2. Episode Length Comparison
    plt.figure(figsize=(12, 6))
    gym_lengths = np.array(gym_metrics["episode_lengths"])
    custom_lengths = np.array(custom_metrics["episode_lengths"])

    gym_len_avg = np.convolve(gym_lengths, np.ones(window) / window, mode="valid")
    custom_len_avg = np.convolve(custom_lengths, np.ones(window) / window, mode="valid")

    plt.plot(gym_len_avg, label="Gym Version", color=colors["gym"])
    plt.plot(custom_len_avg, label="Custom Version", color=colors["custom"])
    plt.title("Episode Length Comparison")
    plt.xlabel("Episode")
    plt.ylabel(f"Average Length (over {window} episodes)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{timestamp}_length_comparison.png")
    plt.close()

    # 3. Success Rate Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(gym_metrics["success_rates"], label="Gym Version", color=colors["gym"])
    plt.plot(
        custom_metrics["success_rates"], label="Custom Version", color=colors["custom"]
    )
    plt.title("Success Rate Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1])
    plt.savefig(f"{save_path}/{timestamp}_success_comparison.png")
    plt.close()

    # Print statistical comparison
    print("\nStatistical Comparison:")
    print("\nAverage Rewards:")
    print(f"Gym Version: {np.mean(gym_rewards):.2f} ± {np.std(gym_rewards):.2f}")
    print(
        f"Custom Version: {np.mean(custom_rewards):.2f} ± {np.std(custom_rewards):.2f}"
    )

    print("\nAverage Episode Lengths:")
    print(f"Gym Version: {np.mean(gym_lengths):.2f} ± {np.std(gym_lengths):.2f}")
    print(
        f"Custom Version: {np.mean(custom_lengths):.2f} ± {np.std(custom_lengths):.2f}"
    )

    print("\nFinal Success Rates:")
    print(f"Gym Version: {gym_metrics['success_rates'][-1]:.2%}")
    print(f"Custom Version: {custom_metrics['success_rates'][-1]:.2%}")

    # Save numerical results
    results = {
        "gym_version": {
            "avg_reward": float(np.mean(gym_rewards)),
            "std_reward": float(np.std(gym_rewards)),
            "avg_length": float(np.mean(gym_lengths)),
            "std_length": float(np.std(gym_lengths)),
            "final_success_rate": float(gym_metrics["success_rates"][-1]),
        },
        "custom_version": {
            "avg_reward": float(np.mean(custom_rewards)),
            "std_reward": float(np.std(custom_rewards)),
            "avg_length": float(np.mean(custom_lengths)),
            "std_length": float(np.std(custom_lengths)),
            "final_success_rate": float(custom_metrics["success_rates"][-1]),
        },
    }

    np.save(f"{save_path}/{timestamp}_comparison_results.npy", results)


def main():
    # Run both implementations in parallel
    gym_timestamp, custom_timestamp = run_training_parallel()

    # Load metrics
    gym_metrics = load_metrics(gym_timestamp, "gym_version")
    custom_metrics = load_metrics(custom_timestamp, "custom_version")

    # Create comparison plots and analysis
    plot_comparison(gym_metrics, custom_metrics)


if __name__ == "__main__":
    # Required for Windows
    multiprocessing.freeze_support()
    main()
