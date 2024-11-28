from custom_version.maze_env import MazeEnv
from custom_version.q_learning_agent import QLearningAgent
import numpy as np
import json
import os
from typing import Dict
from typing import List
import matplotlib.pyplot as plt


# class ExperimentManager:
#     def __init__(
#             self,
#             env_config: Dict,
#             agent_config: Dict,
#             training_config: Dict,
#             exp_name: str = None
#     ):
#         self.env = MazeEnv(**env_config)
#         self.agent = QLearningAgent(action_space=self.env.action_space, **agent_config)
#
#         self.env_config = env_config
#         self.agent_config = agent_config
#         self.training_config = training_config
#
#         self.episode_rewards = []
#         self.episode_lengths = []
#         self.success_rate = []
#
#         self.exp_name = exp_name or "experiment"
#         self.setup_directories()
#
#     def setup_directories(self) -> None:
#         self.dirs = {
#             "logs": "experiments/logs",
#             "checkpoints": "experiments/checkpoints",
#             "configs": "experiments/configs",
#         }
#         for dir_path in self.dirs.values():
#             os.makedirs(dir_path, exist_ok=True)
#
#     def save_configs(self) -> None:
#         config = {
#             "environment": self.env_config,
#             "agent": self.agent_config,
#             "training": self.training_config,
#         }
#         config_file = os.path.join(self.dirs["configs"], f"{self.exp_name}_config.json")
#         with open(config_file, "w") as f:
#             json.dump(config, f, indent=4)
#
#     def train(self) -> None:
#         max_episodes = self.training_config["max_episodes"]
#         eval_interval = self.training_config["eval_interval"]
#
#         self.save_configs()
#         print("Starting training...")
#
#         success_window = []
#
#         for episode in range(1, max_episodes + 1):
#             state = self.env.reset()
#             done = False
#
#             while not done:
#                 action = self.agent.choose_action(state)
#                 next_state, reward, done, info = self.env.step(action)
#                 self.agent.store_experience(state, action, reward, next_state, done)
#                 self.agent.learn()
#                 state = next_state
#
#             episode_reward, episode_length = self.agent.end_episode()
#             success = info.get("success", False)
#
#             self.episode_rewards.append(episode_reward)
#             self.episode_lengths.append(episode_length)
#             success_window.append(success)
#
#             if len(success_window) > 100:
#                 success_window.pop(0)
#             success_rate = sum(success_window) / len(success_window)
#             self.success_rate.append(success_rate)
#
#             if episode % 10 == 0:
#                 print(f"Episode {episode}/{max_episodes} - "
#                       f"Reward: {episode_reward:.2f} - "
#                       f"Steps: {episode_length} - "
#                       f"Success Rate: {success_rate:.2%} - "
#                       f"Epsilon: {self.agent.epsilon:.3f}")
#
#             if episode % eval_interval == 0:
#                 self.evaluate()
#
#             if success_rate >= 0.95 and len(success_window) == 100:
#                 print("Early stopping: 95% success rate achieved!")
#                 break
#
#         print("Training completed")
#         self.evaluate()
#
#     def evaluate(self, num_episodes: int = 10) -> None:
#         print("\nStarting evaluation...")
#         eval_rewards = []
#         eval_lengths = []
#         success_count = 0
#
#         for _ in range(num_episodes):
#             state = self.env.reset()
#             done = False
#             episode_reward = 0
#             episode_length = 0
#
#             while not done:
#                 action = self.agent.choose_action(state, eval_mode=True)
#                 state, reward, done, info = self.env.step(action)
#                 episode_reward += reward
#                 episode_length += 1
#
#             eval_rewards.append(episode_reward)
#             eval_lengths.append(episode_length)
#             if info.get("success", False):
#                 success_count += 1
#
#         avg_reward = np.mean(eval_rewards)
#         avg_length = np.mean(eval_lengths)
#         success_rate = success_count / num_episodes
#
#         print(f"Evaluation results over {num_episodes} episodes:")
#         print(f"Average Reward: {avg_reward:.2f}")
#         print(f"Average Length: {avg_length:.2f}")
#         print(f"Success Rate: {success_rate:.2%}")
#
#
# def main():
#     env_config = {
#         "maze_size": 8,
#         "num_obstacles": 5,
#         "max_steps": 200,
#     }
#
#     agent_config = {
#         "learning_rate": 0.1,
#         "discount_factor": 0.95,
#         "epsilon": 1.0,
#         "epsilon_min": 0.01,
#         "epsilon_decay": 0.995,
#         "buffer_size": 10000,
#         "batch_size": 32,
#     }
#
#     training_config = {
#         "max_episodes": 1000,
#         "eval_interval": 100,
#     }
#
#     experiment = ExperimentManager(
#         env_config=env_config,
#         agent_config=agent_config,
#         training_config=training_config,
#     )
#     experiment.train()
#
#
# if __name__ == "__main__":
#     main()


class TrainingVisualizer:
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_rates: List[float] = []
        self.success_window = []

    def update(self, reward: float, length: int, success: bool) -> None:
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        # Update success rate calculation
        self.success_window.append(success)
        if len(self.success_window) > 100:
            self.success_window.pop(0)
        success_rate = sum(self.success_window) / len(self.success_window)
        self.success_rates.append(success_rate)

    def plot_curves(self, save_path: str) -> None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        # Plot episode rewards
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
        ax3.plot(self.success_rates)
        ax3.set_title("Success Rate (100-episode moving average)")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Success Rate")
        ax3.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


class ExperimentManager:
    def __init__(self, env_config: Dict, agent_config: Dict, training_config: Dict):
        self.env = MazeEnv(**env_config)
        self.agent = QLearningAgent(action_space=self.env.action_space, **agent_config)
        self.training_config = training_config
        self.visualizer = TrainingVisualizer()

    def train(self) -> None:
        max_episodes = self.training_config["max_episodes"]

        print("Starting training...")

        for episode in range(1, max_episodes + 1):
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.store_experience(state, action, reward, next_state, done)
                self.agent.learn()

                state = next_state
                episode_reward += reward
                episode_steps += 1

            # Update visualization data
            self.visualizer.update(
                reward=episode_reward,
                length=episode_steps,
                success=info.get("success", False),
            )

            if episode % 10 == 0:
                success_rate = (
                    self.visualizer.success_rates[-1]
                    if self.visualizer.success_rates
                    else 0
                )
                print(
                    f"Episode {episode}/{max_episodes} - "
                    f"Reward: {episode_reward:.2f} - "
                    f"Steps: {episode_steps} - "
                    f"Success Rate: {success_rate:.2%} - "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )

        # Generate and save plots
        self.visualizer.plot_curves("20241119_standalone_training_curves.png")
        print(
            "Training completed. Plots saved to 20241119_standalone_training_curves.png"
        )


def main():
    env_config = {"maze_size": 8, "num_obstacles": 5, "max_steps": 200}

    agent_config = {
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "buffer_size": 10000,
        "batch_size": 32,
    }

    training_config = {"max_episodes": 1000}

    experiment = ExperimentManager(
        env_config=env_config,
        agent_config=agent_config,
        training_config=training_config,
    )
    experiment.train()


if __name__ == "__main__":
    main()
