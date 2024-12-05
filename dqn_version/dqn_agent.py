# dqn_agent.py
import os
import random
import traceback
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNAgent:
    """
    DQN agent for maze navigation.

    Features:
    - Neural network function approximation
    - Experience replay buffer
    - Target network for stable training
    - Epsilon-greedy exploration with decay
    - Saving/loading model checkpoints
    - Training metrics tracking

    Args:
        state_shape (tuple): Shape of the state observation
        action_size (int): Number of possible actions
        learning_rate (float): Learning rate for optimizer
        gamma (float): Discount factor
        epsilon (float): Initial exploration rate
        epsilon_min (float): Minimum exploration rate
        epsilon_decay (float): Decay rate for epsilon
        buffer_size (int): Size of experience replay buffer
        batch_size (int): Number of experiences to sample in each learning step
        target_update_freq (int): Frequency of updating the target network
        device (str): Device to run the computations ('cpu' or 'cuda')
    """

    def __init__(
        self,
        state_shape: tuple,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = None,
    ):
        self.state_shape = state_shape
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        print(f"state_shape in DQNAgent: {self.state_shape}")
        self.policy_net = DQNetwork(
            input_shape=self.state_shape, num_actions=action_size
        ).to(self.device)
        self.target_net = DQNetwork(
            input_shape=self.state_shape, num_actions=action_size
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

        # Training counters
        self.steps_done = 0
        self.episodes_done = 0

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        # Ensure state and next_state are in (channels, height, width) format
        if state.shape != self.state_shape:
            state = np.transpose(state, (2, 0, 1))  # From (H, W, C) to (C, H, W)
        if next_state.shape != self.state_shape:
            next_state = np.transpose(
                next_state, (2, 0, 1)
            )  # From (H, W, C) to (C, H, W)
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, eval_mode=False):
        """Select action using epsilon-greedy policy."""
        # Ensure state is in (channels, height, width) format
        if state.shape != self.state_shape:
            state = np.transpose(state, (2, 0, 1))  # From (H, W, C) to (C, H, W)
        state = (
            torch.FloatTensor(state).unsqueeze(0).to(self.device)
        )  # Add batch dimension
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.argmax(dim=1).item()
        else:
            action = random.randrange(self.action_size)
        return action

    def learn(self):
        """Sample a batch from memory and update the network."""
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = np.array(states)
        next_states = np.array(next_states)

        # Ensure states are in (batch_size, channels, height, width)
        if states.shape[1:] != self.state_shape:
            states = np.transpose(
                states, (0, 3, 1, 2)
            )  # From (N, H, W, C) to (N, C, H, W)
        if next_states.shape[1:] != self.state_shape:
            next_states = np.transpose(
                next_states, (0, 3, 1, 2)
            )  # From (N, H, W, C) to (N, C, H, W)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q(s_t, a)
        q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1

    def save(self, filepath):
        """Save the model and optimizer state."""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filepath,
        )

    def load(self, filepath):
        """Load the model and optimizer state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)


class DQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNetwork, self).__init__()
        c, h, w = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=c, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        conv_out_size = 64 * (h // 4) * (w // 4)

        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals
