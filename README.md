# Reinforcement Learning Maze Navigation: OpenAI Gym vs Custom Implementation

## Project Overview
This project implements and compares two approaches to solving a maze navigation problem using Q-learning:
1. OpenAI Gym-based implementation
2. Custom environment implementation

The goal is to train an agent to navigate through a maze, finding the optimal path from start to goal while avoiding obstacles.

## Project Structure
```
maze_rl/
├── gym_version/
│   ├── maze_env.py        # Gym-based maze environment
│   ├── q_learning_agent.py # Q-learning agent for gym version
│   └── train.py           # Training script for gym version
│
├── custom_version/
│   ├── maze_env.py        # Custom maze environment
│   ├── q_learning_agent.py # Q-learning agent for custom version
│   └── train.py           # Training script for custom version
│
├── utils/
│   └── visualizer.py      # Visualization utilities
│
├── experiments/
│   ├── gym_version/       # Results from gym implementation
│   │   ├── checkpoints/
│   │   ├── plots/
│   │   ├── configs/
│   │   └── logs/
│   ├── custom_version/    # Results from custom implementation
│   │   ├── checkpoints/
│   │   ├── plots/
│   │   ├── configs/
│   │   └── logs/
│   └── comparison/        # Comparative analysis results
│
├── setup.py               # Project setup script
├── compare_implementations.py  # Comparison script
└── requirements.txt       # Project dependencies
```

## Features
- **Environment Features**:
  - Configurable maze size
  - Random obstacle generation
  - Visual rendering using Pygame
  - Customizable reward structure
  - Step-based constraints

- **Agent Features**:
  - Q-learning implementation
  - Experience replay
  - Epsilon-greedy exploration
  - Configurable hyperparameters
  - Checkpoint saving/loading

- **Analysis Features**:
  - Training metrics visualization
  - Performance comparison
  - Real-time training monitoring
  - Statistical analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/maze_rl.git
cd maze_rl
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up project structure:
```bash
python setup.py
```

## Usage

### Running Individual Implementations

1. OpenAI Gym Version:
```bash
python gym_version/train.py
```

2. Custom Version:
```bash
python custom_version/train.py
```

### Running Both Implementations Simultaneously
```bash
python compare_implementations.py
```

### Configuration

Both versions use the following default configuration:

```python
env_config = {
    "maze_size": 4,
    "num_obstacles": 1,
    "max_steps": 50
}

agent_config = {
    "learning_rate": 0.6,
    "discount_factor": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.97,
    "buffer_size": 5000,
    "batch_size": 32
}

training_config = {
    "max_episodes": 1000,
    "eval_interval": 20,
    "save_interval": 100
}
```

Modify these parameters in the respective `train.py` files to experiment with different settings.

## Results and Analysis

Training results are automatically saved in the following locations:
- Gym version: `experiments/gym_version/`
- Custom version: `experiments/custom_version/`
- Comparative analysis: `experiments/comparison/`

Each run generates:
1. Training plots
2. Checkpoint files
3. Configuration logs
4. Performance metrics

## Comparison Metrics

The comparison script generates:
1. Average reward comparison
2. Episode length comparison
3. Success rate comparison
4. Statistical analysis of performance differences

## Visualization

Both implementations provide real-time visualization using Pygame:
- Blue circle: Agent
- Green square: Goal
- Gray squares: Obstacles
- White space: Valid paths

## Troubleshooting

Common issues and solutions:

1. PyGame window not showing:
   - Ensure Python has access to display
   - Check PyGame installation

2. Training not converging:
   - Adjust learning rate
   - Modify epsilon decay rate
   - Change reward structure

3. Memory issues:
   - Reduce replay buffer size
   - Decrease maze size
   - Lower number of episodes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- OpenAI Gym
- PyGame documentation
- Q-learning research papers
- Reinforcement learning community