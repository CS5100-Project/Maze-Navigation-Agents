# Maze Reinforcement Learning Project

This project implements a customizable maze environment and Q-learning agent using OpenAI Gym.

## Features
- Configurable maze size and complexity
- Dynamic elements (moving walls, changing goals)
- Hazards and time constraints
- Q-learning implementation with experience replay
- Visualization support using Pygame

## Setup
1. Clone this repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3. Install requirements:
```bash
pip install -r requirements.txt
```

## Project Structure
- `environments/`: Contains the maze environment implementation
- `agents/`: Contains the Q-learning agent implementation
- `utils/`: Utility functions for visualization and data handling
- `experiments/`: Training scripts and experiments
<!-- - `config/`: Configuration files for different experiments -->