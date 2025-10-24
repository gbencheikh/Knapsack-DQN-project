Knapsack DQN Agent

Description

This project implements a Deep Q-Network (DQN) agent to solve the 0/1 Knapsack problem. The agent is trained to select items to maximize the total value without exceeding the knapsack capacity.

Unlike classical approaches, this DQN generalizes across randomly generated instances of varying sizes and capacities. The agent is compared to optimal solutions computed via dynamic programming (DP).

The project provides:

* Random instance generation for knapsack problems

* A modular PyTorch DQN model

* Experience replay for stable learning

* Evaluation routines with comparison to DP optimal solutions

* Visualization of training rewards and evaluation performance

* Single-instance demonstrations for qualitative inspection

Features

* Train a DQN agent on multiple random instances

* Evaluate the agent against DP optimal solutions

* Save and load trained models (knapsack_dqn_model.pt)

* Generate and save training reward plots and evaluation histograms

* Modular codebase ready for Double DQN, Dueling DQN, Prioritized Replay, or multi-objective extensions

Project Structure
```
knapsack_dqn_project/
│
├─ knapsack_env.py          # Knapsack environment and random instance generation
├─ dqn_model.py             # DQN network architectures
├─ replay_buffer.py         # Experience replay buffer
├─ train_dqn.py             # Training routines for DQN
├─ evaluate_dqn.py          # Evaluation routines
├─ utils.py                 # Utility functions (DP solver, plotting)
└─ main.py                  # Main script to train and evaluate the agent
```

Installation

Clone the repository:
```
git clone 
cd knapsack-dqn
```

Install dependencies:
```
pip install requirements
```

## Usage
Train the agent
```
python main.py
```

* Trains the DQN agent on randomly generated instances

* Saves the trained model as knapsack_dqn_model.pt

* Generates and saves training reward plots in results/training_rewards.png


### Training Rewards

During training, the total reward per episode is recorded and plotted:

![Training Rewards](results/training_rewards.png)

### Evaluation Results

The trained agent is evaluated on multiple random instances, and the performance is compared with the DP optimal solution:

![Evaluation Histogram](results/evaluation_hist.png)

Next Steps / Improvements

* Variable-length instances (10–20 items) with masking for true generalization

License

MIT License