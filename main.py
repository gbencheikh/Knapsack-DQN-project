"""
knapsack_dqn_general.py
DQN that generalizes across randomly generated knapsack instances.

Usage:
    python knapsack_dqn_general.py
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from train_DQN import *
from test_single_Instance import *

# ---------------------------
# Utils: deterministic runs
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == "__main__":

    """
    Responsibilities:
    - Train a DQN agent on randomly generated knapsack instances.
    - Evaluate agent performance on multiple test instances.
    - Run single-instance demonstrations.
    - Plot training rewards and evaluation metrics.

    """

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", DEVICE)

    # hyperparams you can tweak
    EPISODES = 4000
    NUM_ITEMS = 12
    BUFFER = 25000
    BATCH = 64
    LR = 1e-3
    TARGET_UPDATE = 200

    model, rewards, eval_scores = train_dqn_general(
        episodes=EPISODES,
        num_items=NUM_ITEMS,
        buffer_size=BUFFER,
        batch_size=BATCH,
        gamma=0.99,
        lr=LR,
        target_update=TARGET_UPDATE,
        device=DEVICE
    )
    
    MODEL_PATH = "knapsack_dqn_model.pt"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved in : {MODEL_PATH}")
    
    # Plot training reward per episode (noisy because different instances)
    plt.figure(figsize=(9,4))
    plt.plot(rewards, alpha=0.6, label='episode reward')
    # moving average for clarity
    window = max(1, len(rewards)//100)

    if window > 1:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(len(ma)), ma, label=f'ma window={window}')

    plt.xlabel("Episode")
    plt.ylabel("Episode reward (sum of rewards in that episode)")
    plt.title("Training rewards (per episode)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # print evaluation snapshots
    print("\nEvaluation snapshots (episode, mean_value over test set):")
    for ep, val in eval_scores:
        print(f"  Ep {ep}: avg_value = {val:.3f}")

    # run a few single-instance demos with greedy policy
    for _ in range(3):
        run_single_instance(model, num_items=NUM_ITEMS, device=DEVICE, greedy=True)
