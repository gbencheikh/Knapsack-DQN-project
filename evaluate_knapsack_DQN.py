"""
Evaluation script for the generalized Knapsack DQN agent.

Usage:
    python evaluate_knapsack_dqn.py
"""

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from DQN_model import DQNNet
from utils import generate_instance, dp_knapsack
from knapsack_env import KnapsackEnvRandom

# ---------------------------
# Configuration
# ---------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "knapsack_dqn_model.pt" 
NUM_TESTS = 200
NUM_ITEMS = 12
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------
# Load model (or init new)
# ---------------------------
input_dim = 5
output_dim = 2
net = DQNNet(input_dim, output_dim)
if os.path.exists(MODEL_PATH):
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Model has been loaded from {MODEL_PATH}")
else:
    print("No model found.")
net.to(DEVICE)
net.eval()

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_agent(net, n_tests=100, num_items=12, device='cpu'):
    """
    Evaluate a trained DQN agent on multiple random Knapsack instances.

    Parameters
    ----------
    net : DQNNet
        Trained DQN network.
    device : str
        'cpu' or 'cuda'.
    n_tests : int
        Number of random instances to test.
    num_items : int
        Number of items per random instance.

    Returns
    -------
    float
        Average total value achieved by the agent over n_tests instances.

    Notes
    -----
    For each test instance:
    - The agent selects actions greedily (max Q-value).
    - The performance is compared to the optimal value computed via dp_knapsack.
    """
    ratios = []
    values = []
    optimals = []
    for _ in range(n_tests):
        items, capacity = generate_instance(num_items=num_items, w_low=1, w_high=15, v_low=1, v_high=20,
                                                      cap_low=max(5, num_items), cap_high = 15 * num_items // 2 + 5)
        env = KnapsackEnvRandom()
        env.items = items
        env.capacity = capacity
        env.num_items = len(items)
        env.index = 0
        env.remaining = capacity

        total_value = 0.0
        while True:
            state = env._get_state()
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q = net(s_t)
                action = int(torch.argmax(q).item())
            _, r, done = env.step(action)
            total_value += max(0.0, float(r))
            if done:
                break

        optimal = dp_knapsack(items, capacity)
        ratio = total_value / optimal if optimal > 0 else 1.0
        ratios.append(ratio)
        values.append(total_value)
        optimals.append(optimal)

    ratios = np.array(ratios)
    values = np.array(values)
    optimals = np.array(optimals)
    return ratios, values, optimals

ratios, values, optimals = evaluate_agent(net, n_tests=NUM_TESTS, num_items=NUM_ITEMS, device=DEVICE)

# ---------------------------
# Stats & reporting
# ---------------------------
mean_ratio = np.mean(ratios)
std_ratio = np.std(ratios)
mean_value = np.mean(values)
mean_optimal = np.mean(optimals)
print("\nEvaluation Results")
print("----------------------")
print(f"Number of tests:      {NUM_TESTS}")
print(f"Mean value (agent):   {mean_value:.2f}")
print(f"Mean value (optimal): {mean_optimal:.2f}")
print(f"Mean ratio:           {mean_ratio:.3f} ± {std_ratio:.3f}")

# Save CSV
csv_path = os.path.join(RESULTS_DIR, "knapsack_eval.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["test_id", "agent_value", "optimal_value", "ratio"])
    for i, (v, o, r) in enumerate(zip(values, optimals, ratios)):
        writer.writerow([i + 1, v, o, r])
print(f"\nResults saved to: {csv_path}")

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(7,4))
plt.hist(ratios, bins=20, color='skyblue', edgecolor='black')
plt.axvline(mean_ratio, color='red', linestyle='--', label=f"mean={mean_ratio:.2f}")
plt.xlabel("Agent / Optimal Ratio")
plt.ylabel("Frequency")
plt.title("Distribution of performance ratios over test instances")
plt.legend()
plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "knapsack_eval_hist.png")
plt.savefig(fig_path)
plt.show()
print(f"Histogram saved to: {fig_path}")
