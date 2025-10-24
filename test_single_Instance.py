from utils import generate_instance
from knapsack_env import KnapsackEnvRandom
from utils import dp_knapsack
import torch
import random 

def run_single_instance(net, num_items=12, device='cpu', greedy=True):
    """
    Run the DQN agent on a single random Knapsack instance and display results.

    Parameters
    ----------
    net : DQNNet
        Trained DQN network.
    num_items : int
        Number of items in the generated instance.
    device : str
        'cpu' or 'cuda'.
    greedy : bool
        If True, select actions greedily according to Q-values.
        If False, select actions randomly.

    Returns
    -------
    total_value : float
        Total value achieved by the agent on this instance.
    optimal_value : int
        Maximum achievable value computed via dp_knapsack.

    Side Effects
    ------------
    Prints:
    - Knapsack capacity
    - Items (weight, value)
    - Actions taken by the agent
    - Total value achieved
    - Optimal value for comparison
    """
    
    items, capacity = generate_instance(num_items=num_items)
    print("\nSingle instance:")
    print("Capacity:", capacity)
    print("Items (w, v):", items)
    env = KnapsackEnvRandom()
    env.items = items
    env.capacity = capacity
    env.num_items = len(items)
    env.index = 0
    env.remaining = capacity

    total = 0.0
    taken = []
    while True:
        state = env._get_state()
        if greedy:
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q = net(s_t)
                action = int(torch.argmax(q).item())
        else:
            action = random.choice([0, 1])
        _, r, done = env.step(action)
        if r > 0:
            taken.append((env.index - 1, items[env.index - 1]))
        total += max(0.0, r)
        if done:
            break
    optimal = dp_knapsack(items, capacity)
    print("Taken items (index, (w,v)):", taken)
    print("Total value achieved:", total)
    print("Optimal (DP):", optimal)
    return total, optimal