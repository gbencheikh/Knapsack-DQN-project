import torch
from utils import generate_instance
from knapsack_env import KnapsackEnvRandom
from utils import dp_knapsack

def evaluate_policy(net, device='cpu', n_tests=100, num_items=12):
    """
    Evaluate a trained DQN policy on multiple random Knapsack instances.

    Parameters
    ----------
    net : DQNNet
        The trained DQN network to evaluate.
    device : str
        'cpu' or 'cuda' device for computation.
    n_tests : int
        Number of random instances to evaluate the policy on.
    num_items : int
        Number of items in each random instance.

    Returns
    -------
    avg_value : float
        Average total value achieved by the agent across n_tests instances.
    
    Notes
    -----
    - For each test instance, the agent selects actions greedily (highest Q-value).
    - The environment is reset with a new random instance for each test.
    - The function internally computes the optimal value via dp_knapsack 
      for comparison, but only avg_value is returned.
    - Useful for tracking policy performance during training or after training.
    """
    
    device = torch.device(device)
    net.eval()
    total_ratio = 0.0
    total_value = 0.0
    for _ in range(n_tests):
        items, capacity = generate_instance(num_items=num_items)
        env = KnapsackEnvRandom()
        # override instance manually in environment
        env.items = items
        env.capacity = capacity
        env.num_items = len(items)
        env.index = 0
        env.remaining = capacity

        achieved = 0.0
        while True:
            state = env._get_state()
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q = net(s_t)
                action = int(torch.argmax(q).item())
            _, r, done = env.step(action)
            achieved += max(0.0, float(r))
            if done:
                break
        optimal = dp_knapsack(items, capacity)
        # avoid division by zero
        ratio = achieved / optimal if optimal > 0 else 1.0
        total_ratio += ratio
        total_value += achieved
    avg_ratio = total_ratio / n_tests
    avg_value = total_value / n_tests
    net.train()
    # return average achieved value (and ratio could be printed)
    return avg_value