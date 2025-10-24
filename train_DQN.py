import torch
import torch.nn as nn
import torch.optim as optim
import time
from knapsack_env import KnapsackEnvRandom
from DQN_model import DQNNet
from replay_buffer import ReplayBuffer
from evaluate_DQN import evaluate_policy
import random


def train_dqn_general(episodes=4000,
                      num_items=12,
                      buffer_size=20000,
                      batch_size=64,
                      gamma=0.99,
                      lr=1e-3,
                      eps_start=1.0,
                      eps_min=0.05,
                      eps_decay=0.995,
                      target_update=100,
                      device='cpu'):
    
    """
    Train a DQN agent on randomly generated knapsack instances.

    Parameters
    ----------
    episodes : int
        Number of training episodes.
    num_items : int
        Number of items per instance.
    buffer_size : int
        Maximum capacity of the replay buffer.
    batch_size : int
        Batch size for gradient updates.
    gamma : float
        Discount factor.
    lr : float
        Learning rate for optimizer.
    target_update : int
        Number of episodes between target network updates.
    device : str
        'cpu' or 'cuda' device.

    Returns
    -------
    policy_net : DQNNet
        Trained DQN network.
    rewards_history : list of float
        Rewards per episode.
    eval_scores : list of tuple
        Evaluation snapshots (episode, mean_value over test set)
    """
    
    device = torch.device(device)
    env_template = KnapsackEnvRandom(num_items=num_items)
    input_dim = 5
    output_dim = 2
    policy_net = DQNNet(input_dim, output_dim).to(device)
    target_net = DQNNet(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    replay = ReplayBuffer(capacity=buffer_size)
    epsilon = eps_start

    rewards_history = []
    eval_every = max(1, episodes // 40)
    eval_scores = []
    t0 = time.time()
    for ep in range(1, episodes + 1):
        state = env_template.reset()
        ep_reward = 0.0
        while True:
            # action selection
            if random.random() < epsilon:
                action = random.randrange(output_dim)
            else:
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    qvals = policy_net(s_t)
                    action = int(torch.argmax(qvals).item())
            next_state, reward, done = env_template.step(action)
            replay.push(state, action, reward, next_state, float(done))
            ep_reward += reward
            state = next_state

            # learn
            if len(replay) >= batch_size:
                s_batch, a_batch, r_batch, s2_batch, done_batch = replay.sample(batch_size)
                s_batch = torch.FloatTensor(s_batch).to(device)
                a_batch = torch.LongTensor(a_batch).unsqueeze(1).to(device)
                r_batch = torch.FloatTensor(r_batch).to(device)
                s2_batch = torch.FloatTensor(s2_batch).to(device)
                done_batch = torch.FloatTensor(done_batch).to(device)

                q_values = policy_net(s_batch).gather(1, a_batch).squeeze()
                with torch.no_grad():
                    next_q = target_net(s2_batch).max(1)[0]
                target = r_batch + gamma * next_q * (1 - done_batch)
                loss = criterion(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        rewards_history.append(ep_reward)
        epsilon = max(eps_min, epsilon * eps_decay)

        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if ep % eval_every == 0 or ep == episodes:
            mean_eval = evaluate_policy(policy_net, device=device, n_tests=100, num_items=num_items)
            eval_scores.append((ep, mean_eval))
            print(f"Ep {ep}/{episodes} | ep_reward: {ep_reward:.2f} | eval_mean_value: {mean_eval:.2f} | eps: {epsilon:.3f}")

    t1 = time.time()
    print(f"\nTraining finished in {t1 - t0:.1f}s")
    
    return policy_net, rewards_history, eval_scores