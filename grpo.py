# grpo.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

def compute_kl(old_model, current_model, states):
    """
    Compute the average KL divergence between the old and current policies over a set of states.
    """
    kl_divs = []
    for state in states:
        state_tensor = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            old_logits, _ = old_model(state_tensor)
        current_logits, _ = current_model(state_tensor)
        old_prob = torch.softmax(old_logits, dim=-1)
        old_log_prob = torch.log_softmax(old_logits, dim=-1)
        current_log_prob = torch.log_softmax(current_logits, dim=-1)
        # KL divergence for this state: sum(old_prob * (old_log_prob - current_log_prob))
        kl = (old_prob * (old_log_prob - current_log_prob)).sum()
        kl_divs.append(kl)
    return torch.stack(kl_divs).mean()

def train_grpo(env, model, epochs=1000, group_batch_size=10, clip_epsilon=0.2, lr=3e-3, kl_weight=0.1):
    """
    GRPO with group-based advantage estimation.

    For each epoch:
      - Sample a group of trajectories (responses) from the same prompt (i.e. starting state).
      - For each trajectory, accumulate the log probability (the product of per-step probabilities)
        and total reward.
      - Compute the normalized advantage for each trajectory:
            A_i = (R_i - mean(R_group)) / std(R_group)
      - Recompute the current log probabilities for the trajectory and form the probability ratio.
      - Use a clipped surrogate loss (as in PPO) and add a KL divergence penalty computed over all states
        encountered in the trajectories.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        trajectories = []
        # Sample group_batch_size trajectories from the environment (same prompt)
        for _ in range(group_batch_size):
            traj = {'states': [], 'actions': [], 'total_reward': 0.0, 'log_prob': 0.0}
            state = env.reset()
            done = False
            while not done:
                state_tensor = torch.tensor([state], dtype=torch.float32)
                logits, _ = model(state_tensor)
                prob = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(prob)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action))
                
                traj['states'].append(state)
                traj['actions'].append(action)
                traj['log_prob'] += log_prob
                state, reward, done, _ = env.step(action)
                traj['total_reward'] += reward
            trajectories.append(traj)
        
        # Compute group statistics for total rewards
        rewards = [traj['total_reward'] for traj in trajectories]
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        if std_reward < 1e-8:
            std_reward = 1e-8
        
        # Compute normalized advantage for each trajectory
        advantages = [(r - mean_reward) / std_reward for r in rewards]
        
        # Save a copy of the current model to serve as the reference for KL divergence
        old_model = copy.deepcopy(model)
        old_model.eval()
        
        total_policy_loss = 0.0
        all_states = []  # collect states for KL penalty
        
        for traj, adv in zip(trajectories, advantages):
            traj_log_prob = 0.0
            for state, action in zip(traj['states'], traj['actions']):
                state_tensor = torch.tensor([state], dtype=torch.float32)
                logits, _ = model(state_tensor)
                prob = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(prob)
                traj_log_prob += dist.log_prob(torch.tensor(action))
                all_states.append(state)
            ratio = torch.exp(traj_log_prob - traj['log_prob'])
            surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv)
            total_policy_loss += -surrogate
        
        total_policy_loss = total_policy_loss / group_batch_size
        
        # KL divergence penalty over all states encountered
        kl_div = compute_kl(old_model, model, all_states)
        loss = total_policy_loss + kl_weight * kl_div
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"GRPO Epoch {epoch}, Mean Reward: {mean_reward:.2f}, KL: {kl_div.item():.4f}, Loss: {loss.item():.4f}")
