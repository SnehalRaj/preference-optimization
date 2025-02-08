# grpo.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

def compute_kl_batch(old_model, current_model, states):
    """
    Compute the average KL divergence over a batch of states.
    This function converts the list of states to a tensor and computes KL divergence
    in a vectorized fashion.
    """
    states_tensor = torch.tensor(states, dtype=torch.float32)
    with torch.no_grad():
        old_logits, _ = old_model(states_tensor)
    current_logits, _ = current_model(states_tensor)
    old_prob = torch.softmax(old_logits, dim=-1)
    old_log_prob = torch.log_softmax(old_logits, dim=-1)
    current_log_prob = torch.log_softmax(current_logits, dim=-1)
    # Compute KL divergence per state and take the mean.
    kl_div = torch.sum(old_prob * (old_log_prob - current_log_prob), dim=-1)
    return kl_div.mean()

def compute_entropy(model, states):
    """
    Compute the average entropy over a batch of states.
    """
    states_tensor = torch.tensor(states, dtype=torch.float32)
    logits, _ = model(states_tensor)
    prob = torch.softmax(logits, dim=-1)
    log_prob = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(prob * log_prob, dim=-1)
    return entropy.mean()

def train_grpo(env, model, epochs=1000, group_batch_size=10, clip_epsilon=0.2, lr=3e-3,
               kl_weight=0.1, entropy_weight=0.0):
    """
    GRPO with improved group-based advantage estimation.
    
    For each epoch:
      - Sample group_batch_size trajectories (responses) from the same prompt (i.e., starting state).
      - For each trajectory, accumulate the log probability and total reward.
      - Normalize total rewards to get group-based advantages:
            A_i = (R_i - mean(R_group)) / (std(R_group) + epsilon)
      - Recompute the log probability for each trajectory using the current policy.
      - Compute the surrogate loss with clipping, add a KL divergence penalty computed in batch,
        and (optionally) subtract an entropy bonus to encourage exploration.
    
    Hyperparameters such as learning rate, clip epsilon, group batch size, and KL/entropy weights 
    may require tuning.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    eps = 1e-6  # Small constant for stable normalization
    
    for epoch in range(epochs):
        trajectories = []
        # Sample group_batch_size trajectories from the same prompt.
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
        std_reward = std_reward if std_reward > eps else eps  # Prevent division by zero
        
        # Compute normalized advantage for each trajectory.
        advantages = [(r - mean_reward) / std_reward for r in rewards]
        
        # Save a frozen copy of the current model for KL computation.
        old_model = copy.deepcopy(model)
        old_model.eval()
        
        total_policy_loss = 0.0
        all_states = []  # Gather states for KL and entropy computation.
        
        # Process each trajectory: recompute cumulative log probability under current policy.
        for traj, adv in zip(trajectories, advantages):
            traj_log_prob = 0.0
            for state, action in zip(traj['states'], traj['actions']):
                state_tensor = torch.tensor([state], dtype=torch.float32)
                logits, _ = model(state_tensor)
                prob = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(prob)
                traj_log_prob += dist.log_prob(torch.tensor(action))
                all_states.append(state)
            # Compute the probability ratio between new and old log probabilities.
            ratio = torch.exp(traj_log_prob - traj['log_prob'])
            # Clipped surrogate loss (similar to PPO, applied at trajectory level).
            surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv)
            total_policy_loss += -surrogate  # Negative for gradient ascent.
        
        total_policy_loss = total_policy_loss / group_batch_size
        
        # Compute the KL divergence penalty over all collected states.
        kl_div = compute_kl_batch(old_model, model, all_states)
        
        # Optionally compute an entropy bonus to encourage exploration.
        entropy = compute_entropy(model, all_states) if entropy_weight > 0.0 else 0.0
        
        # Total loss: surrogate loss + KL penalty - (entropy bonus).
        loss = total_policy_loss + kl_weight * kl_div - entropy_weight * entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"GRPO Epoch {epoch}, Mean Reward: {mean_reward:.2f}, KL: {kl_div.item():.4f}, "
                  f"Entropy: {entropy:.4f}, Loss: {loss.item():.4f}")
