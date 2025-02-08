# ppo.py
import torch
import torch.optim as optim
import torch.nn.functional as F

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    """
    advantages = []
    gae = 0
    values = values + [0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

def train_ppo(env, model, epochs=1000, gamma=0.99, lam=0.95,
              clip_epsilon=0.2, lr=3e-3, update_steps=5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        states, actions, rewards, log_probs, values = [], [], [], [], []
        state = env.reset()
        done = False
        
        # Collect one trajectory
        while not done:
            state_tensor = torch.tensor([state])
            logits, value = model(state_tensor)
            prob = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(prob)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value.item())
            
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
        
        advantages = compute_advantages(rewards, values, gamma, lam)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        # Convert collected data to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions)
        old_log_probs_tensor = torch.tensor(log_probs)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        
        for _ in range(update_steps):
            logits, value_pred = model(states_tensor)
            prob = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(prob)
            new_log_probs = dist.log_prob(actions_tensor)
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value_pred.squeeze(), returns_tensor)
            loss = policy_loss + value_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 100 == 0:
            print(f"PPO Epoch {epoch}, Total Reward: {sum(rewards)}")
