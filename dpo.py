# dpo.py
import torch
import torch.optim as optim
import torch.nn.functional as F

def train_dpo(env, model, ref_model, epochs=1000, lr=3e-3, pair_batch_size=10, beta=1.0):
    """
    Direct Preference Optimization (DPO) training.

    For each epoch:
      - Generate 2 * pair_batch_size trajectories from the environment.
      - Sort them by total reward.
      - Pair the top half (preferred) with the bottom half (less preferred).
      - For each pair, compute cumulative log probabilities under both the current model and a fixed reference model.
      - Compute the loss using the DPO objective:
            loss = -beta * log(sigmoid( (log_pi_current_better - log_pi_ref_better)
                                         - (log_pi_current_worse - log_pi_ref_worse) ))
    The reference model (ref_model) is fixed (frozen) and serves as the baseline.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        trajectories = []
        # Generate trajectories
        for _ in range(pair_batch_size * 2):
            traj = {'states': [], 'actions': [], 'total_reward': 0.0,
                    'log_prob': 0.0, 'log_prob_ref': 0.0}
            state = env.reset()
            done = False
            while not done:
                state_tensor = torch.tensor([state], dtype=torch.float32)
                logits, _ = model(state_tensor)
                prob = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(prob)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action))
                
                # Get reference model's log probability (frozen)
                with torch.no_grad():
                    ref_logits, _ = ref_model(state_tensor)
                    ref_prob = torch.softmax(ref_logits, dim=-1)
                    ref_dist = torch.distributions.Categorical(ref_prob)
                    log_prob_ref = ref_dist.log_prob(torch.tensor(action))
                
                traj['states'].append(state)
                traj['actions'].append(action)
                traj['log_prob'] += log_prob
                traj['log_prob_ref'] += log_prob_ref
                state, reward, done, _ = env.step(action)
                traj['total_reward'] += reward
            trajectories.append(traj)
        
        # Sort trajectories by total reward (higher is better)
        trajectories.sort(key=lambda x: x['total_reward'], reverse=True)
        better = trajectories[:pair_batch_size]
        worse = trajectories[pair_batch_size:]
        
        loss = 0.0
        for b, w in zip(better, worse):
            # Difference in log probability margins (current minus reference)
            delta = (b['log_prob'] - b['log_prob_ref']) - (w['log_prob'] - w['log_prob_ref'])
            loss += -beta * torch.log(torch.sigmoid(delta))
        loss = loss / pair_batch_size
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"DPO Epoch {epoch}, Loss: {loss.item():.4f}")
