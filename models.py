# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """
    A simple Actor-Critic network.
    It takes the state (a scalar) as input and outputs:
      - policy logits for two actions,
      - a value estimate for the state.
    """
    def __init__(self, state_dim, action_dim, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc_policy = nn.Linear(hidden_size, action_dim)
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, state):
        # state is expected to be a tensor of shape [batch_size]
        x = state.float().unsqueeze(-1)  # reshape to [batch_size, 1]
        x = F.relu(self.fc1(x))
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value
