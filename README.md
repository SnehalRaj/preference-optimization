
# PPO, DPO, and GRPO from Scratch

This repository provides minimal, from‐scratch implementations of three reinforcement learning algorithms in a “Karpathy‐style” (i.e. simple, didactic, and transparent):

- **PPO (Proximal Policy Optimization)**
- **DPO (Direct Preference Optimization)**
- **GRPO (Group Relative Policy Optimization)**

Each algorithm is implemented on a toy 1D random-walk environment (defined in `toy_env.py`), and uses an Actor-Critic network (in `models.py`). The key differences among the methods are:

- **PPO:** Uses per-step advantage estimation (via a critic or Monte Carlo) with a clipped surrogate loss plus optional entropy and KL penalties.
- **DPO:** Uses pairs of trajectories (one preferred, one less preferred) along with a reference model to directly optimize a binary cross-entropy loss. This removes the need for a separate reward model.
- **GRPO:** Samples a group of full trajectories for a given prompt, computes group-normalized advantages (i.e. based on the normalized total reward), and applies a trajectory-level surrogate loss with a KL penalty.

## Running the Code

To run training for a specific algorithm, use:

```bash
python main.py --algo ppo
python main.py --algo dpo
python main.py --algo grpo
```
