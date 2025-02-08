
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


## Environments

- **Toy Environment (toy_env.py):**  
  A simple 1D random walk where the agent starts at the center of a line.  
  - **State:** A scalar representing the current position.  
  - **Actions:** Two actions (move left or right).  
  - **Reward:** -1 when reaching the left boundary and +1 when reaching the right boundary.

- **Grid Environment (grid_env.py):**  
  A grid world where the agent navigates a 5×5 grid with obstacles.  
  - **State:** A 2D vector \([row,\, col]\).  
  - **Actions:** Four actions (up, right, down, left).  
  - **Reward:** -1 per step and +10 for reaching the goal. Obstacles prevent movement into certain cells.

## Algorithms

- **PPO (Proximal Policy Optimization) (ppo.py):**  
  Uses per-step advantage estimation with a clipped surrogate loss to update the policy gradually.  
  - **Key Equation (Clipped Loss):**  
    \[
    L^{\text{clip}}(\theta) = \mathbb{E}\left[\min\left(r(\theta) A,\, \text{clip}\left(r(\theta),\, 1-\epsilon,\, 1+\epsilon\right)A\right)\right],
    \]
    where \( r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \) and \( A \) is the advantage estimated via GAE.

- **DPO (Direct Preference Optimization) (dpo.py):**  
  Directly optimizes preference alignment by comparing pairs of trajectories using a binary cross-entropy–style loss.  
  - **Key Equation:**  
    \[
    \mathcal{L}_{\text{DPO}} = -\beta \log \sigma\Big(\big[\log \pi_\theta(y_u|x) - \log \pi_{\text{ref}}(y_u|x)\big] - \big[\log \pi_\theta(y_l|x) - \log \pi_{\text{ref}}(y_l|x)\big]\Big),
    \]
    where \( y_u \) and \( y_l \) are the preferred and less‑preferred responses, and \(\pi_{\text{ref}}\) is a frozen reference model.

- **GRPO (Group Relative Policy Optimization) (grpo.py):**  
  Samples a group of full trajectories from the same prompt, normalizes their total rewards to compute group-based advantages, and updates the policy using a trajectory-level surrogate loss.
  - **Key Equations:**  
    - **Group‑normalized Advantage:**  
      \[
      A_i = \frac{R_i - \mu_{\mathcal{G}}}{\sigma_{\mathcal{G}} + \epsilon},
      \]
      where \( R_i \) is the total reward for trajectory \( i \), and \(\mu_{\mathcal{G}}\) and \(\sigma_{\mathcal{G}}\) are the mean and standard deviation over the group.
    - **Trajectory‑level Clipped Loss:**  
      \[
      L^{\text{GRPO}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \min\left(r_i A_i,\, \text{clip}(r_i,\, 1-\epsilon,\, 1+\epsilon) A_i\right),
      \]
      where \( r_i = \exp\big(\text{log\_prob}_{\text{new}} - \text{log\_prob}_{\text{old}}\big) \).

## Generalized Advantage Estimation (GAE)

GAE is used in PPO to efficiently estimate the advantage \(A_t\) at each time step by balancing bias and variance. The advantage is computed as a weighted sum of temporal-difference (TD) errors:

\[
A_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}, \quad \text{with} \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]

- **\(\gamma\):** Discount factor.  
- **\(\lambda\):** Parameter controlling the trade-off between bias and variance.  
GAE allows the policy to use multiple future rewards for a stable and low-variance advantage estimate.
