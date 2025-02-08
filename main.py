# main.py
import argparse
from toy_env import ToyEnv
from models import ActorCritic
from ppo import train_ppo
from dpo import train_dpo
from grpo import train_grpo

def main():
    parser = argparse.ArgumentParser(
        description="Train RL algorithms: PPO, DPO, or GRPO on a toy environment."
    )
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dpo", "grpo"],
                        help="Algorithm to run")
    args = parser.parse_args()
    
    env = ToyEnv(size=5)
    model = ActorCritic(state_dim=1, action_dim=2)
    
    if args.algo == "ppo":
        print("Training using PPO...")
        train_ppo(env, model)
    elif args.algo == "dpo":
        print("Training using DPO...")
        # For DPO, create a reference model as a frozen copy of the initial model.
        from models import ActorCritic
        ref_model = ActorCritic(state_dim=1, action_dim=2)
        ref_model.load_state_dict(model.state_dict())
        for param in ref_model.parameters():
            param.requires_grad = False
        train_dpo(env, model, ref_model)
    elif args.algo == "grpo":
        print("Training using GRPO...")
        train_grpo(env, model)
        
if __name__ == "__main__":
    main()
