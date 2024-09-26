import argparse
from PPOdiscrete import PPO


def parse_arguments():
    parser = argparse.ArgumentParser(description="PPO agent for shepherding problem")

    # Optional arguments for reward parameters
    parser.add_argument('--k_R', type=float, default=0.01, help="Reward parameter k_R")
    parser.add_argument('--k_p', type=float, default=5, help="Reward parameter k_p")
    parser.add_argument('--k_all', type=float, default=0, help="Reward parameter k_all")
    parser.add_argument('--k_chi', type=float, default=0, help="Reward parameter k_chi")

    # Optional argument for experiment tag
    parser.add_argument('--exp_tag', type=str, default='1', help="Experiment tag")

    return parser.parse_args()


if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_arguments()

    num_sessions = 1

    # Define the environment parameters, including the reward parameters from the input
    env_params = {
        'num_herders': 1,
        'num_targets_max': 7,
        'num_targets_min': 2,
        'num_targets': 7,
        'noise_strength': .1,
        'rho_g': 5,
        'region_length': 50,
        'k_T': 3,
        'dt': 0.05,
        'termination': False,
        # reward params
        'k_R': args.k_R,
        'k_p': args.k_p,
        'k_all': args.k_all,
        'k_chi': args.k_chi,
    }

    # Use the experiment tag from the input arguments
    exp_tag = args.exp_tag

    for i in range(1, num_sessions + 1):
        agent = PPO(gym_id="Shepherding-v0",
                    exp_name=f"PPO_1M_{exp_tag}_{i}",
                    gym_params=env_params,
                    track=True,
                    seed=10 * i,
                    max_episode_steps=2000,
                    num_episodes=200000,
                    capture_video=False,
                    num_steps=int(128 * 128 / 32),  # 128*128/num_envs
                    num_minibatches=128,
                    gamma=0.98,
                    learning_rate=5e-4,
                    num_validation_episodes=1000,
                    ent_coef=0.00,
                    anneal_lr=False,
                    num_envs=32,  # 32
                    )

        agent.train()
        agent.validate()
        agent.close()
