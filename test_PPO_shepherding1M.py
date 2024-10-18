import argparse
from PPOdiscrete import PPO


def parse_arguments():
    parser = argparse.ArgumentParser(description="PPO agent for shepherding problem")

    # Optional arguments for reward parameters
    parser.add_argument('--k_R', type=float, default=0.01, help="Reward parameter k_R")
    parser.add_argument('--k_p', type=float, default=0, help="Reward parameter k_p")
    parser.add_argument('--k_all', type=float, default=0, help="Reward parameter k_all")
    parser.add_argument('--k_chi', type=float, default=0.2, help="Reward parameter k_chi")

    # Optional argument for validation targets
    parser.add_argument('--num_targets', type=int, default=7, help="Number of targets during validation")

    # Optional argument for hyperparameters
    parser.add_argument('--l_r', type=float, default=0.0005, help="Learning rate")

    # Optional argument for experiment tag
    parser.add_argument('--exp_tag', type=str, default='codetest', help="Experiment tag")

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
        'num_targets': args.num_targets,
        'noise_strength': .1,
        'rho_g': 5,
        'region_length': 50,
        'k_T': 3,
        'dt': 0.05,
        'termination': True,
        'random_targets': False,
        'target_selection_rate': 20,
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
                    exp_name_val=f"PPO_1M_{exp_tag}_M{args.num_targets}_{i}",
                    gym_params=env_params,
                    track=True,
                    seed=10 * 1,
                    max_episode_steps=2000,
                    num_episodes=1000,
                    capture_video=False,
                    render=False,
                    num_steps=int(128 * 128 / 32),  # 128*128/num_envs
                    num_minibatches=128,
                    gamma=0.98,
                    learning_rate=args.l_r,     # 5e-4
                    num_validation_episodes=100,
                    ent_coef=0.00,
                    anneal_lr=False,
                    num_envs=8,  # 32
                    )

        agent.train()
        agent.validate()
        agent.close()
