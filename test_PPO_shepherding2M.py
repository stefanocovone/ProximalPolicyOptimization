from PPOdiscrete import PPO

if __name__ == '__main__':

    num_sessions = 1

    env_params = {
        'num_herders': 2,
        'num_targets_max': 7,
        'num_targets_min': 2,
        'num_targets': 7,
        'noise_strength': .1,
        'rho_g': 5,
        'region_length': 50,
        'k_T': 3,
        'dt': 0.05,
        'termination': False,
    }

    for i in range(2, num_sessions + 2):

        agent = PPO(gym_id="Shepherding-v0",
                    exp_name=f"PPO_2M_{i}",
                    gym_params=env_params,
                    track=False,
                    seed=10*1,
                    max_episode_steps=2000,
                    num_episodes=200000,
                    capture_video=True,
                    num_steps=2048,     # 128*128/num_envs
                    num_minibatches=1,
                    gamma=0.98,
                    learning_rate=5e-4,
                    num_validation_episodes=1000,
                    ent_coef=0.00,
                    anneal_lr=False,
                    num_envs=1,
                    )

        # agent.train()
        agent.validate()
        agent.close()
