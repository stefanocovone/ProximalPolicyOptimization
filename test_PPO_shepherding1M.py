from PPOdiscrete import PPO

if __name__ == '__main__':

    num_sessions = 1

    env_params = {
        'num_herders': 1,
        'num_targets_max': 7,
        'num_targets_min': 2,
        'noise_strength': 1,
        'rho_g': 5,
        'region_length': 60,
        'k_T': 3,
        'dt': 0.05,
    }

    for i in range(1, num_sessions + 1):

        agent = PPO(gym_id="Shepherding-v0",
                    exp_name=f"PPO_1M_random_{i}",
                    gym_params=env_params,
                    track=True,
                    seed=10*i,
                    max_episode_steps=3000,
                    num_episodes=100000,
                    capture_video=False,
                    num_steps=64,
                    num_minibatches=4,
                    gamma=0.99,
                    learning_rate=1e-4,
                    num_validation_episodes=1000,
                    ent_coef=0.002,
                    anneal_lr=False,
                    num_envs=8,
                    )

        agent.train()
        agent.validate()
        agent.close()
