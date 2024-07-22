from PPO import PPO

if __name__ == '__main__':

    num_sessions = 1

    env_params = {
        'num_herders': 1,
        'num_targets': 1,
        'noise_strength': 1,
        'rho_g': 5,
        'region_length': 50,
        'k_T': 3,
        'dt': 0.05,
        'k_rep': 100,
    }

    for i in range(4, num_sessions + 1):

        agent = PPO(gym_id="Shepherding-v0",
                    exp_name=f"PPO_{i}",
                    gym_params=env_params,
                    track=True,
                    seed=60*1,
                    max_episode_steps=1200,
                    num_episodes=20000,
                    capture_video=False,
                    num_steps=4096,
                    num_minibatches=128,
                    gamma=0.98,
                    learning_rate=5e-4,
                    num_validation_episodes=1000,
                    ent_coef=0,
                    anneal_lr=False,
                    num_envs=8,
                    )

        agent.train()
        agent.validate()
        agent.close()
