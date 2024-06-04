from PPO import PPO

num_sessions = 4

for i in range(1, num_sessions + 1):

    agent = PPO(gym_id="Pendulum-v1",
                exp_name=f"PPO_{i}",
                track=True,
                seed=i,
                max_episode_steps=400,
                num_episodes=200,
                capture_video=False,
                num_steps=1024,
                num_minibatches=64,
                gamma=0.9,
                learning_rate=1e-3,
                num_validation_episodes=100,
                ent_coef=-0.003
                )

    agent.train()
    agent.validate(deterministic=False)
    agent.validate(deterministic=True)
    agent.close()
