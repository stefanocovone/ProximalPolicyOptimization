from PPO import PPO


import cProfile
import pstats

num_sessions = 1

env_params = {
    'num_herders': 1,
    'num_targets': 1,
    'noise_strength': 1,
    'rho_g': 5,
    'region_length': 60,
    'beta': 3,
    'dt': 0.05,
}


profiler = cProfile.Profile()
profiler.enable()

for i in range(1, num_sessions + 1):

    agent = PPO(gym_id="Shepherding-v0",
                exp_name=f"PPO_test_{i}",
                gym_params=env_params,
                track=True,
                seed=i,
                max_episode_steps=1200,
                num_episodes=10000,
                capture_video=False,
                num_steps=8192,
                num_minibatches=64,
                gamma=0.98,
                learning_rate=2e-3,
                num_validation_episodes=100,
                ent_coef=0,
                anneal_lr=False,
                )

    agent.train()
    agent.validate(deterministic=False)
    agent.validate(deterministic=True)
    agent.close()


profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('tottime').print_stats(10)