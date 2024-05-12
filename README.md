# Proximal Policy Optimization 
This repository contains a PyTorch implementation of the Proximal Policy Optimization (PPO) algorithm for Deep Reinforcement Learning.

## Dependencies
- gymnasium
- numpy
- pytorch

## Example usage
```python
from PPO import PPO
from customPendulum import CustomPendulumV1

num_sessions = 5

for i in range(1, num_sessions+1):

agent = PPO(gym_id="Pendulum-v1",
            exp_name=f"PPO_{i}",
            track=True,
            seed=i,
            max_episode_steps=400,
            num_episodes=200,
            capture_video=False)

agent.train()
agent.validate()
agent.close()