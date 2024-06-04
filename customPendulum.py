import gymnasium as gym
from gymnasium.envs.classic_control import PendulumEnv
from typing import Optional
import numpy as np


def cartToAngle(state):
    return [np.arctan2(state[1], state[0]), state[2]]


class CustomRewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.reward_range = (-10000, 10000)  # Example range, adjust as needed

        # reward shaping params
        self.sigma = 10000
        self.gamma = 0.99
        self.ks = 150
        self.kz = 150
        self.kout = 1000
        self.theta = 0.05 * np.linalg.norm([np.pi, 8])

        # U_out
        self.r_max = -0.1 * self.theta ** 2
        # L_out
        self.r_min = -np.pi ** 2 - 0.1 * 8 ** 2 - 0.001 * 2 ** 2
        # U_in
        self.r_G_max = 0
        # L_in
        self.r_G_min = -self.theta ** 2 - 0.001 * 2 ** 2

        # sigma is given by Lemma III.14
        delta_in = self.r_G_max - self.r_G_min
        delta_out = self.r_max - self.r_min
        gamma_ks = self.gamma ** self.ks
        gamma_kz_1 = self.gamma ** (self.kz - 1)

        self.sigma = 1.1 * (gamma_ks / (1 - self.gamma) / (1 - gamma_ks) * delta_in + self.r_max / (1 - self.gamma) + \
                            gamma_ks * (1 - gamma_kz_1) / (1 - self.gamma) / (gamma_kz_1 - gamma_ks) * (
                                        gamma_ks * delta_in / (1 - gamma_ks) + delta_out))

        # prize according to Assumption III.10
        self.prize = - self.r_G_min - self.r_min * (1 - gamma_kz_1) / gamma_kz_1 + \
                     self.sigma * (1 - self.gamma) / gamma_kz_1

        # punishment according to Assumption III.4
        self.punishment = -self.r_max - 1 / self.gamma ** (self.kout - 1) * ((self.r_G_max + \
                                                                              self.prize) * (1 + self.gamma ** (
                    self.kout - 1) * (self.gamma - 1)) / (1 - self.gamma) - self.sigma)

        '''
        self.sigma = self.r_max/(1-self.gamma) + delta_in*gamma_ks/(1-self.gamma)/(1-gamma_ks)
        self.prize = self.sigma*(1-self.gamma) - self.r_G_min
        self.punishment = -self.r_max - 1/(self.gamma**(self.kout-1)) * ((self.r_G_max+self.prize) * (1+(self.gamma**(self.kout-1))*(self.gamma-1)) / (1-self.gamma)-self.sigma )
        '''
        self.isPrize = False
        self.isPunishment = False

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation = self.env.unwrapped.state  # Get the next observation from the environment's state
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        self.isPrize = False
        self.isPunishment = False
        if np.linalg.norm(cartToAngle(next_observation)) < self.theta:
            reward += self.prize
            self.isPrize = True
            #print(f"PRIZE           Angle:{cart_to_angle(next_observation)[0]}")
        if np.linalg.norm(observation) <= self.theta < np.linalg.norm(cartToAngle(next_observation)):
            reward += self.punishment
            self.isPunishment = True
            #print(f"PUNISHMENT      Angle:{cart_to_angle(next_observation)[0]}")

        return next_observation, reward, terminated, truncated, info


class CustomPendulumV1(PendulumEnv):
    def __init__(self, render_mode: Optional[str] = None, g=9.81):
        self.custom_g = g
        self.render_mode = render_mode
        super().__init__()

    def reset(self, initial_angle=np.pi, random=False, seed: Optional[int] = None, options: Optional[dict] = None):
        self.g = self.custom_g  # Set custom gravity
        self.max_episode_steps = 400  # Set custom max episode steps
        self.state = np.array([initial_angle, 0])
        if random:
            high = np.array([np.pi, 8])
            low = -high
            self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        return self._get_obs(), {}


# Register your custom environment with Gym
gym.envs.register(
    id='CustomPendulum-v1',
    entry_point='__main__:CustomPendulumV1',
    max_episode_steps=200,  # Default max episode steps
    reward_threshold=195.0,
)


class StableInit(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        low = -np.pi
        high = -low
        self.unwrapped.state = np.array([np.pi, 0])
        self.unwrapped.last_u = None

        if self.render_mode == "human":
            self.render()

        return self.unwrapped._get_obs(), {}
