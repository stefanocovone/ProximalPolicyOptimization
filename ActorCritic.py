import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, envs):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        # Compute action mean and log standard deviation
        action_mean = self.actor_mean(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # Create the normal distribution
        probs = Normal(action_mean, action_std)

        # Sample action if not provided, otherwise use the deterministic option if specified
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = probs.sample()

        # Compute log probability and entropy of the action
        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)

        # Compute the value from the critic
        value = self.critic(x)

        return action, log_prob, entropy, value, action_std

