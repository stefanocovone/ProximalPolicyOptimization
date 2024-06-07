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

        self.device = None
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_shape = np.prod(envs.single_action_space.shape)

        # Convert action bounds to tensors
        self.action_min = torch.tensor(envs.single_action_space.low, dtype=torch.float32)
        self.action_max = torch.tensor(envs.single_action_space.high, dtype=torch.float32)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, action_shape), std=0.01),
            nn.Tanh()  # Apply tanh directly in the model
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_shape))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        # Compute action mean and log standard deviation
        action_mean = self.actor_mean(x)
        action_mean = self.scale_action(action_mean)  # Scale to action range
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

        return action, log_prob, entropy, value

    def get_action(self, x, deterministic=False):
        with torch.no_grad():
            # Compute action mean and log standard deviation
            action_mean = self.actor_mean(x)
            action_mean = self.scale_action(action_mean)  # Scale to action range
            action_log_std = self.actor_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)

            # Create the normal distribution
            probs = Normal(action_mean, action_std)

            # Sample action, otherwise use the deterministic option if specified
            if deterministic:
                action = action_mean
            else:
                action = probs.sample()

        return action, action_mean, action_std

    def get_action_mean(self, x):
        with torch.no_grad():
            return self.scale_action(self.actor_mean(x))

    def scale_action(self, action):
        action_range = (self.action_max - self.action_min) / 2.0
        action_center = (self.action_max + self.action_min) / 2.0
        return action * action_range + action_center
    
    def to(self, device):
        # Override the `to` method to also move action bounds to the specified device
        self.device = device
        self.action_min = self.action_min.to(device)
        self.action_max = self.action_max.to(device)
        return super(ActorCritic, self).to(device)
