import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, envs):
        super(ActorCritic, self).__init__()

        self.device = None
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_shape = envs.single_action_space.nvec
        self.nvec = action_shape

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, action_shape.sum()), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        split_logits = torch.split(logits, self.nvec.tolist(), dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action.T, logprob.sum(0), entropy.sum(0), self.critic(x)

    def get_action(self, x):
        # Compute logits from the actor network
        logits = self.actor(x)

        # Split the logits according to the dimensions specified in self.nvec
        split_logits = torch.split(logits, self.nvec.tolist(), dim=1)

        # Create a list of Categorical distributions from the split logits
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]

        # Compute the action with the highest probability for each categorical distribution
        action = torch.stack([categorical.probs.argmax(dim=-1) for categorical in multi_categoricals])

        # Transpose the action tensor to match the required output shape
        return action.T
