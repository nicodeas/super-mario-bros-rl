import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        # critic only has an output shape of 1 as it is providing feedback to the actor
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # actor has an output shape of the environment's single_action_space
        # this array of size (num possible actions) contains the log odds of performing
        # each action
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def critic_action(self, x):
        return self.critic(x)

    def get_action_and_critic(self, x, action=None):
        # logits are un-normalised action probabilities
        logits = self.actor(x)
        # pass logits into a categorical distribution; discrete action space
        # to get the action probability distribution
        # PPO assumes that policy comes from a probability distribution
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
