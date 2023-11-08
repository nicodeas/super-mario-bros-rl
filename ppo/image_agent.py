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

        # image feature extration through a sequence of convolutional layers
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        # final hidden layer of shape 512 from image network fed into critic
        # critic only has an output shape of 1 as it is providing feedback to the actor

        self.critic = layer_init(nn.Linear(512, 1), std=1)
        # final hidden layer of shape 512 from image network fed into actor
        # actor has an output shape of the environment's single_action_space
        # this array of size (num possible actions) contains the log odds of performing
        # each action
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)

    def critic_action(self, x):
        # scale outputs of network
        hidden = self.network(x / 255.0)
        return self.critic(hidden)

    def get_action_and_critic(self, x, action=None):
        # scale outputs of network
        hidden = self.network(x / 255.0)
        # logits are un-normalised action probabilities
        logits = self.actor(hidden)
        # pass logits into a categorical distribution; discrete action space
        # to get the action probability distribution
        # PPO assumes that policy comes from a probability distribution
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
