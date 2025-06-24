import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
import gymnasium as gym

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, config):
        super().__init__()
        
        # Check if the observation space is image-based (e.g., Atari)
        is_image_space = isinstance(envs.single_observation_space, gym.spaces.Box) and len(envs.single_observation_space.shape) > 1

        if is_image_space:
            print("Image space type environment")
            # CNN for image-based observations
            # TODO: make dims configurable
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(envs.single_observation_space.shape[0], 32, 8, stride=4), std=config['layer_init_std']),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2), std=config['layer_init_std']),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1), std=config['layer_init_std']),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, config['hidden_size']), std=config['layer_init_std']),
                nn.ReLU(),
            )
            self.final_hidden_size = config['hidden_size']  # For image-based observations
        else:
            print("Vector space type environment")
            # MLP for vector-based observations
            input_size = np.array(envs.single_observation_space.shape).prod()
            hidden_sizes = config.get('hidden_sizes', [config['hidden_size']])
            
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.append(layer_init(nn.Linear(prev_size, hidden_size)))
                layers.append(nn.ReLU())
                prev_size = hidden_size
            
            self.network = nn.Sequential(*layers)
            self.final_hidden_size = hidden_sizes[-1]  # Store the final hidden size

        self.actor = layer_init(nn.Linear(self.final_hidden_size, envs.single_action_space.n), std=config['actor_std'])
        self.critic = layer_init(nn.Linear(self.final_hidden_size, 1), std=config['critic_std'])
        self.is_image_space = is_image_space

    def get_value(self, x):
        if self.is_image_space:
            x = x / 255.0
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        if self.is_image_space:
            x = x / 255.0
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden) 