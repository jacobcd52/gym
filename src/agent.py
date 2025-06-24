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
            print(f"Image shape: {envs.single_observation_space.shape}")
            
            # Get image dimensions - handle different possible shapes
            shape = envs.single_observation_space.shape
            print(f"Raw shape: {shape}, length: {len(shape)}")
            
            if len(shape) == 3:  # Could be (channels, height, width) or (height, width, channels)
                # Check if channels are in first or last dimension
                if shape[0] <= 4:  # Likely (channels, height, width)
                    channels, height, width = shape
                    print(f"3D shape detected (channels first): channels={channels}, height={height}, width={width}")
                else:  # Likely (height, width, channels)
                    height, width, channels = shape
                    print(f"3D shape detected (channels last): channels={channels}, height={height}, width={width}")
            elif len(shape) == 2:  # (height, width) - grayscale
                channels, height, width = 1, shape[0], shape[1]
                print(f"2D shape detected: channels={channels}, height={height}, width={width}")
            else:
                raise ValueError(f"Unsupported image shape: {shape}")
            
            # Store the number of channels for later use
            self.in_channels = channels
            
            # Calculate appropriate kernel sizes based on image dimensions
            # Use smaller kernels for larger images
            print(f"Checking dimensions: height={height}, width={width}")
            if height >= 200 and width >= 150:  # Large images like Pacman
                kernel1, stride1 = 4, 2
                kernel2, stride2 = 3, 2
                kernel3, stride3 = 2, 1
                print("Using large image kernels")
            elif height >= 100 and width >= 80:  # Medium images
                kernel1, stride1 = 6, 3
                kernel2, stride2 = 4, 2
                kernel3, stride3 = 3, 1
                print("Using medium image kernels")
            else:  # Small images (like 84x84)
                kernel1, stride1 = 8, 4
                kernel2, stride2 = 4, 2
                kernel3, stride3 = 3, 1
                print("Using small image kernels")
            
            print(f"Using kernels: {kernel1}x{kernel1}, {kernel2}x{kernel2}, {kernel3}x{kernel3}")
            print(f"First conv layer: Conv2d({channels}, 32, {kernel1}, stride={stride1})")
            
            # CNN for image-based observations
            self.cnn = nn.Sequential(
                layer_init(nn.Conv2d(channels, 32, kernel1, stride=stride1), std=config['layer_init_std']),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel2, stride=stride2), std=config['layer_init_std']),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel3, stride=stride3), std=config['layer_init_std']),
                nn.ReLU(),
                nn.Flatten(),
            )
            
            # Calculate the output size of the CNN dynamically
            with torch.no_grad():
                # Create a dummy input to calculate the output size
                dummy_input = torch.zeros(1, channels, height, width)
                print(f"Dummy input shape: {dummy_input.shape}")
                dummy_output = self.cnn(dummy_input)
                cnn_output_size = dummy_output.shape[1]
                print(f"CNN output size: {cnn_output_size}")
            
            # Add the final linear layer
            self.network = nn.Sequential(
                self.cnn,
                layer_init(nn.Linear(cnn_output_size, config['hidden_size']), std=config['layer_init_std']),
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
            # Permute from NHWC to NCHW if needed
            if x.ndim == 4 and x.shape[-1] == self.in_channels:
                x = x.permute(0, 3, 1, 2)
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        if self.is_image_space:
            x = x / 255.0
            # Permute from NHWC to NCHW if needed
            if x.ndim == 4 and x.shape[-1] == self.in_channels:
                x = x.permute(0, 3, 1, 2)
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden) 