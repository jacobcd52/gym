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
            
            cnn_modules = []
            if 'cnn_layers' in config:
                print("Building CNN from configuration...")
                in_ch = channels
                for layer_params in config['cnn_layers']:
                    cnn_modules.append(layer_init(nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=layer_params['out_channels'],
                        kernel_size=layer_params['kernel_size'],
                        stride=layer_params['stride']
                    ), std=config['layer_init_std']))
                    cnn_modules.append(nn.ReLU())
                    in_ch = layer_params['out_channels']
            else:
                print("Warning: 'cnn_layers' not found in config. Using default CNN architecture.")
                # Use smaller kernels for larger images
                print(f"Checking dimensions: height={height}, width={width}")
                if height >= 200 and width >= 150:  # Large images like Pacman
                    k1, s1, c1 = 4, 2, 32
                    k2, s2, c2 = 3, 2, 64
                    k3, s3, c3 = 2, 1, 64
                    print("Using large image kernels")
                elif height >= 100 and width >= 80:  # Medium images
                    k1, s1, c1 = 6, 3, 32
                    k2, s2, c2 = 4, 2, 64
                    k3, s3, c3 = 3, 1, 64
                    print("Using medium image kernels")
                else:  # Small images (like 84x84)
                    k1, s1, c1 = 8, 4, 32
                    k2, s2, c2 = 4, 2, 64
                    k3, s3, c3 = 3, 1, 64
                    print("Using small image kernels")
                
                cnn_modules.extend([
                    layer_init(nn.Conv2d(channels, c1, k1, stride=s1), std=config['layer_init_std']),
                    nn.ReLU(),
                    layer_init(nn.Conv2d(c1, c2, k2, stride=s2), std=config['layer_init_std']),
                    nn.ReLU(),
                    layer_init(nn.Conv2d(c2, c3, k3, stride=s3), std=config['layer_init_std']),
                    nn.ReLU(),
                ])

            cnn_modules.append(nn.Flatten())
            self.cnn = nn.Sequential(*cnn_modules)
            
            # Calculate the output size of the CNN dynamically and print activations
            with torch.no_grad():
                dummy_input = torch.zeros(1, channels, height, width)
                print(f"\nActivation shape analysis (dummy input shape: {dummy_input.shape}):")
                x = dummy_input
                for i, layer in enumerate(self.cnn):
                    x = layer(x)
                    print(f"  - After layer {i} ({layer.__class__.__name__}): {x.shape}")
                
                cnn_output_size = x.shape[1]
                print(f"CNN output size: {cnn_output_size}\n")
            
            # MLP for the part after CNN
            hidden_sizes = config['hidden_sizes']
            
            mlp_layers = []
            prev_size = cnn_output_size
            
            for hidden_size in hidden_sizes:
                mlp_layers.append(layer_init(nn.Linear(prev_size, hidden_size), std=config['layer_init_std']))
                mlp_layers.append(nn.ReLU())
                prev_size = hidden_size
            
            self.network = nn.Sequential(
                self.cnn,
                *mlp_layers,
            )
            self.final_hidden_size = hidden_sizes[-1]
        else:
            print("Vector space type environment")
            # MLP for vector-based observations
            input_size = np.array(envs.single_observation_space.shape).prod()
            hidden_sizes = config['hidden_sizes']
            
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