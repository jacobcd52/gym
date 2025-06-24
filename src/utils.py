import os
import imageio
import numpy as np
import pygame
import torch
import gymnasium as gym
from gymnasium import wrappers

def save_video(frames, folder, filename):
    """Saves a list of frames as an mp4 video."""
    path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    imageio.mimwrite(path, frames, fps=30, macro_block_size=1)

def record_episode(agent, config, run_name, completed_episodes):
    """
    Records an episode by saving observations and advantages.
    """
    trajectories_dir = f"trajectories/{run_name}"
    os.makedirs(trajectories_dir, exist_ok=True)
    
    env_id = config['env_id']
    device = next(agent.parameters()).device

    # Create a single environment for recording
    env = gym.make(env_id, render_mode="rgb_array")
    is_atari = "NoFrameskip" in env_id
    if is_atari:
        env = wrappers.ResizeObservation(env, (84, 84))
        env = wrappers.GrayScaleObservation(env)
        env = wrappers.FrameStack(env, 4)

    obs_list, reward_list, done_list, value_list = [], [], [], []
    
    obs, _ = env.reset()
    done = False
    
    with torch.no_grad():
        while not done:
            # For atari, obs is a LazyFrame. We need to convert it to a numpy array.
            current_obs = np.array(obs)
            obs_tensor = torch.Tensor(current_obs).to(device).unsqueeze(0)
            
            # Get agent's action and value
            action, _, _, value = agent.get_action_and_value(obs_tensor)
            
            # Store observation and value
            obs_list.append(current_obs)
            value_list.append(value.cpu().item())
            
            # Step the environment
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            
            reward_list.append(reward)
            done_list.append(done)
            
            obs = next_obs
            
        # Add the final observation's value
        final_obs = np.array(obs)
        next_value = agent.get_value(torch.Tensor(final_obs).to(device).unsqueeze(0)).cpu().item()

        # Calculate advantages using GAE
        advantages = []
        last_gae_lam = 0
        num_steps = len(reward_list)
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 0.0
                next_val = next_value
            else:
                next_non_terminal = 1.0 - done_list[t] # use done from current step
                next_val = value_list[t+1] if t + 1 < len(value_list) else next_value

            if not (t < len(reward_list) and t < len(value_list)):
                continue

            delta = reward_list[t] + config['gamma'] * next_val * next_non_terminal - value_list[t]
            last_gae_lam = delta + config['gamma'] * config['gae_lambda'] * next_non_terminal * last_gae_lam
            advantages.insert(0, last_gae_lam)

    # Save the trajectory and advantages
    if len(obs_list) == len(advantages):
        filepath = os.path.join(trajectories_dir, f"episode_{completed_episodes}.npz")
        np.savez(filepath, observations=np.array(obs_list), advantages=np.array(advantages))
        print(f"Saved trajectory for episode {completed_episodes} to {filepath}")
    else:
        print(f"Skipping saving trajectory for episode {completed_episodes} due to length mismatch: obs({len(obs_list)}) vs adv({len(advantages)})")

    env.close()

def watch_episode(path):
    """Plays a video file in a pygame window."""
    video = imageio.get_reader(path, 'ffmpeg')
    
    pygame.init()
    screen = pygame.display.set_mode(video.get_meta_data()['size'])
    pygame.display.set_caption(f"Watching {os.path.basename(path)}")
    
    clock = pygame.time.Clock()
    running = True
    
    for frame in video:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not running:
            break
            
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        clock.tick(30) # Lock to 30 FPS
        
    pygame.quit() 