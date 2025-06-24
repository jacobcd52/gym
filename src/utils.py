import os
import imageio
import numpy as np
import pygame
import torch
import gymnasium as gym

def save_video(frames, folder, filename):
    """Saves a list of frames as an mp4 video."""
    path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    imageio.mimwrite(path, frames, fps=30)

def record_episode(agent, config, run_name, episode_idx):
    """Records a single episode and saves it as a video."""
    print(f"\nRecording episode {episode_idx}...")
    
    # Create a single, non-vectorized environment for recording
    env = gym.make(config['env_id'], render_mode="rgb_array")
    
    # Add the same wrappers as the training environment
    is_atari = "NoFrameskip" in config['env_id']
    if is_atari:
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

    frames = []
    obs, _ = env.reset()
    done = False
    
    while not done:
        frames.append(env.render())
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).unsqueeze(0).to(next(agent.parameters()).device))
        obs, _, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
    save_video(frames, f"videos/{run_name}", f"episode-{episode_idx}.mp4")
    print(f"Episode {episode_idx} recording complete.")
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