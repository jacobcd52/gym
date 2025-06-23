import os
import imageio
import numpy as np
import pygame

def save_episode_as_gif(frames, folder, filename):
    """Saves a list of frames as a gif."""
    path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    imageio.mimsave(path, frames, fps=30)

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