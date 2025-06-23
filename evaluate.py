import argparse
import torch
import gymnasium as gym
import yaml
from src.agent import Agent

def evaluate(config_path, model_path, env_id, video_folder):
    """
    Evaluates a trained agent for one episode and saves a video.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override the env_id from the config if a new one is provided
    if env_id is None:
        env_id = config['env_id']

    # Create a single environment for evaluation
    env = gym.make(env_id, render_mode="rgb_array")
    
    # Create a unique name for the video folder
    run_name = f"{env_id}_eval_{model_path.split('/')[-2]}"
    video_path = f"{video_folder}/{run_name}"
    
    # Wrap the environment to record a video of the first episode
    env = gym.wrappers.RecordVideo(env, video_folder=video_path, episode_trigger=lambda e: e == 0)

    # Apply the same observation wrappers as used during training
    is_atari = "NoFrameskip" in env_id
    if is_atari:
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

    # The agent expects a vectorized environment, so we wrap the single env
    envs = gym.vector.SyncVectorEnv([lambda: env])

    device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")
    
    # Instantiate and load the trained agent
    agent = Agent(envs, config).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    print(f"Evaluating agent in '{env_id}'...")
    print(f"A video of the first episode will be saved to: {video_path}")

    obs, _ = envs.reset()
    done = False
    while not done:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        obs, _, terminated, truncated, info = envs.step(action.cpu().numpy())
        done = terminated[0] or truncated[0]

    # The video is saved automatically when the episode ends and the env is closed
    env.close()
    print("Evaluation finished.")
    if "episode" in info:
        print(f"Final score: {info['episode']['r'][0]:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent and save a video.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the training configuration file.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model (.pt) file.")
    parser.add_argument("--env-id", type=str, default=None, help="Gym environment ID to override the one in the config.")
    parser.add_argument("--video-folder", type=str, default="videos/", help="Directory to save the evaluation video.")
    args = parser.parse_args()
    
    evaluate(args.config, args.model_path, args.env_id, args.video_folder) 