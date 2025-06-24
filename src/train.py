import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from .agent import Agent
from .utils import record_episode

class EpisodeTrigger:
    def __init__(self, save_video_freq, video_length):
        self.save_video_freq = save_video_freq
        self.video_length = video_length
        self.episode_count = -1

    def __call__(self, episode_id):
        self.episode_count += 1
        return self.episode_count % self.save_video_freq == 0

def make_env(env_id, seed, idx, capture_video, run_name, video_length, episode_trigger, max_episode_steps):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", 
                                             episode_trigger=episode_trigger, 
                                             video_length=video_length)
        
        # Apply episode length limit if specified
        if max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        
        is_atari = "NoFrameskip" in env_id

        if is_atari:
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env
    return thunk

def train(config):
    run_name = f"{config['env_id']}__{int(time.time())}"
    if config['wandb_project_name']:
        import wandb
        wandb.init(
            project=config['wandb_project_name'],
            entity=config['wandb_entity'],
            sync_tensorboard=True,
            config=config,
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = config['torch_deterministic']

    device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Initial GPU Memory Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        
        # Enable mixed precision training for better GPU utilization
        scaler = GradScaler()
        print("Mixed precision training enabled")
    else:
        scaler = None

    video_trigger = EpisodeTrigger(config['save_video_freq'], config['video_length'])

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(config['env_id'], config['seed'] + i, i, True, run_name, config['video_length'], video_trigger, config['max_episode_steps']) for i in range(config['num_envs'])]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, config).to(device)
    print(f"Agent device: {next(agent.parameters()).device}")
    optimizer = optim.Adam(agent.parameters(), lr=config['learning_rate'], eps=config['optimizer_eps'])

    # ALGO Logic: Storage setup
    obs = torch.zeros((config['num_steps'], config['num_envs']) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config['num_steps'], config['num_envs']) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    rewards = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    dones = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    values = torch.zeros((config['num_steps'], config['num_envs'])).to(device)

    # TRY NOT to MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config['seed'])
    next_obs = torch.Tensor(next_obs).to(device)
    print(f"Initial observation device: {next_obs.device}")
    print(f"Observation shape: {next_obs.shape}")
    next_done = torch.zeros(config['num_envs']).to(device)
    
    completed_episodes = 0
    update = 1
    recent_scores = []  # Track recent scores for printing
    
    while completed_episodes < config['total_episodes']:
        # Print GPU memory usage every 10 updates
        if update % 10 == 0 and torch.cuda.is_available():
            print(f"Update {update}: GPU Memory Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            print(f"Update {update}: GPU Memory Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
            print(f"Update {update}: GPU Memory Utilization: {torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.1f}%")
        
        # Annealing the learning rate if needed
        if config.get('anneal_lr', False):
            frac = 1.0 - completed_episodes / config['total_episodes']
            lrnow = frac * config['learning_rate']
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, config['num_steps']):
            global_step += 1 * config['num_envs']
            obs[step] = next_obs
            dones[step] = next_done
            
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            # Handle episode logging for both old and new Gymnasium versions
            if "final_info" in infos:
                # Old Gymnasium version
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        completed_episodes += 1
                        # Only log if length > 0
                        if info['episode']['l'] > 0:
                            recent_scores.append(info['episode']['r'])  # Track the score
                            if completed_episodes % config['save_video_freq'] == 0 and config['capture_video']:
                                record_episode(agent, config, run_name, completed_episodes)
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                            # Add direct wandb logging to ensure metrics appear
                            if config['wandb_project_name']:
                                import wandb
                                wandb.log({
                                    "episode_return": info["episode"]["r"],
                                    "episode_length": info["episode"]["l"],
                                    "global_step": global_step
                                })
                            
                            # Print score every 50 episodes
                            if completed_episodes % 50 == 0:
                                if len(recent_scores) >= 50:
                                    last_50_scores = recent_scores[-50:]
                                    best_score = max(last_50_scores)
                                    avg_score = sum(last_50_scores) / len(last_50_scores)
                                    # Convert to scalar if needed
                                    best_score = float(best_score) if hasattr(best_score, 'item') else best_score
                                    avg_score = float(avg_score) if hasattr(avg_score, 'item') else avg_score
                                    print(f"\nEpisode: {completed_episodes}, Best Score: {best_score:.2f}, Avg Score: {avg_score:.2f}")
                                else:
                                    best_score = max(recent_scores) if recent_scores else 0.0
                                    avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
                                    # Convert to scalar if needed
                                    best_score = float(best_score) if hasattr(best_score, 'item') else best_score
                                    avg_score = float(avg_score) if hasattr(avg_score, 'item') else avg_score
                                    print(f"\nEpisode: {completed_episodes}, Best Score: {best_score:.2f}, Avg Score: {avg_score:.2f}")
            elif "episode" in infos:
                # New Gymnasium version - episode info is directly in infos
                episode_info = infos["episode"]
                if episode_info is not None:
                    # Handle vectorized environments - episode_info contains arrays
                    if isinstance(episode_info, dict) and "r" in episode_info:
                        # Check if any episodes completed (non-None values)
                        returns = episode_info["r"]
                        lengths = episode_info["l"]
                        
                        # Log each completed episode
                        for i, (ret, length) in enumerate(zip(returns, lengths)):
                            if length > 0:  # Only log real completed episodes
                                completed_episodes += 1
                                recent_scores.append(ret)  # Track the score
                                
                                if completed_episodes % config['save_video_freq'] == 0 and config['capture_video']:
                                    record_episode(agent, config, run_name, completed_episodes)
                                
                                writer.add_scalar("charts/episodic_return", ret, global_step)
                                writer.add_scalar("charts/episodic_length", length, global_step)
                                # Add direct wandb logging to ensure metrics appear
                                if config['wandb_project_name']:
                                    import wandb
                                    wandb.log({
                                        "episode_return": ret,
                                        "episode_length": length,
                                        "global_step": global_step
                                    })
                                
                                # Print score every 50 episodes
                                if completed_episodes % 50 == 0:
                                    if len(recent_scores) >= 50:
                                        last_50_scores = recent_scores[-50:]
                                        best_score = max(last_50_scores)
                                        avg_score = sum(last_50_scores) / len(last_50_scores)
                                        # Convert to scalar if needed
                                        best_score = float(best_score) if hasattr(best_score, 'item') else best_score
                                        avg_score = float(avg_score) if hasattr(avg_score, 'item') else avg_score
                                        print(f"\nEpisode: {completed_episodes}, Best Score: {best_score:.2f}, Avg Score: {avg_score:.2f}")
                                    else:
                                        best_score = max(recent_scores) if recent_scores else 0.0
                                        avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
                                        # Convert to scalar if needed
                                        best_score = float(best_score) if hasattr(best_score, 'item') else best_score
                                        avg_score = float(avg_score) if hasattr(avg_score, 'item') else avg_score
                                        print(f"\nEpisode: {completed_episodes}, Best Score: {best_score:.2f}, Avg Score: {avg_score:.2f}")

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config['num_steps'])):
                if t == config['num_steps'] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config['gamma'] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config['gamma'] * config['gae_lambda'] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config['num_envs'] * config['num_steps'])
        clipfracs = []
        for epoch in range(config['update_epochs']):
            np.random.shuffle(b_inds)
            batch_size = config['num_envs'] * config['num_steps']
            minibatch_size = config['minibatch_size']
            num_minibatches = batch_size // minibatch_size
            
            if update == 1 and epoch == 0:  # Print only once
                print(f"Batch size: {batch_size}, Minibatch size: {minibatch_size}, Number of minibatches: {num_minibatches}")
                
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config['clip_coef']).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config['clip_coef'], 1 + config['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config['ent_coef'] * entropy_loss + v_loss * config['vf_coef']

                optimizer.zero_grad()
                
                # Use mixed precision training if available
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(agent.parameters(), config['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config['max_grad_norm'])
                    optimizer.step()

            if config.get('target_kl') is not None:
                if approx_kl > config['target_kl']:
                    break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("charts/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        # Add direct wandb logging for all metrics
        if config['wandb_project_name']:
            import wandb
            wandb.log({
                "learning_rate": optimizer.param_groups[0]["lr"],
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy": entropy_loss.item(),
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": approx_kl.item(),
                "clipfrac": np.mean(clipfracs),
                "explained_variance": explained_var,
                "SPS": int(global_step / (time.time() - start_time)),
                "global_step": global_step
            })
        
        update += 1
        
    # Save model
    model_path = f"runs/{run_name}/{config['env_id']}.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"model saved to {model_path}")

    # Close environments (with error handling for wandb compatibility)
    try:
        envs.close()
    except AttributeError as e:
        print(f"Warning: Error closing environments: {e}")
        print("This is a known compatibility issue between wandb and RecordVideo wrappers.")
    
    writer.close() 