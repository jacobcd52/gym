import os
import random
import time
from distutils.util import strtobool
import gc
from typing import Optional

import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.time_limit import TimeLimit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from .agent import Agent
from .utils import record_episode

def make_env(env_id, seed, idx, run_name, max_episode_steps):
    def thunk():
        if "NoFrameskip" in env_id:
            env = gym.make(env_id, render_mode="rgb_array", full_action_space=False)
            env = RecordEpisodeStatistics(env)
            env = AtariPreprocessing(env, frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=True)
            env = FrameStack(env, 4)
        else:
            env = gym.make(env_id, render_mode="rgb_array")
            env = RecordEpisodeStatistics(env)

        # Apply episode length limit if specified
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)

        env.action_space.seed(seed)
        return env
    return thunk

class Trainer:
    def __init__(self, config):
        self.config = config
        self.run_name = f"{config['env_id']}__{int(time.time())}"
        
        self.writer: SummaryWriter
        self.device: torch.device
        self.scaler: Optional[GradScaler] = None
        self.envs: gym.vector.SyncVectorEnv
        self.agent: Agent
        self.optimizer: optim.Optimizer
        
        self.obs_space_shape: tuple
        self.action_space_shape: tuple

        self.obs: torch.Tensor
        self.actions: torch.Tensor
        self.logprobs: torch.Tensor
        self.rewards: torch.Tensor
        self.dones: torch.Tensor
        self.values: torch.Tensor

        self.global_step = 0
        self.start_time = time.time()
        self.next_obs: torch.Tensor
        self.next_done: torch.Tensor
        self.completed_episodes = 0
        self.update = 1
        self.recent_scores = []

        self.setup()

    def setup(self):
        self._setup_logging()
        self._setup_pytorch()
        self._setup_environment()
        self._setup_agent()
        self._setup_storage()
        self._setup_trajectory_saving()

    def _setup_logging(self):
        if self.config['wandb_project_name']:
            import wandb
            wandb.init(
                project=self.config['wandb_project_name'],
                entity=self.config['wandb_entity'],
                sync_tensorboard=True,
                config=self.config,
                name=self.run_name,
                monitor_gym=False,
                save_code=True,
            )
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.config.items()])),
        )

    def _setup_pytorch(self):
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.backends.cudnn.deterministic = self.config['torch_deterministic']
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config['cuda'] else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Initial GPU Memory Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            self.scaler = GradScaler()
            print("Mixed precision training enabled")

    def _setup_environment(self):
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(self.config['env_id'], self.config['seed'] + i, i, self.run_name, self.config['max_episode_steps'])
             for i in range(self.config['num_envs'])]
        )
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        assert self.envs.single_observation_space is not None, "single_observation_space is None"
        assert self.envs.single_action_space is not None, "single_action_space is not None"

        obs_shape = self.envs.single_observation_space.shape
        action_shape = self.envs.single_action_space.shape

        assert obs_shape is not None, "observation space shape cannot be None"
        assert action_shape is not None, "action space shape cannot be None"

        self.obs_space_shape = obs_shape
        self.action_space_shape = action_shape

    def _setup_agent(self):
        self.agent = Agent(self.envs, self.config).to(self.device)
        print(f"Agent device: {next(self.agent.parameters()).device}")
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.config['learning_rate'], eps=self.config['optimizer_eps'])

    def _setup_storage(self):
        self.obs = torch.zeros((self.config['num_steps'], self.config['num_envs']) + self.obs_space_shape).to(self.device)
        self.actions = torch.zeros((self.config['num_steps'], self.config['num_envs']) + self.action_space_shape).to(self.device)
        self.logprobs = torch.zeros((self.config['num_steps'], self.config['num_envs'])).to(self.device)
        self.rewards = torch.zeros((self.config['num_steps'], self.config['num_envs'])).to(self.device)
        self.dones = torch.zeros((self.config['num_steps'], self.config['num_envs'])).to(self.device)
        self.values = torch.zeros((self.config['num_steps'], self.config['num_envs'])).to(self.device)

    def _setup_trajectory_saving(self):
        if self.config.get('save_trajectories', False):
            self.trajectory_path = f"trajectories/{self.run_name}"
            os.makedirs(self.trajectory_path, exist_ok=True)
            self.update_obs_buffer = []
            self.update_advantages_buffer = []
    
    def train(self):
        self.start_time = time.time()
        self.next_obs, _ = self.envs.reset(seed=self.config['seed'])
        self.next_obs = torch.Tensor(self.next_obs).to(self.device)
        print(f"Initial observation device: {self.next_obs.device}")
        print(f"Observation shape: {self.next_obs.shape}")
        self.next_done = torch.zeros(self.config['num_envs']).to(self.device)

        while self.completed_episodes < self.config['total_episodes']:
            update_start_time = time.time()
            self._log_gpu_memory()
            self._anneal_lr()
            
            rollout_start_time = time.time()
            self._collect_rollout()
            rollout_time = time.time() - rollout_start_time
            
            advantages_start_time = time.time()
            returns, advantages = self._compute_advantages()
            advantages_time = time.time() - advantages_start_time
            
            trajectory_save_start_time = time.time()
            self._save_trajectories(advantages)
            trajectory_save_time = time.time() - trajectory_save_start_time

            optim_start_time = time.time()
            self._update_policy(returns, advantages)
            optim_time = time.time() - optim_start_time

            update_time = time.time() - update_start_time
            self._log_update_timings(rollout_time, advantages_time, trajectory_save_time, optim_time, update_time)

            self.update += 1
        
        self.cleanup()

    def _collect_rollout(self):
        for step in range(self.config['num_steps']):
            self.global_step += self.config['num_envs']
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            self.next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            self.next_obs = torch.Tensor(self.next_obs).to(self.device)
            self.next_done = torch.Tensor(done).to(self.device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        self.completed_episodes += 1
                        self._log_episode_stats(info)

    def _compute_advantages(self):
        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.config['num_steps'])):
                if t == self.config['num_steps'] - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.config['gamma'] * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.config['gamma'] * self.config['gae_lambda'] * nextnonterminal * lastgaelam
            returns = advantages + self.values
        return returns, advantages

    def _save_trajectories(self, advantages):
        if self.config.get('save_trajectories', False):
            self.update_obs_buffer.append(self.obs.cpu())
            self.update_advantages_buffer.append(advantages.cpu())

            if self.update % self.config['save_every_n_updates'] == 0:
                all_obs = torch.cat(self.update_obs_buffer, dim=0)
                all_advantages = torch.cat(self.update_advantages_buffer, dim=0)
                
                states_to_save = all_obs[::self.config['trajectory_save_every_n_frames'], :self.config['n_envs_to_save']]
                advantages_to_save = all_advantages[::self.config['trajectory_save_every_n_frames'], :self.config['n_envs_to_save']]

                trajectory = {
                    "states": states_to_save.numpy().astype(np.uint8),
                    "advantages": advantages_to_save.numpy().astype(np.float16),
                }
                start_update_num = self.update - self.config['save_every_n_updates']
                save_path = f"{self.trajectory_path}/updates_{start_update_num}_to_{self.update-1}.npz"
                np.savez_compressed(save_path, **trajectory)
                print(f"Saved trajectory for updates {start_update_num} to {self.update-1} to {save_path}")

                del all_obs, all_advantages, states_to_save, advantages_to_save, trajectory
                self.update_obs_buffer = []
                self.update_advantages_buffer = []
                gc.collect()
    
    def _update_policy(self, returns, advantages):
        b_obs = self.obs.reshape((-1,) + self.obs_space_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.action_space_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        b_inds = np.arange(self.config['num_envs'] * self.config['num_steps'])
        clipfracs = []
        
        for epoch in range(self.config['update_epochs']):
            np.random.shuffle(b_inds)
            batch_size = self.config['num_envs'] * self.config['num_steps']
            minibatch_size = self.config['minibatch_size']
            
            if self.update == 1 and epoch == 0:
                print(f"Batch size: {batch_size}, Minibatch size: {minibatch_size}, Number of minibatches: {batch_size // minibatch_size}")

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config['clip_coef']).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config['clip_coef'], 1 + self.config['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.config['ent_coef'] * entropy_loss + v_loss * self.config['vf_coef']

                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config['max_grad_norm'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config['max_grad_norm'])
                    self.optimizer.step()

            if self.config.get('target_kl') and approx_kl > self.config['target_kl']:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        self._log_training_stats(v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var)

    def _log_gpu_memory(self):
        if self.update % 10 == 0 and torch.cuda.is_available():
            print(f"\n--- Update {self.update} ---")
            print(f"Update {self.update}: GPU Memory Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            print(f"Update {self.update}: GPU Memory Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
            print(f"Update {self.update}: GPU Memory Utilization: {torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.1f}%")

    def _anneal_lr(self):
        if self.config.get('anneal_lr', False):
            frac = 1.0 - self.completed_episodes / self.config['total_episodes']
            lrnow = frac * self.config['learning_rate']
            self.optimizer.param_groups[0]["lr"] = lrnow

    def _log_episode_stats(self, info):
        episode_return = info["episode"]["r"]
        episode_length = info["episode"]["l"]
        self.recent_scores.append(episode_return)
        
        self.writer.add_scalar("charts/episodic_return", episode_return, self.global_step)
        self.writer.add_scalar("charts/episodic_length", episode_length, self.global_step)
        
        if self.config['wandb_project_name']:
            import wandb
            wandb.log({
                "episode_return": episode_return,
                "episode_length": episode_length,
                "completed_episodes": self.completed_episodes,
            }, step=self.global_step)
        
        if self.completed_episodes > 0 and self.completed_episodes % 50 == 0:
            if len(self.recent_scores) > 0:
                avg_score = np.mean(self.recent_scores[-50:])
                print(f"\nEpisodes {self.completed_episodes-49}-{self.completed_episodes}: Avg Return={avg_score:.2f}")

    def _log_training_stats(self, v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var):
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.writer.add_scalar("charts/explained_variance", explained_var, self.global_step)
        self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
        
        if self.config['wandb_project_name']:
            import wandb
            wandb.log({
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy": entropy_loss.item(),
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": approx_kl.item(),
                "clipfrac": np.mean(clipfracs),
                "explained_variance": explained_var,
                "SPS": int(self.global_step / (time.time() - self.start_time)),
                "global_step": self.global_step
            })
            
    def _log_update_timings(self, rollout_time, advantages_time, trajectory_save_time, optim_time, update_time):
        if self.update % 10 == 0:
            print(f"Update {self.update} timings:")
            print(f"  Rollout: {rollout_time:.4f}s")
            print(f"  Advantage Calculation: {advantages_time:.4f}s")
            if self.config.get('save_trajectories', False):
                print(f"  Trajectory Saving: {trajectory_save_time:.4f}s")
            print(f"  Optimization: {optim_time:.4f}s")
            print(f"  Total Update: {update_time:.4f}s")
            print(f"  SPS: {int(self.global_step / (time.time() - self.start_time))}")
            print(f"--------------------")

    def cleanup(self):
        model_path = f"runs/{self.run_name}/{self.config['env_id']}.pt"
        torch.save(self.agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        try:
            self.envs.close()
        except AttributeError as e:
            print(f"Warning: Error closing environments: {e}")
            print("This is a known compatibility issue between wandb and RecordVideo wrappers.")
        
        self.writer.close()

        if self.config['wandb_project_name']:
            import wandb
            wandb.finish()


def train(config):
    trainer = Trainer(config)
    trainer.train()

    if config['wandb_project_name']:
        import wandb
        wandb.finish() 