import os
import shutil
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import FlattenObservation
from torch.utils.tensorboard import SummaryWriter

from ActorCriticDiscrete import ActorCritic

from shepherding.wrappers import LowLevelPPOPolicy, TerminateWhenSuccessful


def record_trigger(episode_id: int) -> bool:

    episodes = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 10050, 10100, 10120, 10180]
    record = ((episode_id + 1000) % 1000 == 0)
    val_ep = [10010, 10020, 10040, 10060, 10080, 10100, 10120, 10140, 10160, 10180]
    # val_ep = np.arange(0, 201, 10)
    # val_ep = np.arange(0, 10)
    # return episode_id in episodes
    return record or (episode_id in val_ep)


def make_env(gym_id, seed, idx, capture_video, run_name, max_episode_steps, env_params, random_target=False, render=False):
    def thunk():
        if render:
            render_mode = 'human'
        else:
            render_mode = 'rgb_array'
        if gym_id == "Shepherding-v0":
            env = gym.make(gym_id, render_mode=render_mode, parameters=env_params, rand_target=random_target)
            env._max_episode_steps = max_episode_steps
            if env_params['termination']:
                env = TerminateWhenSuccessful(env, num_steps=200)
            if capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=record_trigger)
            env = FlattenObservation(env)
            env = LowLevelPPOPolicy(env, env_params['target_selection_rate'])
        # Alternative environments to the shepherding
        else:
            env = gym.make(gym_id, render_mode='rgb_array')
            env._max_episode_steps = max_episode_steps
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # env = gym.wrappers.NormalizeReward(env)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
        return env
    return thunk


def _compute_policy_loss(mb_advantages, ratio, clip_coef):
    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    return torch.max(pg_loss1, pg_loss2).mean()


def _compute_value_loss(new_value, clip_vloss, clip_coef, b_returns, b_values, mb_inds):
    new_value = new_value.view(-1)
    if clip_vloss:
        v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
        v_clipped = b_values[mb_inds] + torch.clamp(new_value - b_values[mb_inds], -clip_coef, clip_coef)
        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        return 0.5 * v_loss_max.mean()
    else:
        return 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()


def _remove_dir(directory):
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path) and item.startswith("events"):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)


def reshape_tensor(tensor, episode_length):
    slices = tensor.split(episode_length, dim=0)

    # List to store the reshaped slices
    reshaped_slices = []
    for s in slices:
        reshaped_s = s.permute(1, 0, 2)
        reshaped_slices.append(reshaped_s)

    # Combine the reshaped slices back into a single tensor
    result = torch.cat(reshaped_slices[:-1], dim=0)
    return result


class PPO:
    def __init__(self,
                 exp_name=os.path.basename(__file__).rstrip(".py"),
                 exp_name_val=os.path.basename(__file__).rstrip(".py"),
                 gym_id="CustomPendulum-v1",
                 gym_params=None,
                 max_episode_steps=400,
                 learning_rate=1e-3,
                 seed=1,
                 total_timesteps=int(4 * 1e4),
                 num_episodes=200,
                 num_validation_episodes=100,
                 torch_deterministic=True,
                 cuda=True,
                 track=False,
                 capture_video=False,
                 render=False,
                 num_envs=1,
                 num_steps=1024,
                 anneal_lr=True,
                 gae=True,
                 gamma=0.9,
                 gae_lambda=0.95,
                 num_minibatches=64,
                 update_epochs=10,
                 norm_adv=False,
                 clip_coef=0.2,
                 clip_vloss=True,
                 ent_coef=0.0,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 target_kl=0.01, ):
        self.exp_name = exp_name
        self.exp_name_val = exp_name_val
        self.gym_id = gym_id
        self.gym_params = gym_params
        self.max_episode_steps = max_episode_steps
        self.num_episodes = num_episodes
        self.num_validation_episodes = num_validation_episodes
        self.learning_rate = learning_rate
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gae = gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.track = track
        self.capture_video = capture_video
        self.render = render
        self.torch_deterministic = torch_deterministic
        self.cuda = cuda

        self.total_timesteps = self.num_episodes * self.max_episode_steps
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        self.run_name = f"{self.gym_id}__{self.exp_name}"
        self.run_name_val = f"{self.gym_id}__{self.exp_name_val}"
        self.save_dir = f"runs/{self.run_name}"

        if self.track:
            _remove_dir(self.save_dir)
            self.writer = SummaryWriter(f"runs/{self.run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self).items()])), )

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")

        # env setup
        self.envs = gym.vector.AsyncVectorEnv(
            [make_env(gym_id=self.gym_id, seed=self.seed + i, idx=i, capture_video=self.capture_video,
                      run_name=self.run_name, max_episode_steps=self.max_episode_steps,
                      env_params=self.gym_params, random_target=self.gym_params['random_targets'], render=self.render)
             for i in range(self.num_envs)]
        )
        assert isinstance(self.envs.single_action_space, gym.spaces.MultiDiscrete), "only discrete action space is supported"

        self.agent = ActorCritic(self.envs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

    def _compute_advantage(self, next_obs, rewards, next_done, dones, values):
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            next_non_terminal = 1.0 - torch.cat((dones[1:], next_done.unsqueeze(0)))
            values = torch.cat((values, next_value), dim=0)

            if self.gae:
                deltas = rewards + self.gamma * values[1:] * next_non_terminal - values[:-1]
                advantages = torch.zeros_like(rewards).to(self.device)
                advantages[-1] = deltas[-1]
                for t in reversed(range(self.num_steps - 1)):
                    advantages[t] = deltas[t] + self.gamma * self.gae_lambda * next_non_terminal[t] * advantages[t + 1]
                returns = advantages + values[:-1]
            else:
                returns = torch.zeros_like(rewards).to(self.device)
                returns[-1] = rewards[-1] + self.gamma * next_non_terminal[-1] * next_value
                for t in reversed(range(self.num_steps - 1)):
                    returns[t] = rewards[t] + self.gamma * next_non_terminal[t] * returns[t + 1]
                advantages = returns - values[:-1]

        return advantages, returns

    def train(self):
        self.agent.train()
        print(f"TRAINING {self.run_name}")
        device = self.device

        # Load agent state if needed
        # self.agent.load_state_dict(torch.load(f'runs/{self.run_name}/{self.run_name}_agent.pt', map_location=torch.device(self.device)))

        num_episodes = self.num_episodes
        num_envs = self.num_envs

        # Initialize variables
        global_step = 0
        episode = 0
        start_time = time.time()

        next_obs, _ = self.envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(num_envs).to(device)
        num_updates = int(np.ceil(self.total_timesteps / self.batch_size)) + 2

        # For logging cumulative rewards and settling times
        cumulative_rewards = []
        settling_times = []

        # ALGO Logic: Storage setup
        # We keep the storage tensors for optimization purposes
        obs = torch.zeros((self.num_steps, num_envs) + self.envs.single_observation_space.shape).to(device)
        actions = torch.zeros((self.num_steps, num_envs) + self.envs.single_action_space.shape).to(device)
        log_probs = torch.zeros((self.num_steps, num_envs)).to(device)
        rewards = torch.zeros((self.num_steps, num_envs)).to(device)
        dones = torch.zeros((self.num_steps, num_envs)).to(device)
        values = torch.zeros((self.num_steps, num_envs)).to(device)

        update = 0

        while episode < num_episodes:
            update += 1

            # Annealing the learning rate if instructed to do so
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(self.num_steps):
                global_step += num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                log_probs[step] = logprob

                # Execute the action in the environment
                next_obs_np, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs = torch.Tensor(next_obs_np).to(device)
                next_done = torch.Tensor(done).to(device)

                # Collect per-episode metrics when episodes finish
                if "final_info" in info:
                    for env_idx, final_info in enumerate(info["final_info"]):
                        if final_info is not None:
                            episodic_return = final_info['episode']['r']
                            settling_time = final_info.get('settling_time', None)
                            cumulative_rewards.append(episodic_return)
                            settling_times.append(settling_time)
                            episode += 1
                            print(f"Episode {episode}, Env {env_idx}, Reward: {episodic_return}")

                            if self.track:
                                # Log metrics if needed
                                pass

                            if episode >= num_episodes:
                                break  # Exit if we've collected enough episodes

                if episode >= num_episodes:
                    break  # Exit the time step loop if we've collected enough episodes

            # Compute advantages and returns
            advantages, returns = self._compute_advantage(next_obs, rewards, next_done, dones, values)

            # Optimize the agent using the collected data
            self.optimize(obs, log_probs, actions, advantages, returns, values, global_step, start_time)

        # Save training data if tracking is enabled
        if self.track:
            cumulative_rewards = np.array(cumulative_rewards)
            settling_times = np.array(settling_times)
            np.savez(f"runs/{self.run_name}/{self.run_name}_training.npz",
                     cumulative_rewards=cumulative_rewards,
                     settling_times=settling_times)
            torch.save(self.agent.state_dict(), f'runs/{self.run_name}/{self.run_name}_agent.pt')

    def optimize(self, obs, log_probs, actions, advantages, returns, values, global_step, start_time):
        device = self.device  # Ensuring all operations are on the correct device
        # Flatten the batch
        b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape).to(device)
        b_logprobs = log_probs.reshape(-1).to(device)
        b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape).to(device)
        b_advantages = advantages.reshape(-1).to(device)
        b_returns = returns.reshape(-1).to(device)
        b_values = values.reshape(-1).to(device)

        # Optimizing the policy and value network
        b_inds = torch.arange(obs.size(0), device=device)
        clip_fracs = []

        for epoch in range(self.update_epochs):
            perm = torch.randperm(self.batch_size, device=device)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = perm[start:end]

                _, new_log_prob, entropy, new_value = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                      b_actions.long()[mb_inds].T)
                new_log_prob = new_log_prob.to(device)
                entropy = entropy.to(device)
                new_value = new_value.to(device)
                log_ratio = new_log_prob - b_logprobs[mb_inds]
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    # Calculate approx_kl
                    old_approx_kl = (-log_ratio).mean().to(device)
                    approx_kl = ((ratio - 1) - log_ratio).mean().to(device)
                    clip_fracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Compute policy loss
                pg_loss = _compute_policy_loss(mb_advantages, ratio, self.clip_coef).to(device)

                # Compute value loss
                v_loss = _compute_value_loss(new_value, self.clip_vloss, self.clip_coef, b_returns, b_values,
                                             mb_inds).to(device)

                # Compute entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                # Optimize the agent's parameters
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        # Calculate explained variance
        var_y = torch.var(b_returns)
        explained_var = torch.tensor(float(0), device=device) if var_y == 0 else 1 - torch.var(
            b_returns - b_values) / var_y
        explained_var = explained_var.item()

        # Record rewards for plotting purposes
        if self.track:
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    def validate(self):
        self.agent.eval()
        print(f"VALIDATION {self.run_name}")
        device = self.device

        self.agent.load_state_dict(
            torch.load(f'runs/{self.run_name}/{self.run_name}_agent.pt', map_location=torch.device(self.device)))

        # RESULTS: variables to store results
        num_episodes = self.num_validation_episodes
        num_envs = self.num_envs

        # Initialize episode storage
        episodes = []  # List to store episodes in the order they start
        current_episodes = [None for _ in range(num_envs)]  # Current episode data per environment

        episodes_finished = 0

        next_obs, _ = self.envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)

        while episodes_finished < num_episodes:
            with torch.no_grad():
                action = self.agent.get_action(next_obs)

            # Execute the action in the environment
            next_obs_np, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs_np).to(device)

            done = np.logical_or(terminated, truncated)

            # For each environment
            for env_idx in range(num_envs):
                # If starting a new episode
                if current_episodes[env_idx] is None:
                    # Create a new episode and add it to episodes list
                    current_episode = {
                        'observations': [],
                        'actions': [],
                        'rewards': []
                    }
                    episodes.append(current_episode)
                    current_episodes[env_idx] = current_episode

                # Add data to current episode
                current_episodes[env_idx]['observations'].append(next_obs_np[env_idx])  # Store the raw numpy array
                current_episodes[env_idx]['actions'].append(action[env_idx].cpu().numpy())
                current_episodes[env_idx]['rewards'].append(reward[env_idx])

                if done[env_idx]:
                    # Finalize the episode
                    episodic_return = sum(current_episodes[env_idx]['rewards'])
                    episodes_finished += 1
                    print(f"Episode {episodes_finished} finished in env {env_idx}, reward = {episodic_return}")

                    # Reset current episode for this environment
                    current_episodes[env_idx] = None

                    if episodes_finished >= num_episodes:
                        break  # Exit if we've collected enough episodes

            if episodes_finished >= num_episodes:
                break  # Exit outer loop as well

        # Now process the episodes

        # Optionally pad the sequences to the maximum episode length
        max_length = max(len(episode['observations']) for episode in episodes)

        num_collected_episodes = len(episodes)

        # Prepare arrays for padded data
        obs_shape = episodes[0]['observations'][0].shape
        action_shape = episodes[0]['actions'][0].shape

        obs_padded = np.zeros((num_collected_episodes, max_length) + obs_shape, dtype=np.float32)
        actions_padded = np.zeros((num_collected_episodes, max_length) + action_shape, dtype=np.float32)
        rewards_padded = np.zeros((num_collected_episodes, max_length), dtype=np.float32)
        cumulative_rewards = np.zeros(num_collected_episodes, dtype=np.float32)

        for i, episode in enumerate(episodes):
            length = len(episode['observations'])
            obs_padded[i, :length] = np.stack(episode['observations'])
            actions_padded[i, :length] = np.stack(episode['actions'])
            rewards_padded[i, :length] = np.array(episode['rewards'])
            cumulative_rewards[i] = sum(episode['rewards'])

        # Save the data
        if self.track:
            save_path = f"runs/{self.run_name}/{self.run_name}_validation.npz"
            np.savez(save_path,
                     cumulative_rewards=cumulative_rewards,
                     observations=obs_padded,
                     control_actions=actions_padded)

    def close(self):
        self.envs.close()
        if self.track:
            self.writer.close()
