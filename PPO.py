import os
import shutil
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ActorCritic import ActorCritic

from customPendulum import StableInit


def record_trigger(episode_id: int) -> bool:

    episodes = [0, 30, 100, 150, 199, 200, 250, 350, 450]

    return episode_id in episodes


def make_env(gym_id, seed, idx, capture_video, run_name, max_episode_steps):
    def thunk():
        env = gym.make(gym_id, render_mode='rgb_array')
        env._max_episode_steps = max_episode_steps
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=record_trigger)
        env = gym.wrappers.NormalizeReward(env)
        if gym_id == "Pendulum-v1":
            env = StableInit(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
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


class PPO:
    def __init__(self,
                 exp_name=os.path.basename(__file__).rstrip(".py"),
                 gym_id="CustomPendulum-v1",
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
        self.gym_id = gym_id
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
        self.torch_deterministic = torch_deterministic
        self.cuda = cuda

        self.total_timesteps = self.num_episodes * self.max_episode_steps * self.num_envs
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        self.run_name = f"{self.gym_id}__{self.exp_name}"
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
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(self.gym_id, self.seed + i, i, self.capture_video, self.run_name, self.max_episode_steps)
             for i in range(self.num_envs)]
        )
        assert isinstance(self.envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

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

        # RESULTS: variables to store results
        num_episodes = self.num_episodes
        cumulative_rewards = np.empty(num_episodes)
        observations = np.empty((num_episodes, self.max_episode_steps, *self.envs.observation_space.shape))
        control_actions = np.empty((num_episodes, self.max_episode_steps, *self.envs.action_space.shape))

        # ALGO Logic: Storage setup
        obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape).to(device)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(device)
        log_probs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        episode = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(self.num_envs).to(device)
        num_updates = int(np.ceil(self.total_timesteps // self.batch_size))

        update = 0
        episode_step = 0

        while episode < num_episodes:
            update += 1

            # for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                log_probs[step] = logprob

                observations[episode][episode_step] = next_obs
                control_actions[episode][episode_step] = action

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
                episode_step += 1
                done = np.logical_or(terminated, truncated)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                if "final_info" in info.keys():
                    episode += 1
                    episode_step = 0
                    episodic_return = np.array([x['episode']['r'] for x in info['final_info']])[0]
                    episodic_length = np.array([x['episode']['l'] for x in info['final_info']])[0]
                    cumulative_rewards[episode - 1] = episodic_return
                    print(f"episode={episode}, episodic_return={episodic_return}")
                    if self.track:
                        self.writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        self.writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                    if episode == num_episodes:
                        break

            advantages, returns = self._compute_advantage(next_obs, rewards, next_done, dones, values)
            self.optimize(obs, log_probs, actions, advantages, returns, values, global_step, start_time)

        if self.track:
            np.savez(f"runs/{self.run_name}/{self.run_name}_training.npz",
                     cumulative_rewards=cumulative_rewards,
                     observations=observations,
                     control_actions=control_actions,
                     )
            torch.save(self.agent.state_dict(), f'runs/{self.run_name}/{self.run_name}_agent.pt')

    def optimize(self, obs, log_probs, actions, advantages, returns, values, global_step, start_time):
        # Flatten the batch
        b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clip_fracs = []

        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, new_log_prob, entropy, new_value = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                      b_actions[mb_inds])
                log_ratio = new_log_prob - b_logprobs[mb_inds]
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    # Calculate approx_kl
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Compute policy loss
                pg_loss = _compute_policy_loss(mb_advantages, ratio, self.clip_coef)

                # Compute value loss
                v_loss = _compute_value_loss(new_value, self.clip_vloss, self.clip_coef, b_returns, b_values,
                                             mb_inds)

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
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

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

    def validate(self, deterministic=False):
        self.agent.eval()
        print(f"VALIDATION {self.run_name}")
        device = self.device

        self.agent.load_state_dict(torch.load(f'runs/{self.run_name}/{self.run_name}_agent.pt'))

        # RESULTS: variables to store results
        num_episodes = self.num_validation_episodes
        cumulative_rewards = np.empty(num_episodes)
        observations = np.empty((num_episodes, self.max_episode_steps, *self.envs.observation_space.shape))
        control_actions = np.empty((num_episodes, self.max_episode_steps, *self.envs.action_space.shape))
        control_actions_std = np.empty((num_episodes, self.max_episode_steps, *self.envs.action_space.shape))
        control_actions_mean = np.empty((num_episodes, self.max_episode_steps, *self.envs.action_space.shape))

        next_obs, _ = self.envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)

        for episode in range(num_episodes):
            episode_step = 0

            while episode_step < self.max_episode_steps:
                with torch.no_grad():
                    action, action_mean, action_std = self.agent.get_action(next_obs, deterministic)

                observations[episode][episode_step] = next_obs
                control_actions[episode][episode_step] = action
                control_actions_std[episode][episode_step] = action_std
                control_actions_mean[episode][episode_step] = action_mean

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
                episode_step += 1
                done = np.logical_or(terminated, truncated)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                if "final_info" in info.keys():
                    episodic_return = np.array([x['episode']['r'] for x in info['final_info']])[0]
                    episodic_length = np.array([x['episode']['l'] for x in info['final_info']])[0]
                    cumulative_rewards[episode - 1] = episodic_return
                    print(f"episode={episode}, episodic_return={episodic_return}")
                    if self.track:
                        self.writer.add_scalar("charts/validation_episodic_return",
                                               episodic_return, episode * int(episodic_length[0]))
                    if episode == num_episodes:
                        break

        if self.track:
            if deterministic:
                save_path = f"runs/{self.run_name}/{self.run_name}_validation_det.npz"
            else:
                save_path = f"runs/{self.run_name}/{self.run_name}_validation.npz"
            np.savez(save_path,
                     cumulative_rewards=cumulative_rewards,
                     observations=observations,
                     control_actions=control_actions,
                     control_actions_std=control_actions_std,
                     control_actions_mean=control_actions_mean,
                     )

    def close(self):
        self.envs.close()
        if self.track:
            self.writer.close()
