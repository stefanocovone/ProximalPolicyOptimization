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
from shepherding.utils.control_rules import herder_actions
from torch.utils.tensorboard import SummaryWriter

from ActorCritic import ActorCritic

from customPendulum import StableInit
from shepherding.wrappers import DeterministicReset, SingleAgentReward, FlattenAction


def record_trigger(episode_id: int) -> bool:
    episodes = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 10050, 10100, 10120, 10180]
    record = (episode_id % 1000 == 0)
    val_ep = [10010, 10020, 10040, 10060, 10080, 10100, 10120, 10140, 10160, 10180]
    val_ep = np.arange(0, 201, 10)
    val_ep = np.arange(0, 10)
    # return episode_id in episodes
    return record or (episode_id in val_ep)


def make_env(gym_id, seed, idx, capture_video, run_name, max_episode_steps, env_params):
    def thunk():
        if gym_id == "Shepherding-v0":
            env = gym.make(gym_id, render_mode='rgb_array', parameters=env_params)
            env._max_episode_steps = max_episode_steps
            # env = DeterministicReset(env)
            env = FlattenAction(env)
            env = SingleAgentReward(env, k_4=3)
            env = FlattenObservation(env)
        else:
            env = gym.make(gym_id, render_mode='rgb_array')
            env._max_episode_steps = max_episode_steps
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=record_trigger)
        # env = gym.wrappers.NormalizeReward(env)
        if gym_id == "Pendulum-v1":
            env = StableInit(env)
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


class PPO:
    def __init__(self,
                 exp_name=os.path.basename(__file__).rstrip(".py"),
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
        self.torch_deterministic = torch_deterministic
        self.cuda = cuda

        self.total_timesteps = self.num_episodes * self.max_episode_steps
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
        self.envs = gym.vector.AsyncVectorEnv(
            [make_env(self.gym_id, self.seed + i, i, self.capture_video,
                      self.run_name, self.max_episode_steps, self.gym_params)
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
        # cumulative_rewards = np.empty(num_episodes)
        # observations = torch.zeros((num_episodes, self.max_episode_steps, *self.envs.observation_space.shape)).to(device)
        # control_actions = torch.zeros((num_episodes, self.max_episode_steps, *self.envs.action_space.shape)).to(device)

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
        num_updates = int(np.ceil(self.total_timesteps // self.batch_size)) + 2

        # observations = torch.zeros((self.num_steps * num_updates,
        #                             self.num_envs) + self.envs.single_observation_space.shape).to(device)
        # cumulative_rewards = torch.zeros((self.num_steps * num_updates,
        #                                  self.num_envs)).to(device)

        cum_rewards = np.zeros(self.num_episodes + 100)
        settling_times = np.zeros(self.num_episodes + 100)

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

                # TRY NOT TO MODIFY: execute the game and log observations.
                next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
                episode_step += 1
                done = np.logical_or(terminated, truncated)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                if "final_info" in info.keys():
                    episode += self.num_envs
                    episode_step = 0
                    episodic_return = np.array([x['episode']['r'] for x in info['final_info']])
                    set_times = np.array([x['settling_time'] for x in info['final_info']])
                    # cumulative_rewards[episode - 1] = episodic_return
                    episode_labels = np.array(
                        [f"episode={episode - 7 + i}, reward = " for i in range(len(episodic_return))])
                    print_strings = [label + str(value) for label, value in zip(episode_labels, episodic_return)]
                    print("\n".join(print_strings))
                    # print(f"episode={episode}, episodic_return={episodic_return}")
                    cum_rewards[(episode - self.num_envs):episode] = episodic_return.squeeze()
                    settling_times[(episode - self.num_envs):episode] = set_times.squeeze()
                    if self.track:
                        pass
                        # self.writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    if episode >= num_episodes:
                        break

            # observations[(update-1)*self.num_steps:(update-1)*self.num_steps + self.num_steps] = obs
            # cumulative_rewards[(update - 1) * self.num_steps:(update - 1) * self.num_steps + self.num_steps] = rewards

            advantages, returns = self._compute_advantage(next_obs, rewards, next_done, dones, values)
            self.optimize(obs, log_probs, actions, advantages, returns, values, global_step, start_time)

            if self.track and (episode + 1) % 1000 == 0:
                # np.savez(f"runs/{self.run_name}/{self.run_name}_training.npz",
                #         cumulative_rewards=cumulative_rewards,
                #         observations=observations.cpu().numpy(),
                #         control_actions=control_actions.cpu().numpy(),
                #         )
                torch.save(self.agent.state_dict(), f'runs/{self.run_name}/{self.run_name}_agent.pt')

        # if self.track:
        #     observations = self.reshape_tensor(observations)[:self.num_episodes, :, :]
        #     cumulative_rewards = self.reshape_tensor(
        #         cumulative_rewards.unsqueeze(dim=-1)).squeeze().sum(dim=1)[:self.num_episodes]
        #     np.savez(f"runs/{self.run_name}/{self.run_name}_training.npz",
        #              cumulative_rewards=cumulative_rewards.cpu().numpy(),
        #              observations=observations.cpu().numpy(),
        #              )
        #     torch.save(self.agent.state_dict(), f'runs/{self.run_name}/{self.run_name}_agent.pt')
        if self.track:
            cum_rewards = cum_rewards[:self.num_episodes]
            settling_times = settling_times[:self.num_episodes]
            np.savez(f"runs/{self.run_name}/{self.run_name}_training.npz",
                     cumulative_rewards=cum_rewards,
                     settling_times=settling_times,
                     )
            torch.save(self.agent.state_dict(), f'runs/{self.run_name}/{self.run_name}_agent.pt')

    def reshape_tensor(self, tensor):
        slices = tensor.split(self.max_episode_steps, dim=0)

        # List to store the reshaped slices
        reshaped_slices = []
        for s in slices:
            reshaped_s = s.permute(1, 0, 2)
            reshaped_slices.append(reshaped_s)

        # Combine the reshaped slices back into a single tensor
        result = torch.cat(reshaped_slices[:-1], dim=0)
        return result

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
                                                                                      b_actions[mb_inds])
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

        env = make_env(self.gym_id, self.seed, 0, self.capture_video,
                       self.run_name, self.max_episode_steps, self.gym_params)
        env = env()

        self.agent.load_state_dict(
            torch.load(f'runs/{self.run_name}/{self.run_name}_agent.pt', map_location=torch.device(self.device)))

        # RESULTS: variables to store results
        num_episodes = self.num_validation_episodes

        observations = torch.zeros((int(np.ceil((num_episodes + 8) * self.max_episode_steps / self.num_envs)),
                                    self.num_envs) + self.envs.single_observation_space.shape).to(device)
        cumulative_rewards = torch.zeros((int(np.ceil((num_episodes + 8) * self.max_episode_steps / self.num_envs)),
                                          self.num_envs)).to(device)
        control_actions = torch.zeros((int(np.ceil((num_episodes + 8) * self.max_episode_steps / self.num_envs)),
                                       self.num_envs) + self.envs.single_action_space.shape).to(device)

        next_obs, _ = self.envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)

        step = 0
        episode = 0
        episode_step = 0
        while episode < num_episodes:

            with torch.no_grad():
                action = self.agent.get_action_mean(next_obs)

            # observations[episode][episode_step] = next_obs
            # control_actions[episode][episode_step] = action

            observations[step] = next_obs
            control_actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log observations.
            next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
            cumulative_rewards[step] = torch.Tensor(reward).squeeze().to(device)
            episode_step += 1
            step += 1
            done = np.logical_or(terminated, truncated)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if "final_info" in info.keys():
                episode += self.num_envs
                episode_step = 0
                episodic_return = np.array([x['episode']['r'] for x in info['final_info']])
                # cumulative_rewards[episode - 1] = episodic_return
                episode_labels = np.array(
                    [f"episode={episode - 7 + i}, reward = " for i in range(len(episodic_return))])
                print_strings = [label + str(value) for label, value in zip(episode_labels, episodic_return)]
                print("\n".join(print_strings))
                # print(f"episode={episode}, episodic_return={episodic_return}")
                if self.track:
                    pass
                    # self.writer.add_scalar("charts/episodic_return", episodic_return, global_step)

        if self.track:
            observations = self.reshape_tensor(observations)[:num_episodes, :, :]
            control_actions = self.reshape_tensor(control_actions)[:num_episodes, :, :]
            cumulative_rewards = self.reshape_tensor(
                cumulative_rewards.unsqueeze(dim=-1)).squeeze().sum(dim=1)[:num_episodes]

            save_path = f"runs/{self.run_name}/{self.run_name}_validation.npz"
            np.savez(save_path,
                     cumulative_rewards=cumulative_rewards.cpu().numpy(),
                     observations=observations.cpu().numpy(),
                     control_actions=control_actions.cpu().numpy(),
                     )


    def close(self):
        self.envs.close()
        if self.track:
            self.writer.close()
