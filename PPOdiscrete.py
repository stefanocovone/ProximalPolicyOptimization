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
    """
    Determines whether to record an episode based on its ID.

    Args:
        episode_id (int): The ID of the episode.

    Returns:
        bool: True if the episode should be recorded, False otherwise.
    """
    episodes = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                10050, 10100, 10120, 10180]
    record = ((episode_id + 1000) % 1000 == 0)
    val_ep = [10010, 10020, 10040, 10060, 10080, 10100, 10120, 10140, 10160, 10180]
    return record or (episode_id in val_ep)


def make_env(gym_id, seed, idx, capture_video, run_name,
             max_episode_steps, env_params, random_target=False, render=False):
    """
    Factory function to create a new environment instance.

    Args:
        gym_id (str): The ID of the gym environment to create.
        seed (int): Seed for reproducibility.
        idx (int): Index of the environment instance.
        capture_video (bool): Whether to record videos of the environment.
        run_name (str): Name of the current run for logging purposes.
        max_episode_steps (int): Maximum steps per episode.
        env_params (dict): Parameters for the custom environment.
        random_target (bool): Whether to randomize targets.
        render (bool): Whether to render the environment.

    Returns:
        function: A function that creates and returns the environment instance.
    """

    def thunk():
        # Set render mode based on the 'render' flag
        render_mode = 'human' if render else 'rgb_array'

        if gym_id == "Shepherding-v0":
            # Create the custom shepherding environment
            env = gym.make(gym_id, render_mode=render_mode,
                           parameters=env_params, rand_target=random_target)
            env._max_episode_steps = max_episode_steps

            # Add termination condition if specified
            if env_params['termination']:
                env = TerminateWhenSuccessful(env, num_steps=200)

            # Capture video if required
            if capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}",
                                               episode_trigger=record_trigger)

            # Flatten the observation space for compatibility
            env = FlattenObservation(env)

            # Apply custom policy wrapper
            env = LowLevelPPOPolicy(env, env_params['target_selection_rate'])
        else:
            # Create standard gym environment
            env = gym.make(gym_id, render_mode='rgb_array')
            env._max_episode_steps = max_episode_steps

        # Record episode statistics
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Set seeds for reproducibility
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
        np.random.seed(seed)
        random.seed(seed)
        return env

    return thunk


def _compute_policy_loss(mb_advantages, ratio, clip_coef):
    """
    Computes the clipped policy loss for PPO.

    Args:
        mb_advantages (torch.Tensor): Advantages for the minibatch.
        ratio (torch.Tensor): Probability ratios of new and old policies.
        clip_coef (float): Clipping coefficient.

    Returns:
        torch.Tensor: The policy loss.
    """
    # Compute the unclipped and clipped policy losses
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    # Return the maximum of the two losses
    return torch.max(pg_loss1, pg_loss2).mean()


def _compute_value_loss(new_value, clip_vloss, clip_coef, b_returns, b_values, mb_inds):
    """
    Computes the value function loss, optionally with clipping.

    Args:
        new_value (torch.Tensor): Predicted values from the model.
        clip_vloss (bool): Whether to use value function clipping.
        clip_coef (float): Clipping coefficient.
        b_returns (torch.Tensor): Target returns.
        b_values (torch.Tensor): Predicted values from the old policy.
        mb_inds (np.ndarray): Indices of the minibatch.

    Returns:
        torch.Tensor: The value function loss.
    """
    new_value = new_value.view(-1)
    if clip_vloss:
        # Compute unclipped and clipped value losses
        v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
        v_clipped = b_values[mb_inds] + torch.clamp(
            new_value - b_values[mb_inds], -clip_coef, clip_coef)
        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        return 0.5 * v_loss_max.mean()
    else:
        # Compute standard value loss
        return 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()


def _remove_dir(directory):
    """
    Removes a directory and its contents.

    Args:
        directory (str): Path to the directory to remove.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


def reshape_tensor(tensor, episode_length):
    """
    Reshapes a tensor for episode-based analysis.

    Args:
        tensor (torch.Tensor): The tensor to reshape.
        episode_length (int): The length of each episode.

    Returns:
        torch.Tensor: The reshaped tensor.
    """
    slices = tensor.split(episode_length, dim=0)
    reshaped_slices = [s.permute(1, 0, 2) for s in slices]
    result = torch.cat(reshaped_slices[:-1], dim=0)
    return result


class PPO:
    """
    Proximal Policy Optimization (PPO) agent for reinforcement learning.

    Attributes:
        exp_name (str): Experiment name for logging.
        gym_id (str): Gym environment ID.
        gym_params (dict): Parameters for the custom environment.
        max_episode_steps (int): Maximum steps per episode.
        num_episodes (int): Total number of training episodes.
        num_validation_episodes (int): Number of validation episodes.
        learning_rate (float): Learning rate for the optimizer.
        seed (int): Random seed for reproducibility.
        ...
    """

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
                 target_kl=0.01):
        """
        Initializes the PPO agent with the given parameters.
        """
        # Store parameters
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

        # Calculate total timesteps and batch sizes
        self.total_timesteps = self.num_episodes * self.max_episode_steps
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        # Set up run names and directories
        self.run_name = f"{self.gym_id}__{self.exp_name}"
        self.run_name_val = f"{self.gym_id}__{self.exp_name_val}"
        self.save_dir = f"runs/{self.run_name}"

        # Initialize summary writer for logging
        if self.track:
            _remove_dir(self.save_dir)
            self.writer = SummaryWriter(f"runs/{self.run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % (
                    "\n".join([f"|{key}|{value}|" for key, value in vars(self).items()])),
            )

        # Set seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        # Set device for computation
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")

        # Environment setup
        self.envs = gym.vector.AsyncVectorEnv(
            [make_env(
                gym_id=self.gym_id,
                seed=self.seed + i,
                idx=i,
                capture_video=self.capture_video,
                run_name=self.run_name,
                max_episode_steps=self.max_episode_steps,
                env_params=self.gym_params,
                random_target=self.gym_params.get('random_targets', False),
                render=self.render
            ) for i in range(self.num_envs)]
        )
        # Ensure the action space is discrete
        assert isinstance(self.envs.single_action_space, gym.spaces.MultiDiscrete), \
            "Only discrete action space is supported"

        # Initialize the agent and optimizer
        self.agent = ActorCritic(self.envs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

    def _compute_advantage(self, next_obs, rewards, next_done, dones, values):
        """
        Computes the advantages and returns for policy optimization.

        Args:
            next_obs (torch.Tensor): The observation at the next time step.
            rewards (torch.Tensor): Collected rewards.
            next_done (torch.Tensor): Done flags for the next time step.
            dones (torch.Tensor): Done flags for each time step.
            values (torch.Tensor): Estimated values from the value function.

        Returns:
            tuple: Advantages and returns tensors.
        """
        with torch.no_grad():
            # Get the value for the next observation
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            # Concatenate the values to handle the terminal state
            values = torch.cat((values, next_value), dim=0)

            # Initialize advantage tensor
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(rewards.size(0))):
                if t == rewards.size(0) - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values[:-1]
        return advantages, returns

    def train(self):
        """
        Trains the PPO agent using the collected experiences.
        """
        self.agent.train()
        print(f"TRAINING {self.run_name}")
        device = self.device

        num_episodes = self.num_episodes
        num_envs = self.num_envs

        # Initialize variables
        global_step = 0
        episode = 0
        start_time = time.time()

        # Reset the environments and get initial observations
        next_obs, _ = self.envs.reset()
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)
        num_updates = int(np.ceil(self.total_timesteps / self.batch_size)) + 2

        # Lists to store cumulative rewards and settling times for logging
        cumulative_rewards = []
        settling_times = []

        # Storage tensors for the collected experiences
        obs = torch.zeros((self.num_steps, num_envs) + self.envs.single_observation_space.shape, device=device)
        actions = torch.zeros((self.num_steps, num_envs) + self.envs.single_action_space.shape, device=device)
        log_probs = torch.zeros((self.num_steps, num_envs), device=device)
        rewards = torch.zeros((self.num_steps, num_envs), device=device)
        dones = torch.zeros((self.num_steps, num_envs), device=device)
        values = torch.zeros((self.num_steps, num_envs), device=device)

        update = 0

        while episode < num_episodes:
            update += 1

            # Adjust the learning rate if annealing is enabled
            if self.anneal_lr:
                frac = 1.0 - (episode - 1.0) / num_episodes
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(self.num_steps):
                global_step += num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # Get action, log probability, and value estimate from the agent
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                log_probs[step] = logprob

                # Interact with the environment
                next_obs_np, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
                next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

                # Check if any episodes have finished
                if "final_info" in info:
                    for env_idx, final_info in enumerate(info["final_info"]):
                        if final_info is not None:
                            # Collect episode metrics
                            episodic_return = final_info['episode']['r']
                            settling_time = final_info.get('settling_time', None)
                            cumulative_rewards.append(episodic_return)
                            settling_times.append(settling_time)
                            episode += 1
                            print(f"Episode {episode}, Env {env_idx}, Reward: {episodic_return}")

                            # Log metrics if tracking is enabled
                            if self.track:
                                self.writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                                if settling_time is not None:
                                    self.writer.add_scalar("charts/settling_time", settling_time, global_step)

                            if episode >= num_episodes:
                                break  # Exit if we've collected enough episodes

                if episode >= num_episodes:
                    break  # Exit the time step loop if we've collected enough episodes

            # Adjust tensors if episode limit was reached early
            actual_steps = step + 1
            obs = obs[:actual_steps]
            actions = actions[:actual_steps]
            log_probs = log_probs[:actual_steps]
            rewards = rewards[:actual_steps]
            dones = dones[:actual_steps]
            values = values[:actual_steps]

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
        """
        Performs the optimization step for the PPO agent.

        Args:
            obs (torch.Tensor): Observations.
            log_probs (torch.Tensor): Log probabilities of actions.
            actions (torch.Tensor): Actions taken.
            advantages (torch.Tensor): Computed advantages.
            returns (torch.Tensor): Computed returns.
            values (torch.Tensor): Value estimates.
            global_step (int): Global time step counter.
            start_time (float): Start time of training.
        """
        device = self.device  # Ensuring all operations are on the correct device

        # Flatten the batch dimensions
        b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape).to(device)
        b_logprobs = log_probs.reshape(-1).to(device)
        b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape).to(device)
        b_advantages = advantages.reshape(-1).to(device)
        b_returns = returns.reshape(-1).to(device)
        b_values = values.reshape(-1).to(device)

        # Ensure all tensors have the same length
        assert b_obs.shape[0] == b_actions.shape[0] == b_logprobs.shape[0] == \
               b_advantages.shape[0] == b_returns.shape[0] == b_values.shape[0], \
               "Mismatch in tensor lengths!"

        # Prepare for optimization
        batch_size = b_obs.shape[0]
        b_inds = np.arange(batch_size)
        clip_fracs = []

        # Optimization loop
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                # Get new action log probabilities and value estimates
                _, new_log_prob, entropy, new_value = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds])
                new_log_prob = new_log_prob.view(-1)
                entropy = entropy.view(-1)
                new_value = new_value.view(-1)

                # Compute ratio of new and old action probabilities
                log_ratio = new_log_prob - b_logprobs[mb_inds]
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    # Compute approximate KL divergence
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                # Normalize advantages if required
                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Compute policy loss
                pg_loss = _compute_policy_loss(mb_advantages, ratio, self.clip_coef)

                # Compute value loss
                v_loss = _compute_value_loss(
                    new_value, self.clip_vloss, self.clip_coef, b_returns, b_values, mb_inds)

                # Compute entropy loss for regularization
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                # Backpropagation and optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Early stopping based on KL divergence
            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        # Calculate explained variance for diagnostics
        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log losses and other metrics if tracking is enabled
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
        """
        Validates the trained agent by running it in the environment and collecting data.

        Saves the collected observations, actions, and rewards for analysis.
        """
        self.agent.eval()
        print(f"VALIDATION {self.run_name}")
        device = self.device

        # Load the trained agent
        self.agent.load_state_dict(
            torch.load(f'runs/{self.run_name}/{self.run_name}_agent.pt', map_location=device))

        # Number of episodes to run for validation
        num_episodes = self.num_validation_episodes
        num_envs = self.num_envs

        # Initialize storage for episodes
        episodes = []  # Stores episodes in the order they start
        current_episodes = [None for _ in range(num_envs)]  # Current episode data per environment

        episodes_finished = 0

        # Reset the environments and get initial observations
        next_obs, _ = self.envs.reset()
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

        while episodes_finished < num_episodes:
            with torch.no_grad():
                # Get action from the agent
                action = self.agent.get_action(next_obs)

            # Execute the action in the environment
            next_obs_np, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

            done = np.logical_or(terminated, truncated)

            # For each environment
            for env_idx in range(num_envs):
                # Start a new episode if needed
                if current_episodes[env_idx] is None:
                    # Initialize episode data
                    current_episode = {
                        'observations': [],
                        'actions': [],
                        'rewards': []
                    }
                    episodes.append(current_episode)
                    current_episodes[env_idx] = current_episode

                # Collect data for the current episode
                current_episodes[env_idx]['observations'].append(next_obs_np[env_idx])
                current_episodes[env_idx]['actions'].append(action[env_idx].cpu().numpy())
                current_episodes[env_idx]['rewards'].append(reward[env_idx])

                if done[env_idx]:
                    # Episode finished
                    episodic_return = sum(current_episodes[env_idx]['rewards'])
                    episodes_finished += 1
                    print(f"Episode {episodes_finished} finished in env {env_idx}, reward = {episodic_return}")

                    # Reset current episode for this environment
                    current_episodes[env_idx] = None

                    if episodes_finished >= num_episodes:
                        break  # Exit if we've collected enough episodes

            if episodes_finished >= num_episodes:
                break  # Exit outer loop as well

        # Process and save the collected episodes
        episode_lengths = np.array([len(episode['observations']) for episode in episodes])
        max_length = max(episode_lengths)
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

        # Save the data if tracking is enabled
        if self.track:
            save_path = f"runs/{self.run_name}/{self.run_name}_validation.npz"
            np.savez(save_path,
                     cumulative_rewards=cumulative_rewards,
                     observations=obs_padded,
                     control_actions=actions_padded,
                     episode_lengths=episode_lengths)

    def close(self):
        """
        Closes the environments and any open resources.
        """
        self.envs.close()
        if self.track:
            self.writer.close()
