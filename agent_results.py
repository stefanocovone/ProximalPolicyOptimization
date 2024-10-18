import os
import numpy as np
from collections import namedtuple


Data = namedtuple('Data', ['mean', 'std'])


def compute_successful_episode(settling_times, threshold):
    """
    Compute binary success indicators based on settling times and a threshold.

    Args:
        settling_times (np.ndarray): Array of settling times.
        threshold (float): Threshold for considering an episode successful.

    Returns:
        np.ndarray: Binary array indicating success (1) or failure (0) per episode.
    """
    return (settling_times < threshold).astype(int)


def terminal_episode(successful_episodes, n=20):
    """
    Find the index of the first occurrence of 'n' consecutive successful episodes.

    Args:
        successful_episodes (np.ndarray): Binary array indicating success per episode.
        n (int): Number of consecutive successes required.

    Returns:
        int: Index of the terminal episode, or the length of the array if not found.
    """
    conv_result = np.convolve(successful_episodes, np.ones(n, dtype=int), mode='valid')
    succession_indices = np.where(conv_result == n)[0]
    if len(succession_indices) > 0:
        return succession_indices[0]
    else:
        return len(successful_episodes)


class AgentResults:
    """
    Class to store and process results for an agent.

    Attributes:
        agent_id (str): Identifier for the agent.
        env_id (str): Identifier for the environment.
        sessions (int): Number of sessions.
        rewards (list): List of rewards per session.
        terminal_episodes (list): List of terminal episode indices per session.
        observations_t (list): List of training observations per session.
        settling_times (list): List of settling times per session.
        observations_v (list): List of validation observations per session.
        actions (list): List of actions per session.
        successful_episodes (list): List of success indicators per session.
        cooperative_metric (list): List of cooperative metrics per session.
    """

    def __init__(self, agent_id, env_id, sessions=1):
        self.file_prefix = f"{env_id}__{agent_id}"
        self.agent_id = agent_id
        self.env_id = env_id
        self.sessions = sessions
        self.rewards = []
        self.terminal_episodes = []
        self.settling_times = []
        self.observations_t = []
        self.observations_v = []
        self.actions = []
        self.successful_episodes = []
        self.cooperative_metric = []

    def load_training(self):
        """
        Load training data for the agent.
        """
        for i in range(self.sessions):
            filename = f"{self.file_prefix}_{i + 1}_training.npz"
            file_path = os.path.join(f"./runs/{self.file_prefix}_{i + 1}", filename)

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}. Skipping this session.")
                continue

            training_results = np.load(file_path)
            self.rewards.append(training_results["cumulative_rewards"])

            settling_times = training_results.get("settling_times", None)
            if settling_times is None:
                print(f"Settling times not found in {file_path}. Skipping terminal episode computation.")
                continue

            # Compute successful episodes and terminal episodes
            successful_episode_t = compute_successful_episode(settling_times, threshold=1900)
            terminal_episode_t = terminal_episode(successful_episode_t)
            self.terminal_episodes.append(terminal_episode_t)

    def load_validation(self):
        """
        Load validation data for the agent.
        """
        for i in range(self.sessions):
            filename = f"{self.file_prefix}_{i + 1}_validation.npz"
            file_path = os.path.join(f"./runs/{self.file_prefix}_{i + 1}", filename)

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}. Skipping this session.")
                continue

            validation_results = np.load(file_path)

            observations = validation_results["observations"]
            actions = validation_results["control_actions"]
            episode_lengths = validation_results.get("episode_lengths", None)

            if episode_lengths is None:
                print(f"Episode lengths not found in {file_path}. Unable to process variable-length episodes.")
                continue

            # Process observations and actions for variable-length episodes
            observations_list = [observations[i, :length] for i, length in enumerate(episode_lengths)]
            actions_list = [actions[i, :length] for i, length in enumerate(episode_lengths)]
            self.observations_v.extend(observations_list)
            self.actions.extend(actions_list)

            # Compute settling times and cooperative metrics
            settling_times_v = self.compute_settling_times(observations_list)
            self.settling_times.extend(settling_times_v)

            self.successful_episodes.extend(compute_successful_episode(settling_times_v, threshold=1950))

            cooperative_metric = self.compute_cooperative_metric(actions_list)
            self.cooperative_metric.extend(cooperative_metric)

    def compute_settling_times(self, observations_list):
        """
        Compute settling times for a list of observations.

        Args:
            observations_list (list): List of observations per episode.

        Returns:
            list: Settling times per episode.
        """
        eta = 7
        settling_times = []
        for obs in observations_list:
            # Compute norms of target positions
            target_positions = obs[:, -2:]  # Assuming target positions are the last two columns
            target_norms = np.linalg.norm(target_positions, axis=-1)
            # Identify when targets are inside the goal region
            inside_goal_mask = target_norms < eta
            if np.any(inside_goal_mask):
                settling_time = np.argmax(inside_goal_mask)
            else:
                settling_time = len(obs)
            settling_times.append(settling_time)
        return settling_times

    def compute_cooperative_metric(self, actions_list):
        """
        Compute the cooperative metric for a list of actions.

        Args:
            actions_list (list): List of actions per episode.

        Returns:
            list: Cooperative metric per episode.
        """
        cooperative_metrics = []
        for actions in actions_list:
            unique_actions = [np.unique(step_actions) for step_actions in actions]
            cm = [len(u_actions) / actions.shape[-1] for u_actions in unique_actions]
            cooperative_metrics.append(np.mean(cm))
        return cooperative_metrics

    def moving_average(self, data, window_size):
        """
        Compute the moving average of a list of arrays.

        Args:
            data (list): List of arrays.
            window_size (int): Window size for moving average.

        Returns:
            list: List of arrays with moving average applied.
        """
        smoothed_data = []
        for sequence in data:
            if len(sequence) < window_size:
                smoothed_data.append(np.full_like(sequence, np.mean(sequence)))
            else:
                cumsum = np.cumsum(np.insert(sequence, 0, 0))
                moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
                # Pad the beginning of the array
                moving_avg = np.concatenate((np.full(window_size - 1, moving_avg[0]), moving_avg))
                smoothed_data.append(moving_avg)
        return smoothed_data

    # Methods for computing averages and statistics
    def average_reward(self):
        avg_rewards = [np.mean(rewards) for rewards in self.rewards]
        mean = np.mean(avg_rewards)
        std = np.std(avg_rewards)
        return Data(mean, std)

    def average_term_reward(self):
        avg_term_rewards = []
        for rewards, terminal_episode in zip(self.rewards, self.terminal_episodes):
            avg_term_rewards.append(np.mean(rewards[terminal_episode:]))
        mean = np.mean(avg_term_rewards)
        std = np.std(avg_term_rewards)
        return Data(mean, std)

    def terminal_episode_data(self):
        mean = np.mean(self.terminal_episodes)
        std = np.std(self.terminal_episodes)
        return Data(mean, std)

    def settling_time_data(self):
        settling_times = self.settling_times
        mean = np.mean(settling_times)
        std = np.std(settling_times)
        return Data(mean, std)

    def success_rate_data(self):
        success_rates = [np.mean(success) * 100 for success in self.successful_episodes]
        mean = np.mean(success_rates)
        std = np.std(success_rates)
        return Data(mean, std)

    def cooperative_metric_data(self):
        cm_values = self.cooperative_metric
        mean = np.mean(cm_values)
        std = np.std(cm_values)
        return Data(mean, std)
