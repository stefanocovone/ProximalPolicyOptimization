import os
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

FIGURES_FOLDER = './Figures/'


def cart_to_polar(observations):
    # Rescale all the quantities to +-region_length
    observations = np.array(observations) * 60

    # Check if the shape is (100, 1200, 4)
    if observations.shape[-1] != 4:
        raise ValueError("The last dimension of the input observations must be of size 4")

    # Extract the herder-target vector components and the target position components
    delta_x = observations[..., 0]
    delta_y = observations[..., 1]
    target_x = observations[..., 2]
    target_y = observations[..., 3]

    # Reconstruct the herder position
    herder_x = target_x - delta_x
    herder_y = target_y - delta_y

    # Compute polar coordinates for the herder position
    radius_herder = np.sqrt(herder_x ** 2 + herder_y ** 2)
    angle_herder = np.arctan2(herder_y, herder_x)

    # Compute polar coordinates for the target position
    radius_target = np.sqrt(target_x ** 2 + target_y ** 2)
    angle_target = np.arctan2(target_y, target_x)

    # Combine the polar coordinates into the desired output format
    polar_coords = np.stack((radius_herder, angle_herder, radius_target, angle_target), axis=-1)

    return polar_coords


def compute_settling_time_legacy(obs):
    eta = 5.5

    # Compute the norm of each state in the obs array
    state_norms = obs[..., 2]

    # Create a boolean mask to identify states inside the goal region
    inside_goal_mask = state_norms < eta

    # Reverse the boolean mask along the steps axis
    reversed_mask = np.flip(inside_goal_mask, axis=1).astype(int)
    # Find the index of the first occurrence of True in the reversed mask
    settling_indices = 1200 - np.argmin(reversed_mask, axis=1)

    # Convert indices to settling times
    settling_times = settling_indices

    return settling_times


def compute_settling_time(positions, threshold=5):
    """
    Compute the settling time for each episode.
    The settling time is defined as the minimum timestep t_s in which target_radius[t_s] is inside
    the goal region and the average radius between t_s and t_f is still inside the goal region.

    Parameters:
    positions (numpy.ndarray): A 3D array of shape (num_episodes, episode_steps, states) representing positions.
    threshold (float): The threshold below which the average position must be.

    Returns:
    numpy.ndarray: A 1D array of settling times for each episode.
    """

    positions = positions[..., 2]
    num_episodes, episode_steps = positions.shape

    # Calculate the cumulative sum from the end to the start
    cumulative_sum = np.cumsum(positions[:, ::-1], axis=1)[:, ::-1]

    # Calculate the number of elements from each step to the end
    step_counts = np.arange(episode_steps, 0, -1)

    # Calculate the running average from each step to the end
    running_avg = cumulative_sum / step_counts

    # Check both conditions: position at t_s and running average from t_s to the end
    condition_met = (positions < threshold) & (running_avg < threshold)

    # Find the first index where both conditions are met
    settling_times = np.argmax(condition_met, axis=1)

    # Handle episodes where the condition is never met
    not_found_mask = np.all(~condition_met, axis=1)
    settling_times[not_found_mask] = episode_steps

    return settling_times


def compute_successful_episode(settling_times, threshold):
    # Convert settling times to binary values (0 or 1) based on the threshold
    successful_episode = np.array((settling_times < threshold)).astype(int)
    return successful_episode


def terminal_episode(vector, n=10):
    # Find the indices of the first occurrence of n consecutive True values
    succession_indices = np.where(np.convolve(vector, np.ones(n), mode='valid') == n)[0]

    # Check if any succession of 30 consecutive True values was found
    if len(succession_indices) > 0:
        return succession_indices[0]
    else:
        return 100000


Data = namedtuple('Data', ['mean', 'std'])


class AgentResults:
    def __init__(self, agent_id, env_id, sessions=1):
        self.file_prefix = f"{env_id}__{agent_id}"
        self.agent_id = agent_id
        self.env_id = env_id
        self.sessions = sessions
        self.rewards = None
        self.terminal_episodes = None
        self.settling_times = None
        self.observations_t = None
        self.observations_v = None
        self.actions = None
        self.successful_episodes = None

    def load_training(self):
        self.rewards = []
        self.terminal_episodes = []
        self.observations_t = []
        self.settling_times = []
        for i in range(self.sessions):
            filename = f"{self.file_prefix}_{i + 1}_training.npz"
            file_path = os.path.join(f"./runs/{self.file_prefix}_{i + 1}", filename)
            training_results = np.load(file_path)

            self.rewards.append(training_results["cumulative_rewards"])
            # self.observations_t.append(cart_to_polar(training_results["observations"].squeeze()))
            settling_times = training_results["settling_times"]

            # compute terminal episodes
            # settling_times_t = compute_settling_time(self.observations_t[i])
            successful_episode_t = compute_successful_episode(settling_times, 1000)
            terminal_episode_t = terminal_episode(successful_episode_t)
            self.terminal_episodes.append(terminal_episode_t)

    def load_validation(self):
        self.settling_times = []
        self.actions = []
        self.observations_v = []
        self.successful_episodes = []

        for i in range(self.sessions):
            filename = f"{self.file_prefix}_{i + 1}_validation.npz"
            file_path = os.path.join(f"./runs/{self.file_prefix}_{i + 1}", filename)
            validation_results = np.load(file_path)

            observations_v = cart_to_polar(validation_results["observations"].squeeze())
            self.observations_v.append(observations_v)
            self.actions.append(validation_results["control_actions"].squeeze())

            settling_times_v = compute_settling_time(observations_v)
            self.settling_times.append(settling_times_v)

            self.successful_episodes.append(compute_successful_episode(settling_times_v, 1000))

        self.actions = np.clip(self.actions, -8, 8)

    def average_reward(self):
        avg_rewards = []
        for rewards in self.rewards:
            avg_rewards.append(np.mean(rewards))
        mean = np.mean(avg_rewards)
        std = np.std(avg_rewards)
        return Data(mean, std)

    def average_term_reward(self):
        avg_term_rewards = []
        for rewards, terminalEpisode in zip(self.rewards, self.terminal_episodes):
            avg_term_rewards.append(np.mean(rewards[terminalEpisode:]))
        mean = np.mean(avg_term_rewards)
        std = np.std(avg_term_rewards)
        return Data(mean, std)

    def terminal_episode(self):
        mean = np.mean(self.terminal_episodes)
        std = np.std(self.terminal_episodes)
        return Data(mean, std)

    def settling_time(self):
        settling_times = []
        for settling_time in self.settling_times:
            mean = np.mean(settling_time)
            std = np.std(settling_time)
            print(f"mean: {mean} std: {std}")
            settling_times.extend(settling_time)
        mean = np.mean(settling_times)
        std = np.std(settling_times)
        return Data(mean, std)

    def success_rate(self):
        success_rates = []
        for success in self.successful_episodes:
            success_rate = np.mean(success)*100
            success_rates.append(success_rate)
        mean = np.mean(success_rates)
        std = np.std(success_rates)
        return Data(mean, std)

    def control_effort(self):
        actions = self.actions
        control_norm = []
        for control_actions in actions:
            actions_norm = np.linalg.norm(control_actions, axis=-1)
            control_norm.extend(np.linalg.norm(actions_norm, axis=-1))
        mean = np.mean(control_norm)
        std = np.std(control_norm)
        return Data(mean, std)


def moving_average(data, window_size):
    moving_avg = np.zeros_like(data)
    for i, sequence in enumerate(data):
        for j in range(len(sequence)):
            start_index = max(0, j - window_size + 1)
            end_index = j + 1
            moving_avg[i][j] = np.mean(sequence[start_index:end_index])
    return moving_avg


def plot_rewards(*agents, labels=None, filename=None, moving_avg_size=100, training_length=10000):
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, agent in enumerate(agents):
        label = labels[idx] if labels else agent.file_prefix
        # smoothing signals
        smoothed_rewards = moving_average(agent.rewards, moving_avg_size)
        # compute mean and standard deviation
        mean_rewards = np.mean(smoothed_rewards, axis=0)
        std_rewards = np.std(smoothed_rewards, axis=0)

        ax.plot(mean_rewards, label=label)
        ax.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards,
                        mean_rewards + std_rewards, alpha=0.2)

    ax.set_xlabel('Episodes')
    ax.set_ylabel(r'$J_e^\pi$')
    ax.set_title('Cumulative reward')
    ax.grid(True)

    ax.set_xlim(0, training_length)

    if filename:
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='pdf')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'rewards.pdf'), format='pdf')

    plt.show()


def plot_training_metrics(*agents, labels=None, filename=None):
    # Define the observations
    metrics = ['$J_{avg}^\pi$', '$J_{avg,t}^\pi$', 'Episode']
    stats = [[agent.average_reward(), agent.average_term_reward(), agent.terminal_episode()] for agent in agents]

    means = [[stat.mean for stat in stat_list] for stat_list in stats]
    stds = [[stat.std for stat in stat_list] for stat_list in stats]

    # Print and store observations in results.txt
    with open("../results.txt", "w") as file:
        for j, agent in enumerate(agents):
            agent_prefix = agent.file_prefix
            file.write(f"Agent: {agent_prefix}\n")
            for i, metric in enumerate(metrics):
                file.write(f"Mean {metric}: {means[j][i]}, Std {metric}: {stds[j][i]}\n")
            file.write("\n")

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    for i, metric in enumerate(metrics):
        for j, agent in enumerate(agents):
            agent_label = labels[j] if labels else agent.file_prefix
            axs[i].errorbar([agent_label], [means[j][i]], yerr=[stds[j][i]], fmt='o', capsize=5, label=metric)
            print(f"Mean {metric}: {means[j][i]}, Std {metric}: {stds[j][i]}")  # Print mean and std
        axs[i].grid(True)
        axs[i].set_ylabel(metric)

    axs[0].set_title('Average Cumulative Reward')
    axs[1].set_title('Terminal Cumulative Reward')
    axs[2].set_title('Terminal Episode')
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='pdf')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'training_metrics.pdf'), format='pdf')
    plt.show()


def plot_validation_metrics(*agents, labels=None, filename=None):
    # Define the observations
    metrics = ['Timestep', '%', 'velocity']
    stats = [[agent.settling_time(),
              agent.success_rate(),
              agent.control_effort()] for agent in agents]

    means = [[stat.mean for stat in stat_list] for stat_list in stats]
    stds = [[stat.std for stat in stat_list] for stat_list in stats]

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    for i, metric in enumerate(metrics):
        for j, agent in enumerate(agents):
            agent_label = labels[j] if labels else "agent_name"
            axs[i].errorbar([agent_label], [means[j][i]], yerr=[stds[j][i]], fmt='o', capsize=5, label=metric)
            print(f"Mean {metric}: {means[j][i]}, Std {metric}: {stds[j][i]}")  # Print mean and std
        axs[i].grid(True)
        axs[i].set_ylabel(metric)

    axs[0].set_title('Settling Time')
    axs[1].set_title('Success Rate')
    axs[2].set_title('Control Effort')

    axs[1].set_ylim(0, 110)

    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='pdf')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'validation_metrics.pdf'), format='pdf')
    plt.show()


def plot_agent_data(agent, filename=None):
    # Extract observations for plotting
    data = agent.observations_v[0]
    ep = 1
    herder_radius = data[ep, 1:, 0]
    herder_angle = data[ep, 1:, 1]
    target_radius = data[ep, 1:, 2]
    target_angle = data[ep, 1:, 3]

    # Convert polar coordinates to Cartesian coordinates
    herder_x = herder_radius * np.cos(herder_angle)
    herder_y = herder_radius * np.sin(herder_angle)
    target_x = target_radius * np.cos(target_angle)
    target_y = target_radius * np.sin(target_angle)

    # Calculate relative distance in Cartesian coordinates
    relative_distance = np.sqrt((herder_x - target_x) ** 2 + (herder_y - target_y) ** 2)

    # Calculate phase difference between herder and target
    phase_difference = herder_angle - target_angle

    # Adjust phase difference to be in the range [-pi, pi]
    phase_difference = (phase_difference + np.pi) % (2 * np.pi) - np.pi

    # Create figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot herder and target radii on the first subplot
    axs[0].plot(herder_radius, label='Herder Radius')
    axs[0].plot(target_radius, label='Target Radius')
    axs[0].axhline(y=5, color='r', linestyle='--', label='Goal Radius')
    axs[0].set_title('Herder and Target Radii')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Radius')
    axs[0].set_xlim(0, 1200)
    axs[0].set_ylim(0, 30)
    axs[0].legend()
    axs[0].grid(True)

    # Plot relative distance on the second subplot
    axs[1].plot(relative_distance, label='Relative Distance')
    axs[1].axhline(y=2.5, color='r', linestyle='--', label='Repulsion radius')
    axs[1].set_title('Relative Distance Between Herder and Target')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Distance')
    axs[1].set_xlim(0, 1200)
    # axs[1].set_ylim(0, 3)
    axs[1].legend()
    axs[1].grid(True)

    # Plot phase difference on the third subplot
    axs[2].plot(phase_difference, label='Phase Difference')
    axs[2].set_title('Phase Difference Between Herder and Target')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Phase Difference (radians)')
    axs[2].set_xlim(0, 1200)
    axs[2].legend()
    axs[2].grid(True)

    # plt.show()

    # Adjust layout for better visualization
    plt.tight_layout()

    # Save the figure
    if filename:
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='pdf')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'agent_data.pdf'), format='pdf')

    # Show the figure
    plt.show()
