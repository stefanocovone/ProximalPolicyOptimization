import os
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

FIGURES_FOLDER = './Figures/'


def cart_to_polar(obs):
    cos = obs[:, :, 0]
    sin = obs[:, :, 1]
    angle = np.arctan2(sin, cos)
    vel = obs[:, :, 2]
    polar_obs = np.stack((angle, vel), axis=-1)
    return polar_obs


def compute_settling_time(obs):
    eta = 0.05 * np.linalg.norm([np.pi, 8])

    # Compute the norm of each state in the obs array
    state_norms = np.linalg.norm(obs, axis=-1)

    # Create a boolean mask to identify states inside the goal region
    inside_goal_mask = state_norms < eta

    # Reverse the boolean mask along the steps axis
    reversed_mask = np.flip(inside_goal_mask, axis=1).astype(int)

    # Find the index of the first occurrence of True in the reversed mask
    settling_indices = 400 - np.argmin(reversed_mask, axis=1)

    # Convert indices to settling times
    settling_times = settling_indices

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
        return 10000


def compute_ss_error(obs, settling_times):
    # Get the number of steps per episode
    num_steps_per_episode = obs.shape[1]

    # Create an index array for steps
    steps_idx = np.arange(num_steps_per_episode)

    # Create a mask for steps after settling time
    settling_mask = steps_idx >= settling_times[:, np.newaxis]

    # Compute the cumulative norms for each episode
    cumulative_norms = np.linalg.norm(obs[:, :, :], axis=-1)

    # Apply the mask to cumulative norms to get norms after settling time
    norms_after_settling = cumulative_norms * settling_mask

    # Sum the norms after settling time for each episode
    ss_error = np.sum(norms_after_settling, axis=1)

    # Set steady-state error to NaN where settling time is 0
    ss_error[settling_times == 0] = np.nan

    # Adjust ss_error for the number of steps after settling time
    x_max = np.linalg.norm([np.pi, 8])
    ss_error /= (num_steps_per_episode - settling_times + 1) * x_max

    return ss_error


Data = namedtuple('Data', ['mean', 'std'])


class AgentResults:
    def __init__(self, agent_id, env_id, sessions=1):
        self.observations_v_det = None
        self.actions_std_det = None
        self.actions_mean_det = None
        self.actions_det = None
        self.ss_errors_det = None
        self.settling_times_det = None
        self.file_prefix = f"{env_id}__{agent_id}"
        self.agent_id = agent_id
        self.env_id = env_id
        self.sessions = sessions
        self.rewards = None
        self.terminal_episodes = None
        self.settling_times = None
        self.ss_errors = None
        self.observations_t = None
        self.observations_v = None
        self.actions = None
        self.actions_mean = None
        self.actions_std = None

    def load_training(self):
        self.rewards = []
        self.terminal_episodes = []
        for i in range(self.sessions):
            filename = f"{self.file_prefix}_{i + 1}_training.npz"
            file_path = os.path.join(f"./runs/{self.file_prefix}_{i + 1}", filename)
            training_results = np.load(file_path)

            self.rewards.append(training_results["cumulative_rewards"])
            self.observations_t = cart_to_polar(training_results["observations"].squeeze())

            # compute terminal episodes
            settling_times_t = compute_settling_time(self.observations_t)
            successful_episode_t = compute_successful_episode(settling_times_t, 300)
            terminal_episode_t = terminal_episode(successful_episode_t)
            self.terminal_episodes.append(terminal_episode_t)

    def load_validation(self):
        self.settling_times = []
        self.ss_errors = []
        self.actions = []
        self.actions_mean = []
        self.actions_std = []
        self.observations_v = []

        for i in range(self.sessions):
            filename = f"{self.file_prefix}_{i + 1}_validation.npz"
            file_path = os.path.join(f"./runs/{self.file_prefix}_{i + 1}", filename)
            validation_results = np.load(file_path)

            observations_v = cart_to_polar(validation_results["observations"].squeeze())
            self.observations_v.append(observations_v)
            self.actions.append(validation_results["control_actions"].squeeze())
            self.actions_mean.append(validation_results["control_actions_mean"].squeeze())
            self.actions_std.append(validation_results["control_actions_std"].squeeze())

            settling_times_v = compute_settling_time(observations_v)
            self.settling_times.extend(settling_times_v)
            self.ss_errors.extend(compute_ss_error(observations_v, settling_times_v))

    def load_validation_det(self):
        self.settling_times_det = []
        self.ss_errors_det = []
        self.actions_det = []
        self.actions_mean_det = []
        self.actions_std_det = []
        self.observations_v_det = []

        for i in range(self.sessions):
            filename = f"{self.file_prefix}_{i + 1}_validation_det.npz"
            file_path = os.path.join(f"./runs/{self.file_prefix}_{i + 1}", filename)
            validation_results = np.load(file_path)

            observations_v_det = cart_to_polar(validation_results["observations"].squeeze())
            self.observations_v_det.append(observations_v_det)
            self.actions_det.append(validation_results["control_actions"].squeeze())
            self.actions_mean_det.append(validation_results["control_actions_mean"].squeeze())
            self.actions_std_det.append(validation_results["control_actions_std"].squeeze())

            settling_times_v = compute_settling_time(observations_v_det)
            self.settling_times_det.extend(settling_times_v)
            self.ss_errors_det.extend(compute_ss_error(observations_v_det, settling_times_v))

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

    def settling_time(self, deterministic=False):
        settling_times = self.settling_times_det if deterministic else self.settling_times
        mean = np.mean(settling_times)
        std = np.std(settling_times)
        return Data(mean, std)

    def ss_error(self, deterministic=False):
        ss_errors = self.ss_errors_det if deterministic else self.ss_errors
        mean = np.mean(ss_errors)
        std = np.std(ss_errors)
        return Data(mean, std)

    def action_std(self):
        mean = np.mean(self.actions_std)
        std = np.std(self.actions_std)
        return Data(mean, std)

    def control_effort(self, deterministic=False):
        actions = self.actions_det if deterministic else self.actions
        control_norm = []
        for control_actions in actions:
            control_norm.extend(np.linalg.norm(control_actions, axis=-1))
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


def plot_rewards(*agents, labels=None, filename=None, moving_avg_size=5, training_length=200, detail_window=False,
                 start_detail=8000, sigma=200000):
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

    if detail_window:
        # Add red horizontal line at value 1e8
        ax.axhline(y=sigma, color='red', linestyle='--', label='$\sigma$')
    ax.legend()
    if detail_window:
        # Calculate the position of the detail window based on the zoomed region
        detail_width = 2.2  # Adjust the width of the detail window as needed
        detail_height = 1.4  # Adjust the height of the detail window as needed
        axin = inset_axes(ax, width=detail_width, height=detail_height, loc='lower right')

        for idx, agent in enumerate(agents):
            label = labels[idx] if labels else agent.file_prefix
            # smoothing signals for detail window
            terminal_rewards = [rewards[start_detail:] for rewards in agent.rewards]
            smoothed_rewards_detail = moving_average(terminal_rewards, moving_avg_size)
            # compute mean and standard deviation for detail window
            mean_rewards_detail = np.mean(smoothed_rewards_detail, axis=0)
            std_rewards_detail = np.std(smoothed_rewards_detail, axis=0)

            axin.plot(range(start_detail, start_detail + len(mean_rewards_detail)), mean_rewards_detail,
                      label=label)  # Set x ticks explicitly
            axin.fill_between(range(len(mean_rewards_detail)), mean_rewards_detail - std_rewards_detail,
                              mean_rewards_detail + std_rewards_detail, alpha=0.2)

        axin.set_xlim(start_detail, training_length)  # Set x-axis limits for detail view
        axin.set_ylim(min(mean_rewards_detail), max(mean_rewards_detail))  # Adjust y-axis limits if needed
        axin.set_ylim(0, 3 * sigma)
        axin.grid(True)
        axin.xaxis.tick_top()  # Move x-axis ticks to the top

        # Add red horizontal line at value 1e8 to detail plot
        axin.axhline(y=sigma, color='red', linestyle='--', label='$\sigma$')
        axin.legend()

    if filename:
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='png')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'rewards.png'), format='png')

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
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='eps')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'training_metrics.eps'), format='eps')
    plt.show()


def plot_validation_metrics(*agents, labels=None, filename=None):
    # Define the observations
    metrics = ['Episode', '$e_g$', '$torque$']
    # stats = [[agent.settling_time(deterministic=deterministic),
    #           agent.ss_error(deterministic=deterministic),
    #           agent.control_effort(deterministic=deterministic)] for agent in agents]

    agent = agents[0]
    stats = [[agent.settling_time(), agent.ss_error(), agent.control_effort()],
             [agent.settling_time(deterministic=True),
              agent.ss_error(deterministic=True),
              agent.control_effort(deterministic=True)]]

    means = [[stat.mean for stat in stat_list] for stat_list in stats]
    stds = [[stat.std for stat in stat_list] for stat_list in stats]

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    for i, metric in enumerate(metrics):
        for j in range(2):
            agent_label = labels[j] if labels else "agent_name"
            axs[i].errorbar([agent_label], [means[j][i]], yerr=[stds[j][i]], fmt='o', capsize=5, label=metric)
            print(f"Mean {metric}: {means[j][i]}, Std {metric}: {stds[j][i]}")  # Print mean and std
        axs[i].grid(True)
        axs[i].set_ylabel(metric)

    axs[0].set_title('Settling Time')
    axs[1].set_title('Steady State Error')
    axs[2].set_title('Control Effort')

    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='eps')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'validation_metrics.eps'), format='eps')
    plt.show()


def plot_actions(agent, labels=None, filename=None, episode_length=400, session=1):
    # the function selects the first validation episode of the first realization of the agent
    label = labels if labels else agent.file_prefix

    actions_mean = agent.actions_mean[session - 1][0, :]
    actions_std = agent.actions_std[session - 1][0, :]

    actions_std_dev = agent.action_std()

    # Create a GridSpec layout
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[7, 3])  # 70% and 30% width

    # Create the first subplot (70% width)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(actions_mean, label=label)
    ax0.fill_between(range(len(actions_mean)), actions_mean - actions_std,
                     actions_mean + actions_std, alpha=0.2)
    ax0.set_xlabel('Timesteps')
    ax0.set_ylabel(r'$torque [Nm]$')
    ax0.set_title('Control action')
    ax0.grid(True)
    ax0.set_xlim(0, episode_length)

    # Create the second subplot (30% width)
    ax1 = fig.add_subplot(gs[1])
    ax1.errorbar(label, actions_std_dev.mean,
                 yerr=actions_std_dev.std, fmt='o', capsize=5, label="Std Deviations")
    ax1.grid(True)
    ax1.set_ylabel(r'$torque [Nm]$')
    ax1.set_title('Standard Deviations')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    if filename:
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='png')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'actions.png'), format='png')

    # Show the figure
    plt.show()


def plot_agent_data(agent, labels=None, filename=None, episode_length=400, session=1, deterministic=True):
    # Extract observations for plotting
    if deterministic:
        control_action = agent.actions_det[session - 1][0, :]
        angular_position = agent.observations_v_det[session - 1][0, :, 0]
        angular_velocity = agent.observations_v_det[session - 1][0, :, 1]
    else:
        control_action = agent.actions[session - 1][0, :]
        angular_position = agent.observations_v[session - 1][0, :, 0]
        angular_velocity = agent.observations_v[session - 1][0, :, 1]

    # Create the subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # Plot angular position
    axs[0].plot(angular_position, label='Angular Position')
    axs[0].set_xlabel('Timesteps')
    axs[0].set_ylabel('Angular Position [rad]')
    axs[0].set_title('Angular Position')
    axs[0].grid(True)
    axs[0].set_xlim(0, episode_length)

    # Plot angular velocity
    axs[1].plot(angular_velocity, label='Angular Velocity', color='orange')
    axs[1].set_xlabel('Timesteps')
    axs[1].set_ylabel('Angular Velocity [rad/s]')
    axs[1].set_title('Angular Velocity')
    axs[1].grid(True)
    axs[1].set_xlim(0, episode_length)

    # Plot control actions
    axs[2].plot(control_action, label='Control Action')
    axs[2].set_xlabel('Timesteps')
    axs[2].set_ylabel(r'Torque [Nm]')
    axs[2].set_title('Control Action')
    axs[2].grid(True)
    axs[2].set_xlim(0, episode_length)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    if filename:
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='png')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'agent_data.png'), format='png')

    # Show the figure
    plt.show()
