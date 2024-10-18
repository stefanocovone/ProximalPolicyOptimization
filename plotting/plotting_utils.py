import os
import numpy as np
import matplotlib.pyplot as plt

FIGURES_FOLDER = './Figures/'


def ensure_folder_exists(folder):
    """
    Ensure that a folder exists.

    Args:
        folder (str): Path to the folder.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def plot_rewards(agents, labels=None, filename=None, moving_avg_size=100, training_length=10000):
    """
    Plot cumulative rewards for agents.

    Args:
        agents (list): List of AgentResults instances.
        labels (list): List of labels for the agents.
        filename (str): Filename to save the plot.
        moving_avg_size (int): Window size for moving average.
        training_length (int): Maximum length of training episodes to plot.
    """
    ensure_folder_exists(FIGURES_FOLDER)
    plt.figure(figsize=(10, 5))

    for idx, agent in enumerate(agents):
        label = labels[idx] if labels else agent.file_prefix
        # Compute moving average for each session separately
        smoothed_rewards_sessions = []
        for rewards in agent.rewards:
            smoothed_rewards = agent.moving_average([rewards], moving_avg_size)[0]
            smoothed_rewards_sessions.append(smoothed_rewards)

        # Ensure all sessions have the same length
        max_length = min(training_length, max(len(r) for r in smoothed_rewards_sessions))
        # Truncate or pad each session to the max_length
        session_rewards = []
        for rewards in smoothed_rewards_sessions:
            if len(rewards) >= max_length:
                session_rewards.append(rewards[:max_length])
            else:
                # Pad with the last value if the session is shorter
                padding = np.full(max_length - len(rewards), rewards[-1])
                session_rewards.append(np.concatenate([rewards, padding]))

        # Convert to a 2D array of shape (num_sessions, max_length)
        all_rewards = np.array(session_rewards)

        # Compute mean and standard deviation across sessions
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        episodes = np.arange(1, max_length + 1)
        plt.plot(episodes, mean_rewards, label=label)
        plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

    plt.xlabel('Episodes')
    plt.ylabel(r'$J_e^\pi$')
    plt.title('Cumulative Reward')
    plt.grid(True)
    plt.xlim(0, training_length)
    plt.legend()

    if filename:
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='pdf')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'rewards.pdf'), format='pdf')
    plt.show()


def plot_training_metrics(agents, labels=None, filename=None):
    """
    Plot training metrics for agents.

    Args:
        agents (list): List of AgentResults instances.
        labels (list): List of labels for the agents.
        filename (str): Filename to save the plot.
    """
    ensure_folder_exists(FIGURES_FOLDER)
    metrics = ['$J_{avg}^\pi$', '$J_{avg,t}^\pi$', 'Terminal Episode']
    stats = [[agent.average_reward(), agent.average_term_reward(), agent.terminal_episode_data()] for agent in agents]

    # Transpose stats to have metrics as the first index
    metrics_stats = list(zip(*stats))  # metrics_stats[metric][agent]

    # For each metric, extract means and stds
    means = [[stat.mean for stat in metric_stats] for metric_stats in metrics_stats]
    stds = [[stat.std for stat in stat_stats] for stat_stats in metrics_stats]

    # Define colors for each agent
    num_agents = len(agents)
    # Use a colormap to generate distinct colors
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(num_agents)]

    # Print and store observations in results.txt
    with open("../results.txt", "w") as file:
        for i, metric in enumerate(metrics):
            file.write(f"Metric: {metric}\n")
            for j, agent in enumerate(agents):
                agent_label = labels[j] if labels else agent.file_prefix
                file.write(f"Agent: {agent_label}, Mean: {means[i][j]}, Std: {stds[i][j]}\n")
            file.write("\n")

    # Plotting
    num_metrics = len(metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(10, 3))

    if num_metrics == 1:
        axs = [axs]  # Ensure axs is iterable

    x_labels = [labels[j] if labels else agents[j].file_prefix for j in range(num_agents)]

    # Center positions for errorbars
    center = 0.5
    spread = 0.05  # Adjust this value to control the spacing between errorbars
    if num_agents > 1:
        x_positions = np.linspace(center - spread, center + spread, num_agents)
    else:
        x_positions = [center]  # If only one agent, place it at the center

    for i, metric in enumerate(metrics):
        ax = axs[i]
        mean_values = [means[i][j] for j in range(num_agents)]
        std_values = [stds[i][j] for j in range(num_agents)]

        # Plot errorbars
        for j in range(num_agents):
            ax.errorbar(x_positions[j], mean_values[j], yerr=std_values[j], fmt='o', capsize=5,
                        color=colors[j], markersize=8)

        # Center the errorbars under each agent label
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.grid(True)
        ax.set_ylabel(metric)
        ax.set_title(metric)

        # Adjust x-axis limits to focus on the center area
        ax.set_xlim(center - spread * 2, center + spread * 2)

        # Print mean and std for each agent and metric
        for j in range(num_agents):
            print(f"Mean {metric} for {x_labels[j]}: {mean_values[j]}, Std: {std_values[j]}")

    # Removed the legend as per your request

    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='pdf')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'training_metrics.pdf'), format='pdf')
    plt.show()


def plot_validation_metrics(agents, labels=None, filename=None):
    """
    Plot validation metrics for agents.

    Args:
        agents (list): List of AgentResults instances.
        labels (list): List of labels for the agents.
        filename (str): Filename to save the plot.
    """
    ensure_folder_exists(FIGURES_FOLDER)
    metrics = ['Settling Time', 'Success Rate', 'Cooperative Metric']
    data_functions = ['settling_time_data', 'success_rate_data', 'cooperative_metric_data']

    # Collect data for each agent and metric
    stats = []
    for agent in agents:
        agent_stats = []
        for func_name in data_functions:
            data = getattr(agent, func_name)()
            agent_stats.append(data)
        stats.append(agent_stats)

    # Transpose stats to have metrics as the first index
    metrics_stats = list(zip(*stats))  # metrics_stats[metric][agent]

    # For each metric, extract means and stds
    means = [[stat.mean for stat in metric_stats] for metric_stats in metrics_stats]
    stds = [[stat.std for stat in metric_stats] for metric_stats in metrics_stats]

    # Define colors for each agent
    num_agents = len(agents)
    # Use a colormap to generate distinct colors
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(num_agents)]

    # Plotting
    num_metrics = len(metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(10, 3))

    if num_metrics == 1:
        axs = [axs]  # Ensure axs is iterable

    x_labels = [labels[j] if labels else agents[j].file_prefix for j in range(num_agents)]

    # Center positions for error bars
    center = 0.5
    spread = 0.05  # Adjust this value to control the spacing between error bars
    if num_agents > 1:
        x_positions = np.linspace(center - spread, center + spread, num_agents)
    else:
        x_positions = [center]  # If only one agent, place it at the center

    for i, metric in enumerate(metrics):
        ax = axs[i]
        mean_values = means[i]
        std_values = stds[i]

        # Plot error bars
        for j in range(num_agents):
            ax.errorbar(x_positions[j], mean_values[j], yerr=std_values[j], fmt='o', capsize=5,
                        color=colors[j], markersize=8)

        # Center the error bars under each agent label
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.grid(True)
        ax.set_ylabel(metric)
        ax.set_title(metric)

        # Adjust x-axis limits to focus on the center area
        ax.set_xlim(center - spread * 2, center + spread * 2)

        # Print mean and std for each agent and metric
        for j in range(num_agents):
            print(f"Mean {metric} for {x_labels[j]}: {mean_values[j]}, Std: {std_values[j]}")

    # Remove the legend as per your style
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='pdf')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'validation_metrics.pdf'), format='pdf')
    plt.show()


def plot_agent_data(agent, episode_index=0, filename=None):
    """
    Plot agent data for a specific episode.

    Args:
        agent (AgentResults): AgentResults instance.
        episode_index (int): Index of the episode to plot.
        filename (str): Filename to save the plot.
    """
    ensure_folder_exists(FIGURES_FOLDER)
    data = agent.observations_v[episode_index]
    herder_positions = data[:, :2]  # Assuming herder positions are the first two columns
    target_positions = data[:, -2:]  # Assuming target positions are the last two columns

    # Compute polar coordinates
    herder_radius = np.linalg.norm(herder_positions, axis=-1)
    herder_angle = np.arctan2(herder_positions[:, 1], herder_positions[:, 0])
    target_radius = np.linalg.norm(target_positions, axis=-1)
    target_angle = np.arctan2(target_positions[:, 1], target_positions[:, 0])

    # Relative distance and phase difference
    relative_distance = np.linalg.norm(herder_positions - target_positions, axis=-1)
    phase_difference = herder_angle - target_angle
    phase_difference = (phase_difference + np.pi) % (2 * np.pi) - np.pi

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(herder_radius, label='Herder Radius')
    axs[0].plot(target_radius, label='Target Radius')
    axs[0].axhline(y=5, color='r', linestyle='--', label='Goal Radius')
    axs[0].set_title('Herder and Target Radii')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Radius')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(relative_distance, label='Relative Distance')
    axs[1].axhline(y=2.5, color='r', linestyle='--', label='Repulsion Radius')
    axs[1].set_title('Relative Distance Between Herder and Target')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Distance')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(phase_difference, label='Phase Difference')
    axs[2].set_title('Phase Difference Between Herder and Target')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Phase Difference (radians)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()

    if filename:
        plt.savefig(os.path.join(FIGURES_FOLDER, filename), format='pdf')
    else:
        plt.savefig(os.path.join(FIGURES_FOLDER, 'agent_data.pdf'), format='pdf')
    plt.show()
