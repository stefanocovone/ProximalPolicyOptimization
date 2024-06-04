import os
import matplotlib.pyplot as plt
from agents_plotting_shep import (AgentResults, plot_rewards, plot_training_metrics,
                                  plot_validation_metrics, plot_actions, plot_agent_data)

FIGURES_FOLDER = './Figures/'

save_folder = "Figures"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# STANDARD GYM REWARD

sessions = 1
env_id = "Shepherding-v0"
agents_list = ["PPO_wide_region_noise1"]
agents_label = ["PPO"]
agents = []

# Create and load agents
for agent_type in agents_list:
    agent = AgentResults(agent_type, env_id, sessions)
    agent.load_training()
    agent.load_validation()
    agent.load_validation_det()
    agents.append(agent)

# Plotting
plot_rewards(*agents, labels=agents_label, filename="shepherdingPPO_rewards.png", moving_avg_size=100,
             training_length=10000)
plot_training_metrics(*agents, labels=agents_label, filename="shepherdingPPO_training.eps")
plot_validation_metrics(*agents, labels=["PPOs", "PPOd"], filename="shepherdingPPO_validation.eps")
plot_agent_data(agents[0], labels=agents_label, filename="shepherdingPPO_episode.png", session=1, deterministic=False)
plot_agent_data(agents[0], labels=agents_label, filename="shepherdingPPO_episode_det.png", session=1, deterministic=True)

plt.show()
