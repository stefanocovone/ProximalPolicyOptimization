import os
import matplotlib.pyplot as plt
from agents_plotting_shep import (AgentResults, plot_rewards, plot_training_metrics,
                                  plot_validation_metrics, plot_agent_data)

FIGURES_FOLDER = './Figures/'

save_folder = "Figures"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# STANDARD GYM REWARD

sessions = 1
env_id = "Shepherding-v0"
agents_list = ["PPO_1M_random3"]
agents_label = ["PPO"]
agents = []

# Create and load agents
for agent_type in agents_list:
    agent = AgentResults(agent_type, env_id, sessions)
    agent.load_training()
    # agent.load_validation()
    agents.append(agent)

# Plotting
plot_rewards(*agents, labels=agents_label, filename="shepherdingPPO_rewards.png", moving_avg_size=1000,
             training_length=200000)
plot_training_metrics(*agents, labels=agents_label, filename="shepherdingPPO_training.eps")
plot_validation_metrics(*agents, labels=agents_label, filename="shepherdingPPO_validation.eps")
plot_agent_data(agents[0], filename="shepherdingPPO_episode.png")


plt.show()
