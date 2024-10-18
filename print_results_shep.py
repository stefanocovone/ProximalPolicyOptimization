import os
import matplotlib.pyplot as plt
from agents_plotting_shep_new import (AgentResults, plot_rewards, plot_training_metrics,
                                  plot_validation_metrics, plot_agent_data)

FIGURES_FOLDER = './Figures/'

save_folder = "Figures"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# STANDARD GYM REWARD

sessions = 5
env_id = "Shepherding-v0"
agents_list = ["PPO", "Lama"]
agents_label = ["PPO", "Lama"]
agents = []

# Create and load agents
for agent_type in agents_list:
    agent = AgentResults(agent_type, env_id, sessions)
    agent.load_training()
    agent.load_validation()
    agents.append(agent)

# Plotting
# plot_rewards(*agents, labels=agents_label, filename="shepherdingPPO_L1_rewards.pdf", moving_avg_size=100,
#              training_length=20000)
# plot_training_metrics(*agents, labels=agents_label, filename="shepherdingPPO_L1_training.pdf")
plot_validation_metrics(*agents, labels=agents_label, filename="shepherdingPPO_L1_validation.pdf")
# plot_agent_data(agents[0], filename="shepherdingPPO_L1_episode.pdf")


plt.show()
