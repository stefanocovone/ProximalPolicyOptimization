import os
import matplotlib.pyplot as plt
from agents_plotting_shep_multi import (AgentResults, plot_rewards, plot_training_metrics,
                                  plot_validation_metrics, plot_agent_data)

FIGURES_FOLDER = './Figures/'

save_folder = "Figures"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# STANDARD GYM REWARD

sessions = 1
env_id = "Shepherding-v0"
agents_list = ["PPO_1M_codetest", "Lama_1M"]
agents_label = ["Data-driven", "Heuristic"]
agents = []

# Create and load agents
for agent_type in agents_list:
    agent = AgentResults(agent_type, env_id, sessions)
    agent.load_training()
    agent.load_validation()
    agents.append(agent)

# Plotting
plot_rewards(*agents, labels=agents_label, filename="shepherdingPPO_L3_rewards.pdf", moving_avg_size=1000,
             training_length=200000)
plot_training_metrics(*agents, labels=agents_label, filename="shepherdingPPO_L3_training.pdf")
plot_validation_metrics(*agents, labels=agents_label, filename="shepherdingPPO_L3_validation.pdf")

plt.show()
