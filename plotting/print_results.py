import os
import matplotlib.pyplot as plt
from agents_plotting import (AgentResults, plot_rewards, plot_training_metrics,
                             plot_validation_metrics, plot_actions, plot_agent_data)

FIGURES_FOLDER = './Figures/'

save_folder = "Figures"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# STANDARD GYM REWARD

sessions = 5
env_id = "Pendulum-v1"
agents_list = ["PPO"]
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
plot_rewards(*agents, labels=agents_label, filename="pendulumPPO_rewards.png", moving_avg_size=5)
plot_training_metrics(*agents, labels=agents_label, filename="pendulumPPO_training.eps")
plot_validation_metrics(*agents, labels=["PPOs", "PPOd"], filename="pendulumPPO_validation.eps")
plot_actions(agents[0], labels=agents_label, filename="pendulumPPO_actions.png", session=1)
plot_agent_data(agents[0], labels=agents_label, filename="pendulumPPO_episode.png", session=1, deterministic=False)
plot_agent_data(agents[0], labels=agents_label, filename="pendulumPPO_episode_det.png", session=1, deterministic=True)

plt.show()
