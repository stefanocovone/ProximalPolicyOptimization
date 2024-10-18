import os
from agent_results import AgentResults
from plotting_utils import (
    plot_rewards,
    plot_training_metrics,
    plot_validation_metrics,
    plot_agent_data
)

# Ensure the figures folder exists
FIGURES_FOLDER = './Figures/'
if not os.path.exists(FIGURES_FOLDER):
    os.makedirs(FIGURES_FOLDER)

# Parameters
sessions = 3
env_id = "Shepherding-v0"
agents_list = ["PPO_1M_test_LR3", "PPO_2M"]
agents_label = ["Data-driven", "BB"]
agents = []

# Create and load agents
for agent_type in agents_list:
    agent = AgentResults(agent_type, env_id, sessions)
    agent.load_training()
    agent.load_validation()
    agents.append(agent)

# Plotting
plot_rewards(agents, labels=agents_label, filename="shepherdingPPO_L3_rewards.pdf", moving_avg_size=1000,
             training_length=200000)
plot_training_metrics(agents, labels=agents_label, filename="shepherdingPPO_L3_training.pdf")
plot_validation_metrics(agents, labels=agents_label, filename="shepherdingPPO_L3_validation.pdf")

# Plot data for the first agent
# plot_agent_data(agents[0], episode_index=0, filename="agent_data.pdf")
