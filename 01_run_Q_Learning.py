from environments.gridworld_env import create_env
from agents.q_learning_agent import train_q_learning, visualize_q_table, test_agent

# Hyperparameters for Q-learning
learning_rate = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
no_episodes = 5000  # Reduced for faster initial testing

# Environment configuration
# FIXED: Changed from list to tuple to match environment expectations
goal_pos = (3, 3)
pithole_positions = [(1, 3), (3, 1), (5, 3), (3, 5)]
obstacle_positions = [(1, 1), (1, 5), (5, 1), (5, 5)]

# Create environment
env = create_env(goal_pos=goal_pos,
                 pithole_positions=pithole_positions,
                 obstacle_positions=obstacle_positions)

print("="*60)
print("Ball Environment - Q-Learning Training")
print("="*60)
print(f"Field Size: {env.field_size}x{env.field_size}")
print(f"Goal Position: {goal_pos}")
print(f"Obstacles: {len(obstacle_positions)} positions")
print(f"Pitholes: {len(pithole_positions)} positions")
print(f"Training Episodes: {no_episodes}")
print(f"Learning Rate (α): {learning_rate}")
print(f"Discount Factor (γ): {gamma}")
print(f"Initial Epsilon: {epsilon}")
print("="*60)
print()

# Train the Q-learning agent
q_table, episode_rewards = train_q_learning(
    env=env,
    no_episodes=no_episodes,
    epsilon=epsilon,
    epsilon_min=epsilon_min,
    epsilon_decay=epsilon_decay,
    alpha=learning_rate,
    gamma=gamma,
    q_table_save_path="q_table.npy"
)

print("\n" + "="*60)
print("Visualizing Q-table...")
print("="*60)

# Visualize the learned Q-table
visualize_q_table(
    pithole_positions=pithole_positions,
    obstacle_positions=obstacle_positions,
    goal_pos=goal_pos,
    actions=["Backward", "Forward", "Right", "Left"],
    q_values_path="q_table.npy"
)

print("\n" + "="*60)
print("Testing the trained agent...")
print("="*60)

# Test the trained agent
test_agent(env=env, q_table_path="q_table.npy", num_episodes=5)

print("\n" + "="*60)
print("Training and Testing Complete!")
print("="*60)