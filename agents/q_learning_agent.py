# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):

    # Initialize the Q-table:
    # -----------------------
    q_table = np.zeros((env.field_size, env.field_size, env.action_space.n))

    # Track training metrics
    episode_rewards = []
    success_count = 0

    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    for episode in range(no_episodes):
        state, _ = env.reset()
        state = tuple(state)
        total_reward = 0
        steps = 0

        #! Step 2: Take actions in the environment until "Done" flag is triggered
        #! -------
        while True:
            #! Step 3: Define your Exploration vs. Exploitation
            #! -------
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, info = env.step(action)
            
            # Render only occasionally to speed up training
            if episode % 100 == 0 or episode == no_episodes - 1:
                env.render()

            next_state = tuple(next_state)
            total_reward += reward
            steps += 1

            #! Step 4: Update the Q-values using the Q-value update rule
            #! -------
            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            #! Step 5: Stop the episode if the agent reaches Goal or terminal states
            #! -------
            if done:
                if reward == 100:  # Reached goal
                    success_count += 1
                break
            
            # Add maximum step limit to prevent infinite loops
            if steps > 100:
                break

        #! Step 6: Perform epsilon decay
        #! -------
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        episode_rewards.append(total_reward)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = (success_count / (episode + 1)) * 100
            print(f"Episode {episode + 1}/{no_episodes} | Avg Reward (last 100): {avg_reward:.2f} | "
                  f"Success Rate: {success_rate:.2f}% | Epsilon: {epsilon:.4f}")

    #! Step 7: Close the environment window
    #! -------
    env.close()
    print("\nTraining finished.")
    print(f"Final Success Rate: {(success_count / no_episodes) * 100:.2f}%")

    #! Step 8: Save the trained Q-table
    #! -------
    np.save(q_table_save_path, q_table)
    print(f"Saved the Q-table to {q_table_save_path}")
    
    return q_table, episode_rewards


# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(pithole_positions=[(1, 3), (3, 1), (5, 3), (3, 5)],
                      obstacle_positions=[(1, 1), (1, 5), (5, 1), (5, 5)],
                      goal_pos=(3, 3),
                      actions=["Backward", "Forward", "Right", "Left"],
                      q_values_path="q_table.npy"):

    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path)

        # Create subplots for each action:
        # --------------------------------
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            # Mask the Q-values of Goal State, Pitholes, and Obstacles for visualization:
            # ----------------------------------------------------------------------------
            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_pos[0], goal_pos[1]] = True
            for pit in pithole_positions:
                mask[pit[0], pit[1]] = True
            for obs in obstacle_positions:
                mask[obs[0], obs[1]] = True

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=True, mask=mask, annot_kws={"size": 9})

            # Denote Goal(G), Pitholes(H) and Obstacles(O) in the field:
            # ----------------------------------------------------------
            ax.text(goal_pos[1] + 0.5, goal_pos[0] + 0.5, 'G', color='blue',
                    ha='center', va='center', weight='bold', fontsize=14)
            
            for pit in pithole_positions:
                ax.text(pit[1] + 0.5, pit[0] + 0.5, 'H', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)
            
            for obs in obstacle_positions:
                ax.text(obs[1] + 0.5, obs[0] + 0.5, 'O', color='black',
                        ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}', fontsize=12, weight='bold')
            ax.set_xlabel('Column', fontsize=10)
            ax.set_ylabel('Row', fontsize=10)

        plt.suptitle('Q-Table Visualization for Each Action', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig('q_table_visualization.png', dpi=150, bbox_inches='tight')
        print("Q-table visualization saved as 'q_table_visualization.png'")
        plt.show()

    except FileNotFoundError:
        print(f"No saved Q-table was found at '{q_values_path}'. Please train the Q-learning agent first or check your path.")


# Function 3: Test the trained agent
# -----------
def test_agent(env, q_table_path="q_table.npy", num_episodes=10):
    """Test the trained agent and visualize its performance"""
    try:
        q_table = np.load(q_table_path)
        
        successes = 0
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = tuple(state)
            total_reward = 0
            steps = 0
            
            print(f"\nTest Episode {episode + 1}:")
            
            while True:
                # Always exploit (no exploration during testing)
                action = np.argmax(q_table[state])
                next_state, reward, done, info = env.step(action)
                env.render()
                
                next_state = tuple(next_state)
                total_reward += reward
                steps += 1
                
                state = next_state
                
                if done:
                    if reward == 100:
                        successes += 1
                        print(f"  ✓ Reached goal in {steps} steps! Total reward: {total_reward:.2f}")
                    elif reward == -50:
                        print(f"  ✗ Fell in pithole after {steps} steps. Total reward: {total_reward:.2f}")
                    elif reward == -10:
                        print(f"  ✗ Hit obstacle after {steps} steps. Total reward: {total_reward:.2f}")
                    break
                
                if steps > 100:
                    print(f"  ✗ Timeout after {steps} steps. Total reward: {total_reward:.2f}")
                    break
            
            total_rewards.append(total_reward)
        
        env.close()
        
        print(f"\n{'='*50}")
        print(f"Test Results:")
        print(f"  Success Rate: {(successes / num_episodes) * 100:.1f}%")
        print(f"  Average Reward: {np.mean(total_rewards):.2f}")
        print(f"  Best Reward: {np.max(total_rewards):.2f}")
        print(f"  Worst Reward: {np.min(total_rewards):.2f}")
        print(f"{'='*50}")
        
    except FileNotFoundError:
        print(f"No saved Q-table found at '{q_table_path}'. Please train the agent first.")