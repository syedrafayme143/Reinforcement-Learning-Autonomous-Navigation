# main_dqn.py

# Imports:
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from networks.dqn_network import Qnet
from agents.dqn_agent import ReplayBuffer, train
from environments.gridworld_env import create_env

# User definitions:
train_dqn = True
test_dqn = True  
render_during_training = False  # Set to True to see training visualization (slower)
render_during_testing = True

# Hyperparameters:
learning_rate = 0.0005  # Reduced for more stable learning
gamma = 0.98
buffer_limit = 50_000
batch_size = 32
num_episodes = 2000  # Increased for better learning
max_steps_per_episode = 200  # Maximum steps per episode to prevent infinite loops

# Output directory for results
output_dir = "Results/dqn"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ============================================================================
# TRAINING
# ============================================================================
if train_dqn:
    print("="*60)
    print("Starting DQN Training")
    print("="*60)
    
    # Create environment (no render_mode parameter - padm_env doesn't support it)
    env = create_env(
        goal_pos=(3, 3),
        pithole_positions=[[1, 3], [3, 1], [5, 3], [3, 5]],
        obstacle_positions=[[1, 1], [1, 5], [5, 1], [5, 5]]
    )

    # Initialize the Q Net and the Q Target Net
    q_net = Qnet(no_actions=env.action_space.n, no_states=env.observation_space.shape[0])
    q_target = Qnet(no_actions=env.action_space.n, no_states=env.observation_space.shape[0])

    # Load existing model weights if available
    model_path = os.path.join(output_dir, "dqn.pth")
    try:
        q_net.load_state_dict(torch.load(model_path))
        q_target.load_state_dict(q_net.state_dict())
        print("✓ Loaded existing model weights from", model_path)
    except FileNotFoundError:
        print("✗ No existing weights found. Starting fresh training.")
        q_target.load_state_dict(q_net.state_dict())

    # Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 50
    update_target_interval = 20  # Update target network every N episodes
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    # Tracking metrics
    rewards = []
    losses = []
    success_count = 0

    print(f"\nTraining for {num_episodes} episodes...")
    print(f"Learning Rate: {learning_rate}, Gamma: {gamma}, Batch Size: {batch_size}")
    print("-"*60)

    for n_epi in range(num_episodes):
        # Epsilon decay: Start at 1.0, decay to 0.01
        epsilon = max(0.01, 1.0 - 0.99 * (n_epi / num_episodes))

        s, _ = env.reset()  # FIXED: Properly unpack reset() which returns (state, info)
        done = False
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        steps = 0

        # Run episode
        for step in range(max_steps_per_episode):
            # Select action
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            
            # Take action in environment
            s_prime, reward, done, info = env.step(a)  # FIXED: Unpack 4 values (padm_env returns 4)

            # Render if requested (only occasionally to speed up training)
            if render_during_training and n_epi % 100 == 0:
                env.render()

            done_mask = 0.0 if done else 1.0

            # Store transition in replay buffer
            memory.put((s, a, reward, s_prime, done_mask))
            s = s_prime

            episode_reward += reward
            steps += 1

            # Train if buffer has enough samples
            if memory.size() > 2000:
                loss = train(q_net, q_target, memory, optimizer, batch_size, gamma)
                episode_loss += loss
                loss_count += 1

            if done:
                if reward == 15:  # Reached goal
                    success_count += 1
                break

        # Update target network periodically
        if n_epi % update_target_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())

        # Record metrics
        rewards.append(episode_reward)
        avg_loss = episode_loss / max(loss_count, 1)
        losses.append(avg_loss)

        # Print progress
        if n_epi % print_interval == 0 and n_epi != 0:
            avg_reward = np.mean(rewards[-print_interval:])
            success_rate = (success_count / (n_epi + 1)) * 100
            print(f"Episode {n_epi:4d} | Avg Reward: {avg_reward:7.2f} | "
                  f"Steps: {steps:3d} | Success Rate: {success_rate:5.2f}% | "
                  f"Buffer: {memory.size():5d} | Epsilon: {epsilon:.4f} | Loss: {avg_loss:.4f}")

        # Early stopping if consistently reaching goal
        if n_epi >= 100 and success_count / (n_epi + 1) > 0.95:
            print(f"\n✓ Training converged at episode {n_epi} with {success_rate:.2f}% success rate!")
            break

    env.close()
    
    # Save model
    torch.save(q_net.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    ax1.plot(rewards, alpha=0.6, label='Episode Reward')
    # Plot moving average
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot losses
    ax2.plot(losses, alpha=0.6, label='Loss')
    if len(losses) >= window:
        moving_avg_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(losses)), moving_avg_loss, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Training curves saved to {plot_path}")
    plt.show()

    print("\n" + "="*60)
    print(f"Training Complete!")
    print(f"Final Success Rate: {(success_count / len(rewards)) * 100:.2f}%")
    print(f"Total Episodes: {len(rewards)}")
    print("="*60)


# ============================================================================
# TESTING
# ============================================================================
if test_dqn:
    print("\n" + "="*60)
    print("Testing the Trained DQN Agent")
    print("="*60)
    
    # Create environment
    env = create_env(
        goal_pos=(3, 3),
        pithole_positions=[[1, 3], [3, 1], [5, 3], [3, 5]],
        obstacle_positions=[[1, 1], [1, 5], [5, 1], [5, 5]]
    )

    # Load trained model
    dqn = Qnet(no_actions=env.action_space.n, no_states=env.observation_space.shape[0])
    model_path = os.path.join(output_dir, "dqn.pth")
    
    try:
        dqn.load_state_dict(torch.load(model_path))
        print(f"✓ Loaded trained model from {model_path}\n")
    except FileNotFoundError:
        print(f"✗ No trained model found at {model_path}. Please train first.")
        exit()

    dqn.eval()  # Set to evaluation mode

    num_test_episodes = 10
    test_rewards = []
    test_successes = 0

    for ep in range(num_test_episodes):
        s, _ = env.reset()  # FIXED: Properly unpack reset()
        episode_reward = 0
        steps = 0

        print(f"\nTest Episode {ep + 1}/{num_test_episodes}:")

        for step in range(max_steps_per_episode):
            # Get action from trained network (no exploration)
            with torch.no_grad():
                q_values = dqn(torch.from_numpy(s).float())
                action = q_values.argmax().item()
            
            # Take action
            s_prime, reward, done, info = env.step(action)
            
            # Render if requested
            if render_during_testing:
                env.render()
            
            s = s_prime
            episode_reward += reward
            steps += 1

            if done:
                if reward == 100:
                    test_successes += 1
                    print(f"  ✓ Reached goal in {steps} steps! Reward: {episode_reward:.2f}")
                elif reward == -50:
                    print(f"  ✗ Fell in pithole after {steps} steps. Reward: {episode_reward:.2f}")
                elif reward == -10:
                    print(f"  ✗ Hit obstacle after {steps} steps. Reward: {episode_reward:.2f}")
                break
        
        if not done:
            print(f"  ✗ Episode timeout after {steps} steps. Reward: {episode_reward:.2f}")
        
        test_rewards.append(episode_reward)

    env.close()

    print("\n" + "="*60)
    print("Testing Results:")
    print("-"*60)
    print(f"Success Rate: {(test_successes / num_test_episodes) * 100:.1f}%")
    print(f"Average Reward: {np.mean(test_rewards):.2f}")
    print(f"Best Reward: {np.max(test_rewards):.2f}")
    print(f"Worst Reward: {np.min(test_rewards):.2f}")
    print("="*60)