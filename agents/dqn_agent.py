# Import:
# -------
import torch
import random
import numpy as np
from collections import deque
import torch.nn.functional as F


# Replay Buffer:
# --------------
class ReplayBuffer():
    """
    Experience Replay Buffer for Deep Q-Learning.
    
    Stores transitions (state, action, reward, next_state, done) and allows
    random sampling to break correlation between consecutive experiences.
    """
    
    def __init__(self, buffer_limit):
        """
        Initialize the replay buffer.
        
        Args:
            buffer_limit (int): Maximum number of transitions to store.
                               Older transitions are automatically removed when limit is reached.
        """
        self.buffer = deque(maxlen=buffer_limit)
    
    def put(self, transition):
        """
        Add a transition to the buffer.
        
        Args:
            transition (tuple): A tuple of (state, action, reward, next_state, done_mask)
                              where done_mask is 0.0 if episode ended, 1.0 otherwise.
        """
        self.buffer.append(transition)
    
    def sample(self, n):
        """
        Sample a random batch of transitions from the buffer.
        
        Args:
            n (int): Number of transitions to sample (batch size)
            
        Returns:
            tuple: Five tensors containing batched:
                  - states: (batch_size, state_dim)
                  - actions: (batch_size, 1)
                  - rewards: (batch_size, 1)
                  - next_states: (batch_size, state_dim)
                  - done_masks: (batch_size, 1)
        """
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # OPTIMIZED: Direct conversion without intermediate np.array()
        return (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst, dtype=torch.long),
            torch.tensor(r_lst, dtype=torch.float),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst, dtype=torch.float)
        )
    
    def size(self):
        """
        Get the current number of transitions in the buffer.
        
        Returns:
            int: Number of stored transitions
        """
        return len(self.buffer)


# Training Function:
# ------------------
def train(q_net, q_target, memory, optimizer, batch_size, gamma, 
          train_steps=1, grad_clip_norm=10.0):
    """
    Train the DQN using sampled experiences from replay buffer.
    
    This function performs one or more training iterations on the Q-network
    using experiences sampled from the replay buffer. It implements the
    standard DQN update rule with target network stabilization.
    
    Args:
        q_net (torch.nn.Module): Main Q-network to be trained
        q_target (torch.nn.Module): Target Q-network for stable value estimates
        memory (ReplayBuffer): Replay buffer containing experiences
        optimizer (torch.optim.Optimizer): Optimizer for updating q_net parameters
        batch_size (int): Number of experiences to sample per training step
        gamma (float): Discount factor for future rewards (0 to 1)
        train_steps (int, optional): Number of gradient updates per call. 
                                     Default is 1. Set to 10 for faster training.
        grad_clip_norm (float, optional): Maximum gradient norm for clipping.
                                         Default is 10.0. Set to None to disable.
    
    Returns:
        float: Average loss across all training steps
    
    Training Process:
        1. Sample batch from replay buffer
        2. Compute Q-values for current states: Q(s, a)
        3. Compute target values: r + γ * max_a' Q_target(s', a')
        4. Calculate loss between Q-values and targets
        5. Backpropagate and update Q-network weights
    """
    total_loss = 0.0
    
    for _ in range(train_steps):
        # Sample a batch of experiences
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # Forward pass through main network
        q_out = q_net(s)
        q_a = q_out.gather(1, a)  # Q(s, a) - selected actions
        
        # Compute target Q-values using target network (no gradients needed)
        with torch.no_grad():
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask
        
        # Compute loss (Smooth L1 is more robust than MSE for outliers)
        loss = F.smooth_l1_loss(q_a, target)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # ADDED: Gradient clipping to prevent exploding gradients
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=grad_clip_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    # Return average loss across all training steps
    return total_loss / train_steps


# Alternative Training Function (Single Step):
# --------------------------------------------
def train_single_step(q_net, q_target, memory, optimizer, batch_size, gamma):
    """
    Simplified training function that performs exactly one gradient update.
    
    This is equivalent to calling train() with train_steps=1, but with
    a cleaner interface for standard DQN training loops.
    
    Args:
        q_net (torch.nn.Module): Main Q-network to be trained
        q_target (torch.nn.Module): Target Q-network for stable value estimates
        memory (ReplayBuffer): Replay buffer containing experiences
        optimizer (torch.optim.Optimizer): Optimizer for updating q_net parameters
        batch_size (int): Number of experiences to sample
        gamma (float): Discount factor for future rewards (0 to 1)
    
    Returns:
        float: Loss value for this training step
    """
    return train(q_net, q_target, memory, optimizer, batch_size, gamma, 
                 train_steps=1, grad_clip_norm=10.0)