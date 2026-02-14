# Import:
# -------
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# Deep Q-Network:
# ---------------
class Qnet(nn.Module):
    def __init__(self, no_actions, no_states):
        """
        Deep Q-Network model for reinforcement learning.
        
        Args:
            no_actions (int): Number of possible actions in the environment
            no_states (int): Dimensionality of the state space
        """
        super(Qnet, self).__init__()
        self.no_actions = no_actions  # Store for use in sample_action
        
        # Neural network layers
        self.fc1 = nn.Linear(no_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, no_actions)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, observation, epsilon):
        """
        Sample an action using epsilon-greedy policy.
        
        Args:
            observation: Current state observation (torch tensor)
            epsilon: Exploration probability
            
        Returns:
            Selected action (int)
        """
        # Get Q-values for current state
        with torch.no_grad():  # No need to compute gradients during action selection
            q_values = self.forward(observation)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Exploration: random action
            # FIXED: Use self.no_actions instead of hardcoded 3
            return random.randint(0, self.no_actions - 1)
        else:
            # Exploitation: action with highest Q-value
            return q_values.argmax().item()