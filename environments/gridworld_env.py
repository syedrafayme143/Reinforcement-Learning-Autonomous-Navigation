import gym
from gym import spaces
import numpy as np
import pygame
import cv2  
import os   

# Environment: A ball rolling on the field and searching to hit goal post, in between it has to encountered with some obstacles and also there is a danger of falling in the pitholes which are present on the field.

class BallEnv(gym.Env): 

    def __init__(self, field_size=6, goal_pos=(3, 3), obstacle_positions=None, pithole_positions=None):
        super().__init__()
        # Size of field
        self.field_size = field_size
        # Four Possible actions: Forward, Backward, Left and Right
        self.action_space = gym.spaces.Discrete(4)
        # Reward
        self.reward = 0
        # Defining the field  
        self.observation_space = spaces.Box(low=0, high=field_size-1, shape=(2,), dtype=np.int32)
        # Starting Position of a ball in the horizontal field
        self.state = np.array([0, 0])
        # Goal Post Position in the horizontal field.                               
        self.goal_pos = np.array(goal_pos)
        # Positions of the obstacles in the horizontal field
        self.obstacle_positions = np.array(obstacle_positions) if obstacle_positions else np.array([[1, 1], [1, 5], [5, 1], [5, 5]])
        # Position of pit holes in the horizontal field
        self.pithole_positions = np.array(pithole_positions) if pithole_positions else np.array([[1, 3], [3, 1], [5, 3], [3, 5]])
        
        # Track previous distance for reward shaping
        self.previous_distance = None

        # --- NEW: Video Recording Setup ---
        self.video_writer = None
        self.output_dir = "Results/q_learning"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def reset(self):
        self.state = np.array([0, 0])
        self.reward = 0
        distance_to_goalpost = abs(self.state[0] - self.goal_pos[0]) + abs(self.state[1] - self.goal_pos[1])
        self.previous_distance = distance_to_goalpost
        info = {"Distance to Goal Post": distance_to_goalpost}
        return self.state, info

    def step(self, action):
        # Moving Backward
        if action == 0 and self.state[0] > 0:
            self.state[0] -= 1
        # Moving Forward
        elif action == 1 and self.state[0] < self.field_size - 1:
            self.state[0] += 1
        # Moving Right
        elif action == 2 and self.state[1] < self.field_size - 1:
            self.state[1] += 1
        # Moving Left
        elif action == 3 and self.state[1] > 0:
            self.state[1] -= 1

        self.done = False
        
        # Calculate current distance to goal
        distance_to_goalpost = abs(self.state[0] - self.goal_pos[0]) + abs(self.state[1] - self.goal_pos[1])
        
        # Reward shaping based on distance improvement
        distance_reward = self.previous_distance - distance_to_goalpost
        self.previous_distance = distance_to_goalpost
        
        # Check terminal conditions and assign rewards
        if np.array_equal(self.state, self.goal_pos):
            self.reward = 15  #Reward for reaching goal
            self.done = True
        elif any(np.array_equal(self.state, pithole) for pithole in self.pithole_positions):
            self.reward = -10  #Penalty for falling in pithole
            self.done = True
        elif any(np.array_equal(self.state, obstacle) for obstacle in self.obstacle_positions):
            self.reward = -2  #Moderate penalty for hitting obstacle
            self.done = True   # Episode terminates on obstacle hit
        else:
            self.reward = -0.5 + distance_reward * 0.5  # Small step penalty, but reward getting closer to goal
            self.done = False
        
        info = {"Distance to Goal Post": distance_to_goalpost}
        
        # Return with proper format (state, reward, done, info)
        return self.state, self.reward, self.done, info

    def render(self):
        screen_width, screen_height = 600, 600
        cell_size = screen_width // self.field_size

        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Ball Environment")

        # Handle pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((255, 255, 255))

        def draw_star(surface, color, center, size):  # Star Represents the Goal Post
            points = []
            for i in range(5):
                angle = i * 2 * np.pi / 5 - np.pi / 2
                outer_x = center[0] + size * np.cos(angle)
                outer_y = center[1] + size * np.sin(angle)
                points.append((outer_x, outer_y))
                inner_x = center[0] + (size // 2) * np.cos(angle + np.pi / 5)
                inner_y = center[1] + (size // 2) * np.sin(angle + np.pi / 5)
                points.append((inner_x, inner_y))
            pygame.draw.polygon(surface, color, points)

        def draw_triangle(surface, color, rect):  # Triangle Represents the Hurdles in the field
            points = [
                (rect.centerx, rect.top),
                (rect.left, rect.bottom),
                (rect.right, rect.bottom)
            ]
            pygame.draw.polygon(surface, color, points)

        def draw_diamond(surface, color, rect):  # Diamond Represents the pit holes
            points = [
                (rect.centerx, rect.top),
                (rect.right, rect.centery),
                (rect.centerx, rect.bottom),
                (rect.left, rect.centery)
            ]
            pygame.draw.polygon(surface, color, points)

        for row in range(self.field_size):
            for col in range(self.field_size):
                rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                if np.array_equal(self.state, [row, col]):
                    pygame.draw.circle(self.screen, (0, 255, 0), rect.center, cell_size // 3)  # Ball is represented by Green Circle in the field.
                elif np.array_equal(self.goal_pos, [row, col]):
                    draw_star(self.screen, (0, 0, 255), rect.center, cell_size // 3)  # Goal is represented by Blue Star in the field.
                elif any(np.array_equal(obs, [row, col]) for obs in self.obstacle_positions):
                    draw_triangle(self.screen, (0, 0, 0), rect)  # Obstacles are represented by Black Triangle
                elif any(np.array_equal(pit, [row, col]) for pit in self.pithole_positions):
                    draw_diamond(self.screen, (255, 0, 0), rect)  # Pit holes are represented by Red Diamond
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        pygame.display.flip()

        # --- NEW: Capture Frame and Save to Video ---
        try:
            # 1. Capture the Pygame surface as a 3D array (Width, Height, RGB)
            view = pygame.surfarray.array3d(self.screen)
            
            # 2. Transpose it to match OpenCV format (Height, Width, RGB)
            view = view.transpose([1, 0, 2])
            
            # 3. Convert RGB (Pygame) to BGR (OpenCV)
            img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

            # 4. Initialize VideoWriter if it doesn't exist
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = os.path.join(self.output_dir, "agent_gameplay.mp4")
                self.video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (screen_width, screen_height))
            
            # 5. Write the frame
            self.video_writer.write(img_bgr)
            
        except Exception as e:
            print(f"Warning: Could not save video frame. Error: {e}")

    def close(self):
        # --- NEW: Release Video Writer ---
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"Video saved to {os.path.join(self.output_dir, 'agent_gameplay.mp4')}")
            
        if hasattr(self, 'screen'):
            pygame.quit()
            del self.screen


# Function 1: Create an instance of the environment
# -----------
def create_env(goal_pos, obstacle_positions, pithole_positions):
    # Create the environment:
    # -----------------------
    env = BallEnv(goal_pos=goal_pos, pithole_positions=pithole_positions, obstacle_positions=obstacle_positions)
    return env
