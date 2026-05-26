import random as pyrandom
import gymnasium as gym
import mesa
import numpy as np

from worm_environment import WormWorld

class WormGymEnv(gym.Env):
    
    def __init__(self, width=20, height=20, nociception = False):
        super().__init__()
        self.nociception = nociception
        
        # Create the Mesa world
        self.world = WormWorld(width=width, height=height)
        
        # Tell PPO what observations look like
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Tell PPO what actions are possible
        self.action_space = gym.spaces.Discrete(5)
        
        # Tracking variables
        self.current_step = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.world = WormWorld(width = 20, height = 20)
        self.current_step = 0

        observation = self._get_observation()
        return observation, {}
    
    def _get_observation(self):
        worm = self.world.agents[0]
        gradient_value = self.world.gradient[worm.pos[0]][worm.pos[1]]
        hunger_value = worm.hunger
        if self.nociception:
            noxious_value = self.world.noxious_gradient[worm.pos[0]][worm.pos[1]]
            return np.array([gradient_value, hunger_value, noxious_value], dtype=np.float32)
        return np.array([gradient_value, hunger_value], dtype=np.float32)
    
    def step(self, action):
        self.current_step += 1
        
        # Get the worm
        worm = list(self.world.agents)[0]
        
        # Capture gradient BEFORE moving
        old_noxious = self.world.noxious_gradient[worm.pos[0]][worm.pos[1]]
        old_gradient = self.world.gradient[worm.pos[0]][worm.pos[1]]
        
        # Execute the action - move the worm
        possible_steps = self.world.grid.get_neighborhood(
            worm.pos,
            moore=False,
            include_center=True
        )
        
        # Map action number to movement
        neighbors = list(possible_steps)
        current_x, current_y = worm.pos
        
        if action == 0:   # up
            new_pos = (current_x, min(current_y + 1, self.world.height - 1))
        elif action == 1: # down
            new_pos = (current_x, max(current_y - 1, 0))
        elif action == 2: # left
            new_pos = (max(current_x - 1, 0), current_y)
        elif action == 3: # right
            new_pos = (min(current_x + 1, self.world.width - 1), current_y)
        else:             # stay
            new_pos = worm.pos
        
        # Only move if new position is empty or same
        if new_pos != worm.pos:
            self.world.grid.move_agent(worm, new_pos)
        
        # Update hunger
        worm.hunger = min(worm.hunger + 0.05, 1.0)
        
        # Calculate reward with gradient shaping
        new_gradient = self.world.gradient[worm.pos[0]][worm.pos[1]]
        new_noxious = self.world.noxious_gradient[worm.pos[0]][worm.pos[1]]
        gradient_reward = (new_gradient - old_gradient) * 10
        noxious_penalty = (new_noxious - old_noxious) * 15  # stronger than food reward
        reward = -0.1 + gradient_reward - noxious_penalty
        
        # Check if worm reached food
        if worm.pos == self.world.food_pos:
            reward += 10.0
            worm.hunger = 0.0
            # Respawn food
            self.world.food_pos = (
                pyrandom.randint(1, self.world.width - 2),
                pyrandom.randint(1, self.world.height - 2)
            )
            self.world._update_gradient()
        
        # Check if starving
        terminated = worm.hunger >= 1.0
        if terminated:
            reward -= 5.0
        
        # Check time limit
        truncated = self.current_step >= self.max_steps
        
        # Get new observation
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, {}
    
    def render(self):
        worm = list(self.world.agents)[0]
        print(f"Step: {self.current_step} | "
            f"Worm pos: {worm.pos} | "
            f"Food pos: {self.world.food_pos} | "
            f"Gradient: {self.world.gradient[worm.pos[0]][worm.pos[1]]:.3f} | "
            f"Hunger: {worm.hunger:.3f}")
        
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    
    print("Creating Gymnasium environment...")
    env = WormGymEnv()
    
    print("Checking environment validity...")
    check_env(env)
    print("Environment check passed!")
    
    print("\nRunning 5 manual steps:")
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        print(f"  Reward: {reward:.2f}")
        if terminated or truncated:
            break