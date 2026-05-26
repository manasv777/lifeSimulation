import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from worm_gym_env import WormGymEnv
import random
from connectome_network import ConnectomeNetwork
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from connectome_policy import ConnectomePolicy

class ConnectomeFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim = 64):
        super().__init__(observation_space, features_dim)

        self.connectome = ConnectomeNetwork()
        self.output = nn.Linear(5, features_dim)

    def forward(self, observations):
        action_logits, values = self.connectome(observations)
        return self.output(action_logits)


class SimpleLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
    
    def _on_step(self):
        self.current_episode_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            wandb.log({
                "episode_reward": self.current_episode_reward,
                "episode": len(self.episode_rewards)
            })
            self.current_episode_reward = 0
        return True

run = wandb.init(
    project="c-elegans-devai",
    name="connectome-mlp-nociception",
    config={
        "algorithm": "PPO",
        "policy": "ConnectomeFeatureExtractor",
        "total_timesteps": 100_000,
        "grid_size": 20,
        "max_steps_per_episode": 200,
        "learning_rate": 3e-4,
    }
)
env = WormGymEnv()

model = PPO(
    'MlpPolicy',  # ← direct class, no quotes
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

print("Starting training...")
print("Watch live at wandb.ai\n")

model.learn(
    total_timesteps=100_000,
    callback=SimpleLoggingCallback()
)

model.save("models/worm_ppo_mlp_nociception")
print("\nModel saved!")

test_env = WormGymEnv()
print("\nTesting trained model for 5 episodes:")
total_rewards = []

for episode in range(5):
    obs, _ = test_env.reset()
    episode_reward = 0
    steps = 0
    worm_start = list(test_env.world.agents)[0].pos
    food_location = test_env.world.food_pos
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        episode_reward += reward
        steps += 1
        if terminated or truncated:
            break
    
    total_rewards.append(episode_reward)
    print(f"  Episode {episode+1}: start={worm_start}, food={food_location}, "
          f"{steps} steps, reward: {episode_reward:.2f}")

print(f"\nAverage reward: {sum(total_rewards)/len(total_rewards):.2f}")
print(f"Best episode: {max(total_rewards):.2f}")
print(f"Worst episode: {min(total_rewards):.2f}")

wandb.finish()