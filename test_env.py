import numpy as np
import matplotlib.pyplot as plt
import mesa
import torch
import wandb
import gymnasium as gym
from stable_baselines3 import PPO

print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("gymnasium:", gym.__version__)
print("All imports successful")