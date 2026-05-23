import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from connectome_network import ConnectomeNetwork

class ConnectomePolicy(ActorCriticPolicy):

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
        # Override the action distribution
        self.action_dist = CategoricalDistribution(self.action_space.n)

    def _build(self, lr_schedule):
        # Build connectome network here so parameters exist when optimizer is created
        self.connectome = ConnectomeNetwork()
        
        # Set up optimizer for all parameters
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    def forward(self, obs: torch.Tensor, deterministic: bool = False): 
        #Process observation through connectome
        action_logits, values = self.connectome(obs)

        #Creates the action distribtion from logits
        distribution = self.action_dist.proba_distribution(action_logits = action_logits)

        #Samlpe action (determinstic during testing, stochastic during training)
        action = distribution.get_actions(deterministic=deterministic)

        #Calculate log probability of the action for training
        log_prob = distribution.log_prob(action)

        return action, values, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor): 
        #Process observation through connectome
        action_logits, values = self.connectome(obs)

        #Creates the action distribtion from logits
        distribution = self.action_dist.proba_distribution(action_logits = action_logits)

        #Calculate log probability of the given action
        log_prob = distribution.log_prob(action)

        #Calculate entropy of the distribution
        entropy = distribution.entropy()

        return values, log_prob, entropy
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        # Process observation through connectome
        action_logits, values = self.connectome(obs)
        return values
        
    def _predict(self, observation: torch.Tensor, deterministic: bool = False):
        action_logits, values = self.connectome(observation)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions
    
if __name__ == "__main__":
    import gymnasium as gym
    from stable_baselines3.common.env_util import make_vec_env
    from worm_gym_env import WormGymEnv
    
    print("Testing ConnectomePolicy...")
    
    # Create environment
    env = WormGymEnv()
    
    # Create a simple lr_schedule
    lr_schedule = lambda _: 3e-4
    
    # Build policy
    policy = ConnectomePolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lr_schedule
    )
    
    print("Policy built successfully!")
    
    # Test forward pass
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor([obs])
    
    actions, values, log_prob = policy.forward(obs_tensor)
    print(f"Actions: {actions}")
    print(f"Values: {values}")
    print(f"Log prob: {log_prob}")
    
    print("\nConnectomePolicy working correctly!")