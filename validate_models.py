import numpy as np
import random as pyrandom
from stable_baselines3 import PPO
from worm_gym_env import WormGymEnv
from connectome_policy import ConnectomePolicy

SEEDS = [42, 28, 7, 15, 99, 123, 256, 512, 77, 33]
N_EPISODES_PER_SEED = 20
MODELS = {
    "MLP Baseline":     "models/worm_ppo_baseline_v3",
    "Connectome Opt A": "models/worm_ppo_connectome_signed_v3",
    "Connectome Opt B": "models/worm_ppo_option_b_v3",
    "Recurrent Opt B":  "models/worm_ppo_recurrent_v3",
}

def generate_fixed_episodes(seed, n=N_EPISODES_PER_SEED, grid_size=20):
    pyrandom.seed(seed)
    np.random.seed(seed)
    episodes = []
    for _ in range(n):
        while True:
            start = (pyrandom.randint(0, grid_size-1), pyrandom.randint(0, grid_size-1))
            food  = (pyrandom.randint(1, grid_size-2), pyrandom.randint(1, grid_size-2))
            if start != food:
                episodes.append((start, food))
                break
    return episodes

def evaluate_model(model_path, episodes, custom_policy=None):
    try:
        if custom_policy:
            model = PPO.load(model_path, custom_objects={"policy_class": custom_policy})
        else:
            model = PPO.load(model_path)
    except Exception as e:
        print(f"  Could not load {model_path}: {e}")
        return None

    rewards = []
    for start_pos, food_pos in episodes:
        env = WormGymEnv()
        obs, _ = env.reset()
        env.world.grid.move_agent(list(env.world.agents)[0], start_pos)
        env.world.food_pos = food_pos
        env.world._update_gradient()
        obs = env._get_observation()

        episode_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)

    return np.mean(rewards)

def main():
    # seed_means[model_name] = list of per-seed mean rewards
    seed_means = {name: [] for name in MODELS}

    for seed in SEEDS:
        print(f"\nSeed {seed}")
        episodes = generate_fixed_episodes(seed)
        for name, path in MODELS.items():
            is_option_b = "Opt B" in name
            mean_reward = evaluate_model(
                path, episodes,
                custom_policy=ConnectomePolicy if is_option_b else None
            )
            if mean_reward is not None:
                seed_means[name].append(mean_reward)
                print(f"  {name}: {mean_reward:.2f}")

    print("\n" + "=" * 65)
    print(f"{'Model':<28} {'Mean':>8} {'Std':>8} {'Best':>8} {'Worst':>8}")
    print("=" * 65)
    for name, means in seed_means.items():
        print(f"{name:<28} {np.mean(means):>8.2f} "
              f"{np.std(means):>8.2f} "
              f"{np.max(means):>8.2f} "
              f"{np.min(means):>8.2f}")
    print("=" * 65)

if __name__ == "__main__":
    main()