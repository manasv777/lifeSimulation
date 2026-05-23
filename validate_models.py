import numpy as np
import random as pyrandom
from stable_baselines3 import PPO
from worm_gym_env import WormGymEnv
from connectome_policy import ConnectomePolicy

# ── Configuration ──────────────────────────────────────────────
SEED = 42
N_EPISODES = 100
MODELS = {
    "MLP Baseline":              "models/worm_ppo_baseline",
    "Connectome Signed (Opt A)": "models/worm_ppo_connectome_signed_v2",
    "Two-Matrix (Opt A)":        "models/worm_ppo_connectome_two_matrix_v2",
    "Two-Matrix + Option B":     "models/worm_ppo_two_matrix_option_b_v2",
}

# ── Generate fixed test episodes ───────────────────────────────
pyrandom.seed(SEED)
np.random.seed(SEED)

def generate_fixed_episodes(n=N_EPISODES, grid_size=20):
    episodes = []
    for _ in range(n):
        while True:
            start = (pyrandom.randint(0, grid_size-1),
                     pyrandom.randint(0, grid_size-1))
            food  = (pyrandom.randint(1, grid_size-2),
                     pyrandom.randint(1, grid_size-2))
            if start != food:
                episodes.append((start, food))
                break
    return episodes

# ── Evaluate one model ─────────────────────────────────────────
def evaluate_model(model_path, episodes, custom_policy=None):
    try:
        if custom_policy:
            model = PPO.load(model_path, custom_objects={"policy_class": custom_policy})
        else:
            model = PPO.load(model_path)
    except Exception as e:
        print(f"  Could not load {model_path}: {e}")
        return None

    print(f"  Loaded: {model_path}")
    print(f"  Policy type: {type(model.policy)}")

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

    return rewards

# ── Run validation ─────────────────────────────────────────────
def main():
    print("Generating fixed test episodes...")
    episodes = generate_fixed_episodes()
    print(f"Generated {len(episodes)} episodes with seed {SEED}")
    print(f"First 3 episodes: {episodes[:3]}\n")

    results = {}

    for name, path in MODELS.items():
        print(f"Evaluating: {name}")
        is_option_b = "Option B" in name
        rewards = evaluate_model(
            path,
            episodes,
            custom_policy=ConnectomePolicy if is_option_b else None
        )

        if rewards:
            results[name] = rewards
            mean  = np.mean(rewards)
            std   = np.std(rewards)
            best  = np.max(rewards)
            worst = np.min(rewards)
            print(f"  Mean:  {mean:.2f}")
            print(f"  Std:   {std:.2f}")
            print(f"  Best:  {best:.2f}")
            print(f"  Worst: {worst:.2f}")
        print()

    print("=" * 65)
    print(f"{'Model':<28} {'Mean':>8} {'Std':>8} {'Best':>8} {'Worst':>8}")
    print("=" * 65)
    for name, rewards in results.items():
        print(f"{name:<28} {np.mean(rewards):>8.2f} "
              f"{np.std(rewards):>8.2f} "
              f"{np.max(rewards):>8.2f} "
              f"{np.min(rewards):>8.2f}")
    print("=" * 65)

if __name__ == "__main__":
    main()