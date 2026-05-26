import numpy as np
import random as pyrandom
from stable_baselines3 import PPO
from worm_gym_env import WormGymEnv
from connectome_policy import ConnectomePolicy

SEED = 42
N_EPISODES = 20
NOXIOUS_THRESHOLD = 0.3

MODELS = {
    "MLP Nociception Baseline": "models/worm_ppo_mlp_nociception",
    "Connectome Nociception":   "models/worm_ppo_nociception",
}

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

    rewards = []
    noxious_entries = []

    for start_pos, food_pos in episodes:
        env = WormGymEnv()

        obs, _ = env.reset()
        env.world.grid.move_agent(list(env.world.agents)[0], start_pos)
        env.world.food_pos = food_pos
        env.world._update_gradient()
        obs = env._get_observation()

        episode_reward = 0
        episode_noxious = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            # Count noxious zone entries
            worm = list(env.world.agents)[0]
            if env.world.noxious_gradient[worm.pos[0]][worm.pos[1]] > NOXIOUS_THRESHOLD:
                episode_noxious += 1

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        noxious_entries.append(episode_noxious)

    return rewards, noxious_entries

def main():
    print("Generating fixed test episodes...")
    episodes = generate_fixed_episodes()
    print(f"Generated {len(episodes)} episodes with seed {SEED}")
    print(f"Noxious zone position: (5, 5)")
    print(f"First 5 starting positions: {[e[0] for e in episodes[:5]]}\n")

    results = {}

    for name, path in MODELS.items():
        print(f"Evaluating: {name}")
        is_connectome = "Connectome" in name
        result = evaluate_model(
            path,
            episodes,
            custom_policy=ConnectomePolicy if is_connectome else None
        )

        if result:
            rewards, noxious = result
            results[name] = (rewards, noxious)
            print(f"  Mean reward:          {np.mean(rewards):.2f}")
            print(f"  Std reward:           {np.std(rewards):.2f}")
            print(f"  Mean noxious entries: {np.mean(noxious):.2f}")
            print(f"  Total noxious entries:{np.sum(noxious)}")
        print()

    print("=" * 70)
    print(f"{'Model':<30} {'Mean Rew':>10} {'Std':>8} {'Nox/Ep':>10}")
    print("=" * 70)
    for name, (rewards, noxious) in results.items():
        print(f"{name:<30} {np.mean(rewards):>10.2f} "
              f"{np.std(rewards):>8.2f} "
              f"{np.mean(noxious):>10.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()