import numpy as np
import random as pyrandom
from stable_baselines3 import PPO
from worm_gym_env import WormGymEnv
from connectome_policy import ConnectomePolicy

SEEDS = [42, 28, 7, 15, 99, 123, 256, 512, 77, 33]
N_EPISODES_PER_SEED = 20
NOXIOUS_THRESHOLD = 0.4

MODELS = {
    "MLP Nociception Baseline": "models/worm_ppo_nociception_mlp_v4",
    "Connectome Nociception":   "models/worm_ppo_nociception_recurrent_v4",
}


def generate_fixed_episodes(seed, n=N_EPISODES_PER_SEED, grid_size=20):
    pyrandom.seed(seed)
    np.random.seed(seed)
    episodes = []
    for _ in range(n):
        while True:
            start = (pyrandom.randint(0, grid_size-1), pyrandom.randint(0, grid_size-1))
            food  = (pyrandom.randint(1, grid_size-2), pyrandom.randint(1, grid_size-2))
            nox   = (pyrandom.randint(1, grid_size-2), pyrandom.randint(1, grid_size-2))
            if start != food and nox != food and nox != start:
                episodes.append((start, food, nox))
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
    noxious_entries = []

    for start_pos, food_pos, nox_pos in episodes:
        env = WormGymEnv(nociception=True)
        obs, _ = env.reset()
        env.world.grid.move_agent(list(env.world.agents)[0], start_pos)
        env.world.food_pos = food_pos
        env.world.noxious_pos = nox_pos
        env.world._update_gradient()
        env.world._update_noxious_gradient()
        obs = env._get_observation()

        episode_reward = 0
        episode_noxious = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            worm = list(env.world.agents)[0]
            if env.world.noxious_gradient[worm.pos[0]][worm.pos[1]] > NOXIOUS_THRESHOLD:
                episode_noxious += 1

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        noxious_entries.append(episode_noxious)

    # return per-episode means for this seed
    return np.mean(rewards), np.mean(noxious_entries)


def main():
    # seed_results[name] = {"reward": [...per seed...], "noxious": [...per seed...]}
    seed_results = {name: {"reward": [], "noxious": []} for name in MODELS}

    for seed in SEEDS:
        episodes = generate_fixed_episodes(seed)
        for name, path in MODELS.items():
            is_connectome = "Connectome" in name
            result = evaluate_model(
                path, episodes,
                custom_policy=ConnectomePolicy if is_connectome else None
            )
            if result is not None:
                mean_rew, mean_nox = result
                seed_results[name]["reward"].append(mean_rew)
                seed_results[name]["noxious"].append(mean_nox)
        print(f"Seed {seed} done")

    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Mean Rew':>10} {'Rew Std':>9} {'Nox/Ep':>9} {'Nox Std':>9}")
    print("=" * 70)
    for name, d in seed_results.items():
        r = np.array(d["reward"]); n = np.array(d["noxious"])
        print(f"{name:<30} {r.mean():>10.2f} {r.std():>9.2f} {n.mean():>9.2f} {n.std():>9.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()