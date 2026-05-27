import numpy as np
import random as pyrandom
from stable_baselines3 import PPO
from worm_gym_env import WormGymEnv
from connectome_policy import ConnectomePolicy
from connectome_network import ConnectomeNetwork

SEED = 42
N_EPISODES = 1000
NOXIOUS_THRESHOLD = 0.3

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

def evaluate_model(model_path, episodes, ablate_ash=False):
    # Load with ConnectomePolicy
    try:
        model = PPO.load(model_path, custom_objects={"policy_class": ConnectomePolicy})
    except Exception as e:
        print(f"  Could not load {model_path}: {e}")
        return None

    # Apply ASH ablation if requested
    if ablate_ash:
        model.policy.connectome.ablate_ash = True
        print(f"  DEBUG: ablate_ash flag = {model.policy.connectome.ablate_ash}")
        print(f"  DEBUG: nociception flag = {model.policy.connectome.nociception}")
        
        # Verify ablation was applied
        ash_idx = model.policy.connectome.ash_idx[0]
        gap_sum = model.policy.connectome.gap_adjacency.data[ash_idx, :].sum()
        chem_sum = model.policy.connectome.chemical_adjacency.data[ash_idx, :].sum()
        print(f"  DEBUG: ASH row sum after ablation — gap: {gap_sum:.4f}, chem: {chem_sum:.4f}")
        print(f"  DEBUG: Should both be 0.0000")

    rewards = []
    noxious_entries = []

    for start_pos, food_pos in episodes:
        env = WormGymEnv(nociception=True)

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

            worm = list(env.world.agents)[0]
            if env.world.noxious_gradient[worm.pos[0]][worm.pos[1]] > NOXIOUS_THRESHOLD:
                episode_noxious += 1

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        noxious_entries.append(episode_noxious)

    return rewards, noxious_entries

def main():
    print("ASH Ablation Experiment")
    print("Mirrors laser ablation studies: Bargmann et al. 1990, Hilliard et al. 2002")
    print("=" * 65)

    episodes = generate_fixed_episodes()
    print(f"Generated {N_EPISODES} episodes with seed {SEED}\n")

    model_path = "models/worm_ppo_nociception_ash"

    print("Condition 1: ASH Intact")
    result_intact = evaluate_model(model_path, episodes, ablate_ash=False)

    print("Condition 2: ASH Ablated")
    result_ablated = evaluate_model(model_path, episodes, ablate_ash=True)

    if result_intact and result_ablated:
        rewards_i, noxious_i = result_intact
        rewards_a, noxious_a = result_ablated

        print("\n" + "=" * 65)
        print(f"{'Condition':<25} {'Mean Rew':>10} {'Nox/Ep':>10} {'Total Nox':>10}")
        print("=" * 65)
        print(f"{'ASH Intact':<25} {np.mean(rewards_i):>10.2f} "
              f"{np.mean(noxious_i):>10.2f} "
              f"{np.sum(noxious_i):>10}")
        print(f"{'ASH Ablated':<25} {np.mean(rewards_a):>10.2f} "
              f"{np.mean(noxious_a):>10.2f} "
              f"{np.sum(noxious_a):>10}")
        print("=" * 65)

        increase = np.mean(noxious_a) - np.mean(noxious_i)
        print(f"\nNoxious entries increased by {increase:.2f} per episode after ASH ablation")
        if increase > 0:
            print("✓ Result consistent with biological ASH ablation studies")
        else:
            print("✗ Unexpected result — ASH ablation did not increase noxious entries")

if __name__ == "__main__":
    main()