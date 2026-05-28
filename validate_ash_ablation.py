import numpy as np
import random as pyrandom
from stable_baselines3 import PPO
from worm_gym_env import WormGymEnv
from connectome_policy import ConnectomePolicy
from connectome_network import ConnectomeNetwork

SEED = 42
N_EPISODES = 500
NOXIOUS_THRESHOLD = 0.6 

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
            noxious = (pyrandom.randint(1, grid_size-2),
                       pyrandom.randint(1, grid_size-2))
            if start != food and noxious != food and noxious != start:
                episodes.append((start, food, noxious))
                break
    return episodes

def evaluate_model(model_path, episodes, ablate_ash=False, noxious_penalty = True):
    # Load with ConnectomePolicy
    try:
        model = PPO.load(model_path, custom_objects={"policy_class": ConnectomePolicy})
    except Exception as e:
        print(f"  Could not load {model_path}: {e}")
        return None

    # Apply ASH ablation if requested
    if ablate_ash:
        model.policy.connectome.ablate_ash = True

    # Debug — verify forward pass differs between conditions
    import torch
    test_obs = torch.FloatTensor([[0.5, 0.1, 0.8]])
    with torch.no_grad():
        logits, vals = model.policy.connectome(test_obs)
    print(f"  DEBUG: ablate_ash = {model.policy.connectome.ablate_ash}")
    print(f"  DEBUG: logits = {logits}")

    rewards = []
    noxious_entries = []

    for start_pos, food_pos, noxious_pos in episodes:
        env = WormGymEnv(nociception=True, noxious_penalty=noxious_penalty)
        obs, _ = env.reset()
        env.world.grid.move_agent(list(env.world.agents)[0], start_pos)
        env.world.food_pos = food_pos
        env.world.noxious_pos = noxious_pos
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
        if len(rewards) < 5:
            worm = list(env.world.agents)[0]
            print(f"  Episode {len(rewards)+1}: worm={start_pos}, food={food_pos}, noxious={noxious_pos}")

    return rewards, noxious_entries

def main():
    print("ASH Ablation Experiment")
    print("Mirrors laser ablation studies: Bargmann et al. 1990, Hilliard et al. 2002")
    print("=" * 65)

    episodes = generate_fixed_episodes()
    print(f"Generated {N_EPISODES} episodes with seed {SEED}\n")

    model_path = "models/worm_ppo_nociception_final"

    print("Condition 1: ASH Intact")
    result_intact = evaluate_model(model_path, episodes, ablate_ash=False, noxious_penalty = False)

    print("Condition 2: ASH Ablated")
    result_ablated = evaluate_model(model_path, episodes, ablate_ash=True, noxious_penalty = False)

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