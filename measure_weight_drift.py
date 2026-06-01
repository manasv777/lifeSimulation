"""
measure_weight_drift.py

Compares a trained ConnectomeNetwork's adjacency matrices against a fresh
biological initialization to quantify how far PPO moved the weights from
their biological starting point.

Reports:
  1. Mean absolute weight change (overall)
  2. Mean absolute change on BIOLOGICAL connections (initially non-zero)
  3. How many NON-biological connections (initially zero) became non-zero
  4. Correlation between trained and biological weights (did structure persist?)
"""

import numpy as np
import torch
from stable_baselines3 import PPO
from connectome_network import ConnectomeNetwork, build_adjacency_matrices
from connectome_policy import ConnectomePolicy

# ── Which trained model to analyze ─────────────────────────────
MODEL_PATH = "models/worm_ppo_opt_b_v4"
ZERO_TOL = 1e-3  # below this, treat a weight as effectively zero


def get_trained_matrices(model_path):
    """Load a trained Option B / recurrent model and pull its two matrices."""
    model = PPO.load(model_path, custom_objects={"policy_class": ConnectomePolicy})
    net = model.policy.connectome
    gap = net.gap_adjacency.detach().cpu().numpy()
    chem = net.chemical_adjacency.detach().cpu().numpy()
    return gap, chem


def get_biological_matrices():
    """Build a fresh biological initialization, normalized the same way the
    network does it in __init__, so the comparison is apples-to-apples."""
    gap_matrix, chemical_matrix, _, _ = build_adjacency_matrices()
    gap_norm = gap_matrix / gap_matrix.max()
    chem_norm = chemical_matrix / np.abs(chemical_matrix).max()
    return gap_norm, chem_norm


def analyze(name, trained, bio):
    print(f"\n=== {name} ===")

    diff = trained - bio
    abs_diff = np.abs(diff)

    # 1. Overall mean absolute change
    print(f"Mean abs change (all entries):        {abs_diff.mean():.6f}")
    print(f"Max abs change (single entry):        {abs_diff.max():.6f}")

    # 2. Change on biological (initially non-zero) connections
    bio_mask = np.abs(bio) > ZERO_TOL
    n_bio = bio_mask.sum()
    if n_bio > 0:
        mean_change_bio = abs_diff[bio_mask].mean()
        mean_mag_bio = np.abs(bio[bio_mask]).mean()
        print(f"Biological connections:               {n_bio}")
        print(f"  Mean abs change on them:            {mean_change_bio:.6f}")
        print(f"  Mean original magnitude:            {mean_mag_bio:.6f}")
        print(f"  Relative change:                    {100*mean_change_bio/mean_mag_bio:.1f}%")

    # 3. Non-biological (initially zero) connections that became non-zero
    zero_mask = ~bio_mask
    became_nonzero = (np.abs(trained[zero_mask]) > ZERO_TOL).sum()
    total_zero = zero_mask.sum()
    print(f"Non-biological (initially zero) entries: {total_zero}")
    print(f"  Became non-zero after training:       {became_nonzero} "
          f"({100*became_nonzero/total_zero:.2f}%)")
    if became_nonzero > 0:
        leaked = np.abs(trained[zero_mask])
        leaked = leaked[leaked > ZERO_TOL]
        print(f"  Mean magnitude of new connections:    {leaked.mean():.6f}")
        print(f"  (compare to biological mean above)")

    # 4. Did the overall structure persist? Correlation of flattened matrices.
    corr = np.corrcoef(trained.flatten(), bio.flatten())[0, 1]
    print(f"Correlation (trained vs biological):  {corr:.4f}")


def main():
    print(f"Analyzing model: {MODEL_PATH}")
    trained_gap, trained_chem = get_trained_matrices(MODEL_PATH)
    bio_gap, bio_chem = get_biological_matrices()

    analyze("GAP JUNCTIONS", trained_gap, bio_gap)
    analyze("CHEMICAL SYNAPSES", trained_chem, bio_chem)

    print("\n" + "=" * 55)
    print("Interpretation guide:")
    print("- High correlation (>0.9) = biological structure mostly preserved")
    print("- Low relative change = training only fine-tuned the biology")
    print("- Many new non-zero entries = training added non-biological")
    print("  connections (structure NOT preserved)")
    print("=" * 55)


if __name__ == "__main__":
    main()