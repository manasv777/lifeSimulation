"""
test_constraints.py — isolate whether enforce_constraints() actually works,
with NO training involved. Pure mechanism test.
"""
import numpy as np
import torch
from connectome_network import ConnectomeNetwork

ZERO_TOL = 1e-6

net = ConnectomeNetwork(nociception=False, recurrent=False)

# Snapshot the biological masks (what SHOULD stay zero / keep sign)
gap_mask = net.gap_mask.cpu().numpy()
chem_mask = net.chem_mask.cpu().numpy()
gap_sign = net.gap_sign.cpu().numpy()
chem_sign = net.chem_sign.cpu().numpy()

# 1) Deliberately corrupt the matrices the way training would:
#    - add noise everywhere (fills zeros, flips some signs, breaks symmetry)
with torch.no_grad():
    net.gap_adjacency.data += torch.randn_like(net.gap_adjacency.data) * 0.1
    net.chemical_adjacency.data += torch.randn_like(net.chemical_adjacency.data) * 0.1

# Confirm it's corrupted
gap_after_noise = net.gap_adjacency.data.cpu().numpy()
leaked_before = ((np.abs(gap_after_noise) > ZERO_TOL) & (gap_mask == 0)).sum()
print(f"After corrupting: non-bio gap entries non-zero = {leaked_before}")

# 2) Now enforce constraints and re-check
net.enforce_constraints()

gap = net.gap_adjacency.data.cpu().numpy()
chem = net.chemical_adjacency.data.cpu().numpy()

# Check existence: did any should-be-zero entry survive?
gap_leak = ((np.abs(gap) > ZERO_TOL) & (gap_mask == 0)).sum()
chem_leak = ((np.abs(chem) > ZERO_TOL) & (chem_mask == 0)).sum()
print(f"\nAfter enforce_constraints():")
print(f"  GAP   non-bio entries still non-zero: {gap_leak}  (should be 0)")
print(f"  CHEM  non-bio entries still non-zero: {chem_leak}  (should be 0)")

# Check sign: did any biological entry end up with the wrong sign?
gap_signflip = ((np.sign(gap) != gap_sign) & (gap_mask == 1) & (np.abs(gap) > ZERO_TOL)).sum()
chem_signflip = ((np.sign(chem) != chem_sign) & (chem_mask == 1) & (np.abs(chem) > ZERO_TOL)).sum()
print(f"  GAP   sign-flipped bio entries: {gap_signflip}  (should be 0)")
print(f"  CHEM  sign-flipped bio entries: {chem_signflip}  (should be 0)")

# Check symmetry of gap
sym_err = np.abs(gap - gap.T).max()
print(f"  GAP   max asymmetry: {sym_err:.8f}  (should be ~0)")