# Life Simulation: A *C. elegans* Connectome as a Learnable Reinforcement Learning Policy

An independent research project that uses the complete *Caenorhabditis elegans* connectome — the only fully mapped nervous system, comprising 302 neurons and ~3,363 synaptic connections — as the **policy network** of a reinforcement learning agent, trained to perform chemotaxis (navigating toward food along a chemical gradient).

Rather than using the connectome as a fixed template or as a system controlled from the outside, here the worm's own synaptic weights *are* the trainable parameters. The agent learns by reinforcement (reward and punishment), and biological structure is enforced throughout training so the network stays anatomically faithful while its connection strengths adapt.

> **Status:** Research in progress. The chemotaxis results below are complete and written up in a draft paper. A nociception extension and an inheritance/reproduction study are planned as follow-up work.

---

## Key Finding

Across four policy architectures, enforcing the connectome's biological structure during training (which connections exist, their sign, and gap-junction symmetry) **improved** the consistency of learned behavior rather than degrading it — and biologically constrained networks matched or exceeded an unconstrained baseline on mean performance. A recurrent, fully biological variant was the most consistent of all.

In short: biological structure behaved as a *beneficial inductive bias* rather than a handicap. A weight-drift analysis confirmed that training preserved the connectome's wiring exactly while reshaping the strengths of existing synapses.

---

## How It Works

1. **Environment** (`worm_environment.py`, `worm_gym_env.py`) — a 20×20 grid world (built on Mesa, wrapped as a Gymnasium environment) containing a worm and a food source that emits a chemical gradient. The worm senses the local gradient and its hunger; reward is shaped to encourage moving up the gradient toward food.

2. **Connectome network** (`connectome_network.py`) — the connectome is loaded from real connectivity data (OpenWorm's c302) into two 302×302 adjacency matrices: one for chemical synapses (directional, signed by neurotransmitter class) and one for electrical gap junctions (symmetric). Sensory input is injected into sensory neurons, propagated through the network, and read out from motor neurons as action logits and a value estimate.

3. **Biological constraints** (`enforce_constraints` in `connectome_network.py`) — three invariants are re-enforced during training so the network stays biologically valid:
   - **Existence:** connections that don't exist in the real worm stay at zero.
   - **Sign:** excitatory connections stay excitatory, inhibitory stay inhibitory; only magnitude changes.
   - **Symmetry:** gap junctions remain bidirectionally symmetric.

4. **Policy & training** (`connectome_policy.py`, `train_worm.py`) — the agent is trained with PPO (Stable-Baselines3). Four architectures are compared: an MLP baseline, the connectome as a feature extractor, the connectome as the full policy, and a recurrent version of the full-policy connectome.

5. **Evaluation** (`validate_models.py`, `measure_weight_drift.py`) — models are evaluated across multiple random seeds for both mean reward and behavioral consistency, and the weight-drift analysis quantifies how far learning moved the weights from their biological initialization.

---

## Repository Structure

| File | Purpose |
|------|---------|
| `worm_environment.py` | Mesa world: grid, worm agent, food source, chemical gradient |
| `worm_gym_env.py` | Gymnasium wrapper defining observations, actions, and reward |
| `connectome_network.py` | The connectome as a PyTorch network + biological constraint enforcement |
| `connectome_policy.py` | Custom Actor-Critic policy that uses the connectome as its network |
| `train_worm.py` | Training script (PPO) for all four architectures |
| `validate_models.py` | Multi-seed evaluation of mean reward and consistency |
| `measure_weight_drift.py` | Compares trained weights against the biological initialization |
| `test_constraints.py` | Unit-style check that constraint enforcement works in isolation |
| `data/neuron_classification.json` | Sensory / motor / interneuron classification of the 302 neurons |
| `models/` | Saved trained models |

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/manasv777/lifeSimulation.git
cd lifeSimulation

# Set up a virtual environment (Python 3.10)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Train a model
```bash
python3 train_worm.py
```

### Evaluate across seeds
```bash
python3 validate_models.py
```

### Check how far learning moved the biological weights
```bash
python3 measure_weight_drift.py
```

---

## Built With

- **Python 3.10**
- **PyTorch** — connectome network implementation
- **Stable-Baselines3** — PPO reinforcement learning
- **Gymnasium** — RL environment interface
- **Mesa** — agent-based world simulation
- **c302 / OpenWorm** — biological connectome data
- **Weights & Biases** — training run tracking

---

## Acknowledgments

Connectome connectivity data is from the [OpenWorm](http://www.openworm.org/) project's c302 framework, derived from the electron-microscopy reconstructions of White et al. (1986) and Varshney et al. (2011).

---

## Author

**Manas Vellaturi** — University of North Carolina at Chapel Hill
