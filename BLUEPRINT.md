# Research Blueprint — *C. elegans* Connectome Reinforcement Learning

A dated research log. Entries are kept as written at the time, including hypotheses that
were later revised; where a later result overturned or corrected an earlier entry, a dated
correction note is appended rather than editing the original. This preserves the log as an
honest record of how the project actually unfolded.

---

## Day 1

**What is the specific gap my research fills?**

My research fills two specific gaps. The first is combining reinforcement learning with a
biologically faithful connectome architecture — letting the worm's behavior emerge through
experience rather than through pre-programmed weights. The second, deeper gap is what happens
across generations: when two trained agents reproduce, do their offspring exhibit novel
behaviors that neither parent was explicitly trained on? This question of hereditary weight
transfer and multi-generational behavioral emergence has not been studied using a biological
connectome as the substrate. That is the headline contribution I am ultimately working toward.

> *Correction (post–Day 11): The reproduction/inheritance study became the focus of planned
> follow-up work (Paper 2) rather than the first paper. Paper 1 focuses on the first gap —
> the connectome as a learnable RL policy — which proved to be a complete contribution on its own.*

---

## Day 2

**What is each tool I installed actually for in this project?**

- **Python** — to code and connect all the software.
- **Mesa** — to create the environment for the worm, food, and so on.
- **PyTorch** — to build the connectome-based neural network, where each node represents a real
  *C. elegans* neuron and each edge a real synaptic connection. The weights on those edges are
  what get learned through training.
- **Gymnasium** — to connect the Mesa world to the worm agent through a standard RL interface.
- **Stable-Baselines3** — to implement PPO (Proximal Policy Optimization), the RL algorithm that
  adjusts the network weights based on rewards and punishments over thousands of training episodes.
- **Weights & Biases** — to log and visualize results across training runs: reward curves, episode
  lengths, and hyperparameters, so I can compare runs and see whether the agent is actually learning.
- **NumPy** — for numerical calculations.
- **Matplotlib** — to create visualizations.

---

## Day 3

**What are the three neuron types in *C. elegans*, what does each do, and how do they map to my simulation?**

**Sensory neurons** — detect the environment and pass information forward; they don't process it.
For example, ASEL and ASER detect salt concentration, and ASH neurons detect harmful stimuli. In my
simulation these become the input layer, receiving the chemical gradient value and other environmental
signals.

**Interneurons** — the "processor" neurons. They receive signals from sensory neurons, process them, and
pass decisions to the motor neurons. The most important are AVAL and AVAR; anything related to movement
routes through them, and they have the highest connectivity of any neurons in the connectome (AVAL had
342 incoming and 256 outgoing). These are the hidden layer of my network.

**Motor neurons** — receive signals from interneurons and drive muscle movement. DA, DB, VA, and VB control
body-wall muscles for forward and backward movement; DD and VD are inhibitory and coordinate timing. These
are the output layer of my network.

**What did I discover about the connectome today that surprised me?**

I was surprised that the connectome of such a simple organism is so dense — there are so many connections
between neurons. I was also struck by how these connections can be represented as a neural network of layers
and nodes. That was the goal, but it was interesting to reason through how it would actually lay out.

**Why am I using c302's SpreadsheetDataReader instead of PyOpenWorm directly?**

Because PyOpenWorm was not cooperating. c302's SpreadsheetDataReader still gives me the necessary
per-neuron connectivity information, which is sufficient for this level of the project.

---

## Day 4

**What is the observation — what does the worm receive as input each timestep?**

Each timestep the worm receives the chemical gradient at its current position (a number between 0 and 1
indicating how close it is to food) and its current hunger level (between 0 and 1, where 1 means starvation).
Together, the gradient tells the worm where the food is and hunger tells it how urgently it needs to find it.

**What is the action space?**

Five possible actions each timestep: move up, down, left, right, or stay in place. Staying is included
because sometimes the optimal behavior is to wait or reassess rather than always move.

**What is the reward function — when does the worm get rewarded or punished, and by how much?**

Three components. First, +10 when the worm reaches the food — the primary signal that drives learning.
Second, −0.1 every timestep regardless of outcome — this encourages finding food efficiently rather than
wandering forever. Third, −5.0 if hunger reaches 1.0 (starvation) — this creates urgency. When food is
eaten, hunger resets to 0.0 and food respawns.

Today I completed the Gymnasium wrapper connecting the Mesa world to PPO. The environment check passed.
The wrapper exposes an observation space of shape (2,) — gradient value and hunger — an action space of
5 discrete movements, and the full step/reset/render interface. The worm is ready to train. Next session:
connect PPO and run the first training.

---

## Day 5

First successful training run. The worm learned chemotaxis — navigating from random starting positions to
food by following the chemical gradient. Tested from 5 different starting positions, including (18,3) to
food at (10,10) in 20 steps. This is biologically valid gradient-following behavior. The baseline MLP policy
is validated. Next: replace the MLP with the connectome-based PyTorch network.

---

## Day 6

Built the connectome-based PyTorch network: 302 neurons, biologically seeded weights. Signal flows from the
86 sensory neurons through interneurons to the 127 motor neurons via the real adjacency matrix. A linear
output layer translates motor-neuron activations into 5 action probabilities (pick the highest). The network
is verified — correct input/output shapes, gradient tracking confirmed. Tomorrow: connect this network to the
PPO pipeline and compare against the MLP baseline.

> *Correction (later): the total synaptic connection count is 3,363, not the figure I used in some early
> notes. The 302-neuron count is correct.*

---

## Day 7

**Notes — running the ablation study.**

Planned improvement: initialize adjacency-matrix weights with biological sign — excitatory neurotransmitters
(glutamate, acetylcholine) get positive weights, inhibitory (GABA) get negative weights, and gap junctions
stay positive. This makes the initial weight matrix biologically faithful beyond just connection strength,
also encoding signal direction.

- INHIBITORY = [GABA, FMRFamide]
- EXCITATORY = everything else

**Summary:** Today I replaced the MLP policy's network with the connectome network and trained it. The first
time, all weights started between 0 and 1; the second time, inhibitory chemical neurotransmitters started with
negative weights. Both connectome models showed superior *starting* performance versus the MLP baseline
(−6.x vs −14 at iteration 1), showing that biological weight initialization provides a measurable head start.
Planned next step: represent chemical and electrical connections separately (possibly two adjacency matrices —
one for gap junctions, one for chemical) to model their signaling independently and see whether it brings the
model closer to or past the MLP.

---

## Day 8

**Hypothesis:** the two-matrix implementation will show better initial performance than the single signed
matrix, due to separate electrical and chemical signal processing.

**Result after implementing two matrices (early single-run figures):**

- MLP (arbitrary architecture, random start): −7.33 after full training
- Connectome, unsigned: −8.47
- Connectome, signed: −7.96
- Connectome, two-matrix: −7.47

Each successive increase in biological fidelity — unsigned to signed weights, single to dual signal channels —
produced a measurable improvement, approaching the unconstrained baseline while keeping biological
interpretability. Four-architecture comparison underway. Ready for the Option B fully custom policy and
statistical validation.

**Update:** Added the fully custom policy and ran statistical validation. Connectome accuracy turned out to
matter more than having a fully custom policy for good results.

> *Correction (post–Day 11): these are early single-run numbers under a fixed-position setup. They are
> superseded by the final 10-seed validation in Table 1, which uses fully randomized positions and biological
> constraints. Treat the Day 8 figures as historical, not final.*

---

## Day 9

Implemented the recurrent network and ran models through an early 20-episode, seed-42 validation. At that
stage Recurrent Option B reached mean −5.52, roughly equivalent to the MLP baseline −5.46, and I described two
useful model families: consistency-optimized (signed/two-matrix) and performance-optimized (recurrent). Full
ablation study complete; next is nociception.

> *Correction (post–constraints): the single-seed std figures recorded here (e.g. recurrent "std 5.30") were
> an artifact of single-seed evaluation and a fixed food position. Under the final 10-seed protocol with
> randomized positions and biological constraints, the recurrent connectome is the MOST consistent model
> (std 0.61), not a high-variance one. See final Table 1.*

---

## Day 10

**Hypothesis:** the nociception model will show worse mean reward than chemotaxis-only models due to increased
task complexity (balancing food approach against noxious avoidance).

The hypothesis was wrong, and I think the reason is that the worm now has a second incoming signal to learn
from, which can actually make it more effective because there is more structure to exploit. Finished an early
nociception model and validated it against an MLP baseline; at that stage it showed a ~50% reduction in noxious
zone entries vs the MLP (0.15 vs 0.30 per episode), and I made an early "Table 2."

Early biological-validation notes (chemotaxis consistent with the pirouette model, Pierce-Shimomura et al. 1999;
nociception via ASH injection; low variance in connectome models matching reliable real-worm behavior). I also
noted that, unlike biological ASH ablation which eliminates avoidance entirely, computational ablation of the
ASH pathways did not significantly reduce avoidance in the trained agent — suggesting PPO agents develop
distributed avoidance strategies through reward shaping, distinct from circuit-dependent biological nociception.

> *Correction (post–Day 11): the ~50% nociception result did NOT survive re-testing under biological constraints,
> randomized noxious positions, and the 10-seed protocol — the connectome and MLP became statistically
> indistinguishable on noxious avoidance. Nociception, the ASH ablation, and the Bargmann/WormBook nociception
> citations were therefore CUT from Paper 1 and moved to planned Paper 2 work. The distributed-vs-circuit
> observation remains an interesting lead for that follow-up, not a result of Paper 1.*

---

## Day 11

ASH ablation experiment attempted across multiple architectures and settings — confirmed what looked like a
fundamental limitation of RL reward shaping versus biological circuit dependency (ablating ASH did not remove
avoidance). At the time I planned to write this up as a Discussion point. Abstract drafted. Also identified what
I thought was the "true" ablation test: remove the noxious penalty from the reward during validation so the only
remaining avoidance signal is the learned ASH circuit.

> *Correction (post–Day 11): even with the penalty removed and a wider noxious gradient, ablation produced no
> significant behavioral difference. The honest conclusion is that, in this setup, the trained agent's avoidance
> is not ASH-circuit-dependent. This is a genuine but negative result, and the reason nociception was deferred
> to Paper 2 rather than reported in Paper 1.*

---

## Day 12 — Biological constraints (the turning point for Paper 1)

Discovered, via a weight-drift analysis, that the unconstrained connectome models were not staying biologically
faithful during training: PPO was filling in tens of thousands of connections that don't exist in the real worm
(~35% of all non-existent connections became active), and the trained matrices correlated only weakly with the
biological connectome. Realized that a zero in the adjacency matrix is a biological fact (no synapse), not a free
parameter.

Designed and implemented three biological constraints, re-enforced throughout training:
1. **Existence** — connections that start at zero stay at zero (no inventing synapses).
2. **Sign** — excitatory stays excitatory, inhibitory stays inhibitory; only magnitude changes.
3. **Symmetry** — gap junctions stay bidirectionally symmetric (averaged with their transpose).

Hit and fixed a real bug: constraints were initially enforced in the wrong place (during the rollout step rather
than after PPO's gradient updates), so the saved model was still corrupted. Fixed by enforcing at the start of
each rollout phase (so the agent always *acts* on a legal network) plus once more before saving. Verified the fix
two ways: an isolation test that corrupts every entry and confirms enforcement zeros all ~90,000 illegal entries
with no sign or symmetry violations, and a post-training drift check confirming ZERO non-biological connections in
the trained model.

Also made the environment fully reproducible by giving each world its own seeded RNG (so food respawns are
deterministic per episode), and retrained all architectures with fully randomized worm and food positions.

**Final result (official Table 1, 10 seeds × 20 episodes, biologically constrained, randomized positions):**

- MLP Baseline: mean −6.35, std 1.10
- Connectome (Opt A): mean −5.87, std 0.94
- Connectome (Opt B): mean −5.48, std 0.85
- Recurrent Connectome: mean −5.61, std 0.61

All three connectome architectures exceed the MLP on mean reward, and all show lower variance. The recurrent
connectome is the most consistent of all (std 0.61, roughly half the baseline's). The weight-drift analysis shows
structure preserved exactly while connection strengths moved substantially — meaning the value of the connectome
here is its topology and sign structure, not the specific anatomical strengths.

**Interpretation:** biological structure acts as a beneficial inductive bias (a form of regularization), improving
both consistency and mean rather than trading one for the other. This became the thesis of Paper 1.

---

## Day 13 — Paper draft and outreach

Wrote Paper 1 end to end (abstract, introduction, methods, results, discussion, conclusion, references), scoped to
chemotaxis only, with nociception explicitly deferred to future work. Verified every reference against primary
sources; cut three that no longer anchored to claims in the paper (Bargmann 1990, Hilliard 2002, Porta-de-la-Riva
2012) and added Lechner, Grosu & Hasani (2017) as the closest related work, with a gap paragraph distinguishing it.

Sent the draft to a professor and PhD student in UNC Department of Computer Science. Response: the "it works"
result is interesting, but the paper would be much stronger as an **interpretability** study — identifying *which*
component of the biological structure (sparsity, sign, topology, symmetry) is responsible for the advantage.

**Decision:** pursue the interpretability work (component-wise structural ablations) to build a stronger paper,
targeting ALIFE 2027 full paper as the anchor deadline, with the journal *Artificial Life* as a possible upgrade if
results are strong. ALIFE 2026 late-breaking (poster) deadline (~July 20) was missed for the full track but remains
a fallback.

**Parked for Paper 2:** the nociception investigation (why the ASH circuit doesn't drive avoidance in this setup),
and the original headline idea — hereditary weight transfer / multi-generational behavioral emergence between
reproducing agents.

**Next:** design the component-wise structural ablation experiments (isolating sparsity vs sign vs topology vs
symmetry) to answer the "which component is responsible" question.
