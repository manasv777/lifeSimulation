Day 1 Entry

What is the specific gap my research fills?
My research has two specific gaps it fills. The first gap is combining reinforcement learning with a biologically faithful connectome architecture — letting the worm's behavior emerge through experience rather than pre-programmed weights. The second and deeper gap is what happens across generations: when two trained agents reproduce, do their offspring exhibit novel behaviors that neither parent was explicitly trained on? This question of hereditary weight transfer and multi-generational behavioral emergence has never been studied using a biological connectome as the substrate. That is the headline contribution of this research.

Day 2 Entry

What is each tool I installed actually for in this project?
Python - to code and connect all the softwares
Mesa - to create the environment for the worm, food, etc. 
PyTorch - to build the connectome-based neural network where each node represents a real C. elegans neuron and each edge represents a real synaptic connection. The weights on those edges are what get learned through training."
Gymnasium - to connect Mesa to the worm agent 
Stable-baselines3 - to implement the PPO (Proximal Policy Optimization) algorithm that trains my agent. PPO is the specific RL algorithm that adjusts the network weights based on rewards and punishments over thousands of training episodes
Weights & Biases -  to log and visualize experiment results across training runs. It records reward curves, episode lengths, and hyperparameters so I can compare different training runs and see whether the agent is actually learning over time
NumPy - to make calculations
MatPlotLib - to create visuals

Day 3 Entry

What are the three neuron types in C. elegans, what does each do, and how do they map to my simulation?

Sensory neurons - these detect the environment and pass the information forward, they don't process or do anything. For example: ASEL, ASER detect salt concentration, ASH neurons detect harmful stimuli. In my simulation, these become the input layer of the network. They recieve chemcial gradient value and other environmental signals.

Interneurons - These are the "processor" neurons. They receive signals form the sensory neurons, process them, and pass the decisions onto the moter neurons. The most important ones in the worm are AVAL and AVAR. Anything that is related to movement routes through those two neurons. They have the highest connectivity of any neurons in the entire connectome - AVAL had 342 incoming and 256 outgoing. These are the hidden layer of my network

Motor neurons - These receive signals from interneurons and drive muscle movmenet. DA, DB, VA, VB control the body wall muscles for forward and backward movement. DD and VD are inhibitory, they coordinate timing. These neurons will be the output layer of my network.

What did I discover about the connectome today that surprised me?
I was surprised that the connectome for such a simple organism is so dense. There are so many connections between each neuron. I also was surprised how you can represent these connections through a vast neural network of layers and nodes. That was my goal, but it was interesting to see and think logically how it would lay out. 

Why am I using c302's SpreadsheetDataReader instead of PyOpenWorm directly?
I'm using c302's SpreadsheetDataReader because PyOpenWorm was not cooperating. I can still get the necessary information I want for each neuron in the organism, and that is sufficient enough for this level. 

Day 4

What is the observation — what information does the worm receive as input each timestep?
Each timestep, the worm will recieve chemcial gradient at the current position (which is a number between 0 and 1 to indicate how close it is to food) and its current hunger level (between 0 and 1, 1 means starvation). Both combined tell the worm where the food is, and hunger tells the worm how urgently it needs to find it. 

What is the action space — what are the possible outputs the worm can choose from?
The worm has five possible actions each timestep: move up, move down, move left, move right, or stay in place. Staying is included because sometimes the optimal behavior involves waiting or reassessing rather than always moving.

What is the reward function — exactly when does the worm get rewarded and punished and by how much?
The reward function has three components. First, a reward of +10 when the worm reaches the food position — this is the primary signal that drives learning. Second, a penalty of -0.1 every single timestep regardless of what happens — this encourages the worm to find food efficiently rather than wandering forever. Third, a penalty of -5.0 if hunger reaches 1.0 meaning starvation — this creates urgency. When food is eaten, hunger resets to 0.0 and food respawns at a new location.

Today I completed the Gymnasium wrapper connecting the Mesa world to PPO. Environment check passed. The wrapper correctly exposes observation space of shape (2,) containing gradient value and hunger level, action space of 5 discrete movements, and the full step/reset/render interface. The worm is ready to be trained. Next session: connect PPO and run first training.

Day 5 Complete
First successful training run. Worm learned chemotaxis — navigating from random starting positions to food by following chemical gradient. Tested from 5 different starting positions including (18,3) to food at (10,10) — 20 steps. This is biologically valid gradient following behavior. Baseline MLP policy validated. Next step: replace MLP with connectome-based PyTorch network.

Day 6
Built connectome-based PyTorch network. 302 neurons, 3,119 connections, biologically seeded weights. Signal flows from 86 sensory neurons through interneurons to 127 motor neurons via real adjacency matrix. Linear output layer translates motor neuron activations to 5 action probabilities, pick the highest probability. Network verified — correct input/output shapes, gradient tracking confirmed. Tomorrow: connect this network to PPO training pipeline and compare against MLP baseline.

Day 7

Notes
Running Ablation Study 
Planned improvement: initialize adjacency matrix weights with biological sign — excitatory neurotransmitters (Glutamate, Acetylcholine) get positive weights, inhibitory (GABA) get negative weights, GapJunctions stay positive. This makes initial weight matrix biologically faithful beyond just strength — also encoding signal direction.
INHIBITORY = ['GABA', 'FMRFamide']
EXCITATORY = everything else

Summary: Today, I replaced the MLP Policy's neural network with connectome neuron network and trained the model. The first time, I had all weights start between 0 and 1 for every connection, and the second time, I made inhibitory chemical neurotransmitters start with negative weights. Both connectome models showed superior starting performance compared to the MLP baseline (-6.x vs -14 at iteration 1), demonstrating that biological weight initialization provides a measurable head start. The MLP's final test average of -7.33 reflects 100,000 steps of learning from a random start, while the signed connectome model reached -7.96 starting from a biologically informed position. Planned improvement to do tomorrow is to change how the chemical and electrical connections are represented in the adjacency matrix (might have to change the system to use seperate adjacency matrices for gap junction (electrical) and chemical) to model their singaling independelty and see if that brings the model closer or better than the MLP model's test average.