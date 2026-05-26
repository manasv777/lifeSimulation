import torch
import torch.nn as nn
import numpy as np
import json
import c302.SpreadsheetDataReader as reader

def build_adjacency_matrices():
    #Load connectome data
    cells, neuron_connections = reader.read_data(include_nonconnected_cells=True)

    #Create neuron index dictionary
    neuron_to_idx = {name: idx for idx, name in enumerate(cells)}
    n_neurons = len(cells)           

    #Creates the 302x302 adjacency matrix for both types of neurons (empty at beginning)
    gap_matrix = np.zeros((n_neurons, n_neurons))  # matrix for gap connections
    chemical_matrix = np.zeros((n_neurons, n_neurons))  # matrix for chemical connections

    # Neurotransmitters that produce inhibitory signals in C. elegans
    INHIBITORY = ['GABA', 'FMRFamide']

    # Fill the adjacency matrix with biologically signed connection weights
    for conn in neuron_connections:
        if conn.pre_cell in neuron_to_idx and conn.post_cell in neuron_to_idx:
            i = neuron_to_idx[conn.pre_cell]
            j = neuron_to_idx[conn.post_cell]

            if conn.syntype == 'GapJunction':
                gap_matrix[i][j] += conn.number
                gap_matrix[j][i] += conn.number 
            else: #chemical synapse
                if conn.synclass in INHIBITORY:
                    chemical_matrix[i][j] -= conn.number  # inhibitory = negative weight
                else:
                    chemical_matrix[i][j] += conn.number  # excitatory = positive weight

    return gap_matrix, chemical_matrix, neuron_to_idx, cells

def load_neuron_classification(neuron_to_idx):
    with open('data/neuron_classification.json', 'r') as f:
        classification = json.load(f)
    
    sensory_idx = [neuron_to_idx[n] for n in classification['sensory'] if n in neuron_to_idx]
    motor_idx = [neuron_to_idx[n] for n in classification['motor'] if n in neuron_to_idx]
    interneuron_idx = [neuron_to_idx[n] for n in classification['interneuron'] if n in neuron_to_idx]
    dual_idx = [neuron_to_idx[n] for n in classification['sensory_and_motor'] if n in neuron_to_idx]

    return sensory_idx, motor_idx, interneuron_idx, dual_idx

class ConnectomeNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Build two adjacency matrices from biological data
        gap_matrix, chemical_matrix, neuron_to_idx, cells = build_adjacency_matrices()
        sensory_idx, motor_idx, interneuron_idx, dual_idx = load_neuron_classification(neuron_to_idx)
        
        # Store neuron indices
        self.sensory_idx = sensory_idx
        self.motor_idx = motor_idx
        self.n_neurons = len(cells)
        self.previous_state = None

        self.value_layer = nn.Linear(len(motor_idx), 1)  # Critic head for value estimation
        
        # Normalize and store gap junction matrix
        gap_normalized = gap_matrix / gap_matrix.max()
        self.gap_adjacency = nn.Parameter(
            torch.FloatTensor(gap_normalized),
            requires_grad=True
        )
        
        # Normalize and store chemical synapse matrix
        chem_max = np.abs(chemical_matrix).max()
        chem_normalized = chemical_matrix / chem_max
        self.chemical_adjacency = nn.Parameter(
            torch.FloatTensor(chem_normalized),
            requires_grad=True
        )
        
        # Linear layer: motor neurons → 5 actions
        self.output_layer = nn.Linear(len(motor_idx), 5)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Step 1 — create 302-length input vector, starting with previous activations or zeros
        if self.previous_state is None or self.previous_state.shape[0] != batch_size:
            neuron_activations = torch.zeros(batch_size, self.n_neurons)
        else:
            neuron_activations = self.previous_state.detach()
        
        # Step 2 — inject gradient value into sensory neurons
        gradient_value = x[:, 0]  # first observation value
        for idx in self.sensory_idx:
            neuron_activations[:, idx] = gradient_value
        
        # Step 3 — propagate through both matrices seperately
        electrical_signal = neuron_activations @ self.gap_adjacency
        chemical_signal = neuron_activations @ self.chemical_adjacency

        # Step 4 - combine the signals and apply activation
        neuron_activations = torch.tanh(electrical_signal + chemical_signal)
        
        # Step 5 — read motor neuron outputs
        motor_outputs = neuron_activations[:, self.motor_idx]
        
        # Step 6 — linear layer to 5 action probabilities
        action_logits = self.output_layer(motor_outputs)
        
        values = self.value_layer(motor_outputs)

        self.previous_state = neuron_activations
        return action_logits, values

            


if __name__ == "__main__":
    # Test two matrix system
    gap_matrix, chemical_matrix, neuron_to_idx, cells = build_adjacency_matrices()
    
    print(f"Total neurons: {len(cells)}")
    
    print(f"\nGap junction matrix:")
    print(f"  Non-zero connections: {np.count_nonzero(gap_matrix)}")
    print(f"  Max strength: {gap_matrix.max()}")
    print(f"  Symmetric check: {np.allclose(gap_matrix, gap_matrix.T)}")
    
    print(f"\nChemical synapse matrix:")
    print(f"  Non-zero connections: {np.count_nonzero(chemical_matrix)}")
    print(f"  Max strength: {chemical_matrix.max()}")
    print(f"  Min strength: {chemical_matrix.min()}")
    print(f"  Excitatory: {np.sum(chemical_matrix > 0)}")
    print(f"  Inhibitory: {np.sum(chemical_matrix < 0)}")
    
    sensory_idx, motor_idx, interneuron_idx, dual_idx = load_neuron_classification(neuron_to_idx)
    print(f"\nSensory neurons (inputs): {len(sensory_idx)}")
    print(f"Motor neurons (outputs): {len(motor_idx)}")
    print(f"Interneurons (hidden): {len(interneuron_idx)}")
    print(f"Dual sensory+motor: {len(dual_idx)}")
    
    print("\nBuilding connectome network...")
    network = ConnectomeNetwork()
    
    test_obs = torch.FloatTensor([[0.073, 0.1]])
    action_logits, values = network(test_obs)
    print(f"Input shape: {test_obs.shape}")
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Action logits: {action_logits}")
    print(f"Values: {values}")
    print("\nConnectome network working correctly!")
