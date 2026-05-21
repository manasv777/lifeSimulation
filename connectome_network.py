import torch
import torch.nn as nn
import numpy as np
import json
import c302.SpreadsheetDataReader as reader

def build_adjacency_matrix():
    #Load connectome data
    cells, neuron_connections = reader.read_data(include_nonconnected_cells=True)

    #Create neuron index dictionary
    neuron_to_index = {name: idx for idx, name in enumerate(cells)}
    n_neurons = len(cells)           

    #Creates the 302x302 adjacency matrix (empty at beginning)
    matrix = np.zeros((n_neurons, n_neurons))

    # Neurotransmitters that produce inhibitory signals in C. elegans
    INHIBITORY = ['GABA', 'FMRFamide']

    # Fill the adjacency matrix with biologically signed connection weights
    for conn in neuron_connections:
        if conn.pre_cell in neuron_to_index and conn.post_cell in neuron_to_index:
            i = neuron_to_index[conn.pre_cell]
            j = neuron_to_index[conn.post_cell]
            if conn.synclass in INHIBITORY:
                matrix[i][j] -= conn.number  # inhibitory = negative weight
            else:
                matrix[i][j] += conn.number  # excitatory = positive weight

    return matrix, neuron_to_index, cells

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
        
        # Build adjacency matrix from biological data
        matrix, neuron_to_idx, cells = build_adjacency_matrix()
        sensory_idx, motor_idx, interneuron_idx, dual_idx = load_neuron_classification(neuron_to_idx)
        
        # Store neuron indices
        self.sensory_idx = sensory_idx
        self.motor_idx = motor_idx
        self.n_neurons = len(cells)
        
        # Convert adjacency matrix to PyTorch tensor
        # Normalize by max value so weights start between 0 and 1
        matrix_normalized = matrix / matrix.max()
        self.adjacency = nn.Parameter(
            torch.FloatTensor(matrix_normalized),
            requires_grad=True
        )
        
        # Linear layer: motor neurons → 5 actions
        self.output_layer = nn.Linear(len(motor_idx), 5)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Step 1 — create 302-length input vector
        neuron_activations = torch.zeros(batch_size, self.n_neurons)
        
        # Step 2 — inject gradient value into sensory neurons
        gradient_value = x[:, 0]  # first observation value
        for idx in self.sensory_idx:
            neuron_activations[:, idx] = gradient_value
        
        # Step 3 — propagate through connectome matrix
        neuron_activations = torch.tanh(
            neuron_activations @ self.adjacency
        )
        
        # Step 4 — read motor neuron outputs
        motor_outputs = neuron_activations[:, self.motor_idx]
        
        # Step 5 — linear layer to 5 action probabilities
        action_logits = self.output_layer(motor_outputs)
        
        return action_logits

            


if __name__ == "__main__":
    matrix, neuron_to_idx, cells = build_adjacency_matrix()
    print(f"Matrix shape: {matrix.shape}")
    print(f"Total neurons: {len(cells)}")
    print(f"Non-zero connections: {np.count_nonzero(matrix)}")
    print(f"Max connection strength: {matrix.max()}")
    
    sensory_idx, motor_idx, interneuron_idx, dual_idx = load_neuron_classification(neuron_to_idx)
    print(f"\nSensory neurons (inputs): {len(sensory_idx)}")
    print(f"Motor neurons (outputs): {len(motor_idx)}")
    print(f"Interneurons (hidden): {len(interneuron_idx)}")
    print(f"Dual sensory+motor: {len(dual_idx)}")
    
    print("\nBuilding connectome network...")
    network = ConnectomeNetwork()
    
    # Test with a fake observation
    test_obs = torch.FloatTensor([[0.073, 0.1]])  # gradient=0.073, hunger=0.1
    output = network(test_obs)
    print(f"Input shape: {test_obs.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Action logits: {output}")
    print("\nConnectome network working correctly!")
    print(f"Min connection strength: {matrix.min()}")
    print(f"Inhibitory connections: {np.sum(matrix < 0)}")
    print(f"Excitatory connections: {np.sum(matrix > 0)}")
