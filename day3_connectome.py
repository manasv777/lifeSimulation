import c302 
import c302.SpreadsheetDataReader as reader

# Load the connectome data
cells, neuron_connections = reader.read_data(include_nonconnected_cells=True)

print(f"Total neurons loaded: {len(cells)}")
print(f"Total connections loaded: {len(neuron_connections)}")

# Look at what information is stored about each connection
print("\nLet's look at one connection in detail:")
conn = neuron_connections[0]
print(f"  Type of object: {type(conn)}")
print(f"  Full details: {conn}")
print(f"  From neuron: {conn.pre_cell}")
print(f"  To neuron: {conn.post_cell}")
print(f"  Connection type: {conn.syntype}")
print(f"  Number of connections: {conn.number}")
print(f"  Neurotransmitter: {conn.synclass}")