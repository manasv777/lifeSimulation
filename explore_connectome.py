import c302
import c302.SpreadsheetDataReader as reader
from collections import defaultdict

cells, neuron_connections = reader.read_data(include_nonconnected_cells=True)

print(f"Total neurons: {len(cells)}")
print(f"Total synaptic connections: {len(neuron_connections)}")

# Connection types
gap_junctions = [c for c in neuron_connections if c.syntype == 'GapJunction']
chemical_synapses = [c for c in neuron_connections if c.syntype == 'Send']

print(f"\nGap junctions: {len(gap_junctions)}")
print(f"Chemical synapses: {len(chemical_synapses)}")

# Synapse classes (neurotransmitter types)
synclasses = set(c.synclass for c in neuron_connections)
print(f"\nSynapse classes present:")
for sc in sorted(synclasses):
    print(f"  {sc}")

# Connection weight range
weights = [c.number for c in neuron_connections]
print(f"\nSynapse count range: {min(weights)} to {max(weights)}")
print(f"Average synapses per connection: {sum(weights)/len(weights):.2f}")

# Most connected neurons by outgoing synapses
out_connections = defaultdict(int)
for conn in neuron_connections:
    out_connections[conn.pre_cell] += conn.number

top_10 = sorted(out_connections.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"\nTop 10 most connected neurons (outgoing synapses):")
for neuron, count in top_10:
    print(f"  {neuron}: {count}")
# Read neuron types from the spreadsheet
neuron_info = reader.read_neuron_table()

print("Neuron table sample:")
for name, info in list(neuron_info.items())[:5]:
    print(f"  {name}: {info}")

print(f"\nTotal neurons with type info: {len(neuron_info)}")

# Count neuron types
type_counts = defaultdict(int)
for name, info in neuron_info.items():
    type_counts[info.get('type', 'unknown')] += 1

print("\nNeuron type breakdown:")
for ntype, count in sorted(type_counts.items()):
    print(f"  {ntype}: {count}")
# Most connected neurons by incoming synapses
in_connections = defaultdict(int)
for conn in neuron_connections:
    in_connections[conn.post_cell] += conn.number

top_10_in = sorted(in_connections.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"\nTop 10 most connected neurons (incoming synapses):")
for neuron, count in top_10_in:
    print(f"  {neuron}: {count}")

# Neurons with no outgoing connections (pure sensory/motor endpoints)
all_pre = set(c.pre_cell for c in neuron_connections)
all_post = set(c.post_cell for c in neuron_connections)
only_receive = all_post - all_pre
only_send = all_pre - all_post
print(f"\nNeurons that only receive signals: {len(only_receive)}")
print(f"Neurons that only send signals: {len(only_send)}")