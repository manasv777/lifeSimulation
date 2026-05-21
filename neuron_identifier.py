import c302.SpreadsheetDataReader as reader
import xlrd
import os
import json
from collections import defaultdict

xls_path = os.path.join(reader.spreadsheet_location, 'CElegansNeuronTables.xls')
wb = xlrd.open_workbook(xls_path)

# Get all 302 neurons from connectome
connectome_sheet = wb.sheet_by_name('Connectome')
all_neurons = set()
for row in range(1, connectome_sheet.nrows):
    all_neurons.add(connectome_sheet.cell_value(row, 0))
    all_neurons.add(connectome_sheet.cell_value(row, 1))

# Get sensory neurons with their functions
sensory_sheet = wb.sheet_by_name('Sensory')
sensory_neurons = {}
for row in range(1, sensory_sheet.nrows):
    name = sensory_sheet.cell_value(row, 0)
    function = sensory_sheet.cell_value(row, 6)
    neurotransmitter = sensory_sheet.cell_value(row, 4)
    sensory_neurons[name] = {
        'function': function,
        'neurotransmitter': neurotransmitter
    }

# Get motor neurons from muscle connections
muscle_sheet = wb.sheet_by_name('NeuronsToMuscle')
motor_neurons = set()
for row in range(1, muscle_sheet.nrows):
    motor_neurons.add(muscle_sheet.cell_value(row, 0))

# Classify all neurons
interneurons = all_neurons - set(sensory_neurons.keys()) - motor_neurons
# Some neurons are both sensory and motor
sensory_and_motor = set(sensory_neurons.keys()) & motor_neurons

classification = {
    'sensory': sorted(list(sensory_neurons.keys())),
    'motor': sorted(list(motor_neurons)),
    'interneuron': sorted(list(interneurons)),
    'sensory_and_motor': sorted(list(sensory_and_motor))
}

print(f"Sensory neurons: {len(sensory_neurons)}")
print(f"Motor neurons: {len(motor_neurons)}")
print(f"Interneurons: {len(interneurons)}")
print(f"Dual sensory+motor: {len(sensory_and_motor)}")
print(f"Total accounted for: {len(all_neurons)}")

print(f"\nSensory neuron functions:")
functions = defaultdict(list)
for name, info in sensory_neurons.items():
    functions[info['function']].append(name)
for func, neurons in sorted(functions.items()):
    print(f"  {func}: {neurons}")

# Save as your foundation data file
with open('data/neuron_classification.json', 'w') as f:
    json.dump(classification, f, indent=2)

print("\nSaved to data/neuron_classification.json")
print("This file is the input layer definition for your PyTorch network.")