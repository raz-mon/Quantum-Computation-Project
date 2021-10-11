import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import process_fidelity
from qiskit.quantum_info import state_fidelity
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
from qiskit.quantum_info import random_statevector
from qiskit.extensions import Initialize
from qiskit import IBMQ, Aer, transpile, assemble, BasicAer
from qiskit.visualization import plot_histogram, plot_bloch_multivector, array_to_latex

# circuit will be our main circuit.
qreg_q = QuantumRegister(7, 'q')
creg_c = ClassicalRegister(7, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)
# We Make qreg_q_2 and circuit2 just to make an operator out of the scrabmling circtuit (takes 3 qubits as arguments).
qreg_q_2 = QuantumRegister(3, 'scramb')
circuit2 = QuantumCircuit(qreg_q_2)
circuit2.rxx(pi/2, qreg_q_2[0], qreg_q_2[1])
circuit2.rxx(pi/2, qreg_q_2[0], qreg_q_2[2])
circuit2.rz(pi/2, qreg_q_2[0])
circuit2.rxx(pi/2, qreg_q_2[1], qreg_q_2[2])
circuit2.rz(pi/2, qreg_q_2[1])
circuit2.rz(pi/2, qreg_q_2[2])
circuit2.rxx(pi/2, qreg_q_2[0], qreg_q_2[1])
circuit2.rxx(pi/2, qreg_q_2[0], qreg_q_2[2])
circuit2.rz(-pi/2, qreg_q_2[0])
circuit2.rxx(pi/2, qreg_q_2[1], qreg_q_2[2])
circuit2.rz(-pi/2, qreg_q_2[1])
circuit2.rz(-pi/2, qreg_q_2[2])

# Generate a gate from the circuit (later can apply it to qubits at will).
scramble_gate = circuit2.to_gate(label='Scrambler')


# Make a circuit that makes a bell pair out of the two given qubits, give it a name.
def create_bell_pair(qc, a, b):
    """Creates a bell pair in qc using qubits a & b"""
    qc.h(a)             # Put qubit a into state |+>
    qc.cx(a,b)          # CNOT with a as control and b as target


# Generate a quantum circuit that pairs two qubits, in an EPR state.
qreg = QuantumRegister(2, 'q')
circ = QuantumCircuit(qreg)
create_bell_pair(circ, qreg[0], qreg[1])
# Generate a gate from the circuit.
bell_gate = circ.to_gate(label = 'bell')


def create_bell_pair(qc, a, b):
    """Creates a bell pair in qc using qubits a & b"""
    qc.h(a) # Put qubit a into state |+>
    qc.cx(a,b) # CNOT with a as control and b as target


# Generate Bell-pairs from the pairs: (1,4), (2,3), (5,6).
circuit.append(bell_gate, [1, 4])
circuit.append(bell_gate, [2, 3])
circuit.append(bell_gate, [5, 6])

# Apply the scrambling unitary to qubits 0-2 and 3-5 (because U is the same as U^* in this case).
circuit.append(scramble_gate, [1, 2, 3])
circuit.append(scramble_gate, [4, 5, 6])

# Generate a random state_vector:
psi = random_statevector(dims=2)
# Make a gate from this operator, names 'init_gate':
init_gate = Initialize(psi)
init_gate.label = 'init_gate'

# Save the inverse of the gate (in order to return later to the state psi - with the 'target'
# qubit of the teleportation).
inverse_init_gate = init_gate.gates_to_uncompute()
inverse_init_gate.label = 'inverse_init_gate'

# Initialize the first qubit to the random state generated (this is the state we wish to teleport to qubit 6).
circuit.append(init_gate, [qreg_q[0]])

# Append to the circuit the inverse of 'init_gate' on qubit 6 (the 'target'):
circuit.append(inverse_init_gate, [qreg_q[6]])

# Measure qubit 6, to classical bit 6:
circuit.measure(qreg_q[6], creg_c[6])

# Simulation (several options):
"""
sim = BasicAer.get_backend('statevector_simulator')
circuit.save_statevector()
out_vector = sim.run(circuit).result().get_statevector()
plot_bloch_multivector(out_vector)
"""

"""
# Another option:
backend = BasicAer.get_backend('statevector_simulator')
job = backend.run(transpile(circuit, backend))
circuit_state = job.result().get_statevector(circuit)
print(circuit_state)
plot_bloch_multivector(circuit_state)
counts = job.result().get_counts()
plot_histogram(counts)
"""

# And another one:
# Transpile for simulator
simulator = Aer.get_backend('aer_simulator')
t_circuit = transpile(circuit, simulator)

# Run and get statevector
result = simulator.run(t_circuit).result()
counts = result.get_counts()
print(counts)
plot_histogram(counts, title='probabilities')
