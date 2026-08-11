from qiskit import QuantumCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation, CXGate, XGate, QFT, HGate, RYGate
from qiskit.quantum_info import Statevector
import numpy as np
import math

#Given the ability to apply the 2x2 matrix, B, that is located along the diagonal of a NxN matrix, A,
# with all other values of A being 0, apply the A matrix to the quantum circuit. The B matrix is located
# at position [m:m+2,m:m+2] in the A matrix, where m must be a multiple of 2 (if it is not a multiple of two
# in the current discretization, N can just be doubled to make it divisible by 2)

# Create circuit: 1 ancilla + n data qubits
n = 5
N = int(math.pow(2,n))

# define block_position, then calculate m
#block_position = [1,1] # state of all qubits (from most to least significant) except the least significant one to extract the submatrix, B
#m = int(sum([math.pow(2,n-i-1) * block_position[i] for i in range(n-1)])) # position of start of B_matrix in larger, padded A_matrix

# define m, then calculate block_position
m = 24
block_position = [int(i) for i in bin(int(m/2))[2:].zfill(n-1)]

theta = 0.8
B_mat = np.array([[math.cos(theta/2), -math.sin(theta/2)],[math.sin(theta/2), math.cos(theta/2)]]) # should be able to be any arbitrary ((unitary) matrix
A_mat = np.zeros((N,N))
A_mat[m:m+2,m:m+2] = B_mat
qc = QuantumCircuit(n+1)

# q[0] = ancilla, q[1] = q2, q[2] = q1, q[3] = q0

# Step 1: Initialize state
#initial_state = np.array([1,2,3,4,5,6,7,8])
initial_state = np.random.rand(N)
initial_state = initial_state / np.linalg.norm(initial_state)
eigvec_input_state = StatePreparation(initial_state)
qc.append(eigvec_input_state, list(range(n)))

# Step 3: Apply B to q0
for i,b in enumerate(block_position):
    if b == 0:
        qc.x(n-i-1)

#controlled_B = RYGate(0.8).control(n-1)
controlled_B = UnitaryGate(B_mat).control(n-1)
qc.append(controlled_B, list(range(n-1,-1,-1)))  # controlled-B from ancilla to q0

for i,b in enumerate(block_position):
    if b == 0:
        qc.x(n-i-1)

# Step 4: Make all other states 0
for i,b in enumerate(block_position):
    if b == 0:
        qc.x(n-i-1)

controlled_X = XGate().control(n-1)
qc.append(controlled_X, list(range(1,n+1)))  # controlled-X from ancilla to q0

for i,b in enumerate(block_position):
    if b == 0:
        qc.x(n-i-1)

# Simulate and extract statevector
state = Statevector.from_instruction(qc)

# Extract amplitudes where ancilla = |1⟩ (i.e., successful projection)
post_selected_state = state.data[int(N):]  # indices 8–15: ancilla = 1
expected_state = A_mat @ initial_state
print("actual state: ", post_selected_state)
print("expected state: ", expected_state)

# Show amplitudes for successful post-selection
#print("Post-selected (A|ψ⟩, unnormalized):")
#for i, amp in enumerate(post_selected):
#    if abs(amp) > 1e-6:
#        print(f"|{i:03b}>: {amp}")

qc.draw('mpl', filename="padded-matrix-pic.png")
