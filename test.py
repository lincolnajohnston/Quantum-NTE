import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cmath
import math
import LcuFunctions
from QPE import PhaseEstimation
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation
from qiskit import Aer, execute
from qiskit_aer.aerprovider import QasmSimulator
from qiskit import transpile
import random

# from a vector of counts for each basis vector, return the normalized state representing the amplitudes for each of the basis vectors
def getStateFromCounts(counts_vec):
    norm = math.sqrt(np.sum(counts_vec))
    return np.sqrt(counts_vec) / norm

A_bits = 4
A_mat_size = int(math.pow(2,A_bits))

# make random Hermitian matrix
random.seed(938619)
np.random.seed(1019368)
A = np.round(np.random.random((A_mat_size, A_mat_size)), decimals=2)
for i in range(A_mat_size):
    for j in range(i+1, A_mat_size):
        A[i,j] = A[j,i]


A_eigenvalues, A_eigenvectors = np.linalg.eig(A)

U = sp.linalg.expm(2j * math.pi * A)
U_eigenvalues, U_eigenvectors = np.linalg.eig(U)
print("U is unitary: ", LcuFunctions.is_unitary(U))
U_control = np.kron(np.array([[1,0],[0,0]]),np.eye(A_mat_size)) + np.kron(np.array([[0,0],[0,1]]),U)

eigenvector_input = U_eigenvectors[:,0] * U_eigenvectors[0,0].real / np.abs(U_eigenvectors[0,0].real)
eigenvalue_expected = U_eigenvalues[0]
expected_phase = math.atan2(eigenvalue_expected.imag, eigenvalue_expected.real) / (2*math.pi)

n_eig_eval_bits = 1
num_iter = 1000000
qc = QuantumCircuit(A_bits + n_eig_eval_bits, A_bits + n_eig_eval_bits)
n_basis_vecs = int(math.pow(2,qc.num_qubits))
eigvec_input_state = StatePreparation(eigenvector_input)

qc.append(eigvec_input_state, list(range(n_eig_eval_bits, n_eig_eval_bits + A_bits)))
#qc.save_statevector()
#qpe = PhaseEstimation(n_eig_eval_bits, U, A_bits, circuit=qc)
qc.h(0)
A_unitary = UnitaryGate(U_control)
qc.append(A_unitary, list(range(n_eig_eval_bits, n_eig_eval_bits + A_bits)) + [0])

qc.save_statevector()

# measure eigenvalue qubits
#qc.measure(list(range(n_eig_eval_bits)), list(range(n_eig_eval_bits)))
#qc.measure([0],[0])
#qc.measure(list(range(n_eig_eval_bits, qc.num_qubits)), list(range(n_eig_eval_bits, qc.num_qubits))) # measure eigenvector qubits

# perform simulation using counts
'''backend = Aer.get_backend('qasm_simulator')
shots = 1000000
results = execute(qc, backend=backend, shots=shots).result()
counts = results.get_counts()

test = '{:b}'.format(i).zfill(qc.num_qubits)
count_array_0 = np.array([counts['{:b}'.format(i).zfill(A_bits).ljust(qc.num_qubits, '0')] if '{:b}'.format(i).zfill(A_bits).ljust(qc.num_qubits,'0') in counts else 0 for i in range(A_mat_size)])
count_array_1 = np.array([counts['{:b}'.format(i).zfill(A_bits).ljust(qc.num_qubits, '1')] if '{:b}'.format(i).zfill(A_bits).ljust(qc.num_qubits,'1') in counts else 0 for i in range(A_mat_size)])

print("eigenvector state: ", eigenvector_input)
print("0 state: ", getStateFromCounts(count_array_0))
print("1 state: ", getStateFromCounts(count_array_1))'''

# run circuit with statevector
backend = QasmSimulator(method="statevector")
new_circuit = transpile(qc, backend)
print("circuit transpiled")
job = backend.run(new_circuit)
job_result = job.result()
state_vec = job_result.get_statevector(qc).data
non_control_state = state_vec[::int(math.pow(2,n_eig_eval_bits))]
control_state = state_vec[1::int(math.pow(2,n_eig_eval_bits))]

non_control_state = non_control_state / np.linalg.norm(non_control_state)
control_state = control_state / np.linalg.norm(control_state)

non_control_state_phase = math.atan2(non_control_state[0].imag, non_control_state[0].real) / (2*math.pi)
non_control_state = non_control_state * cmath.exp(-2j*math.pi*non_control_state_phase)

control_state_phase = math.atan2(control_state[0].imag, control_state[0].real) / (2*math.pi)
control_state = control_state * cmath.exp(-2j*math.pi*control_state_phase)

print("eigenvector_input: ", eigenvector_input)
print("0 control state: ", non_control_state)
print("1 control state: ", control_state)
print("phase imposed: ", control_state_phase)
print("expected phase from fundamental eigenvalue: ", expected_phase)

qc.draw('mpl', filename="test_trash.png")

#test = '{:b}'.format(i).zfill(qc.num_qubits)
test = '{:b}'.format(i).zfill(n_eig_eval_bits).ljust(qc.num_qubits,'0')
#counts_dict = {'{:b}'.format(i).zfill(n_eig_eval_bits):counts['{:b}'.format(i).zfill(qc.num_qubits)] if '{:b}'.format(i).zfill(qc.num_qubits) in counts else 0 for i in range(int(math.pow(2,n_eig_eval_bits)))}
#counts_dict = {'{:b}'.format(i).zfill(n_eig_eval_bits):counts['{:b}'.format(i).zfill(n_eig_eval_bits).ljust(qc.num_qubits,'0')] if '{:b}'.format(i).zfill(n_eig_eval_bits).ljust(qc.num_qubits,'0') in counts else 0 for i in range(int(math.pow(2,A_bits)))}
counts_dict = {'{:b}'.format(i).zfill(n_eig_eval_bits):counts['{:b}'.format(i).zfill(n_eig_eval_bits) + '0001'] if '{:b}'.format(i).zfill(n_eig_eval_bits) + '0001' in counts else 0 for i in range(int(math.pow(2,A_bits)))}

#counts_of_A_matrix = np.array([counts['{:b}'.format(i).zfill(A_bits).ljust(qc.num_qubits,'0')] if '{:b}'.format(i).zfill(A_bits).ljust(qc.num_qubits,'0') in counts else 0 for i in range(A_mat_size)])
counts_of_A_matrix = np.array([counts['{:b}'.format(i).zfill(A_bits)+'0001'] if '{:b}'.format(i).zfill(A_bits)+'0001' in counts else 0 for i in range(A_mat_size)])
predicted_state_from_counts = getStateFromCounts(counts_of_A_matrix)
print("Predicted State From Counts: ", predicted_state_from_counts)
print("Input Eigenvector", eigenvector_input)

print("expected phase: ", expected_phase)
print("counts dictionary: ", counts_dict)

print("a")