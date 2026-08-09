import sys
import os
sys.path.append(os.getcwd())
import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import math
import cmath

from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation, CXGate, XGate, QFT, HGate, RYGate, U1Gate
from qiskit.quantum_info import Statevector
from QPE import PhaseEstimation
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Operator
import fable

def get_eigenvalues(N, K_min, K_max, x_range):
    h = x_range / N
    return np.diag([1/(h*h) * (2 - 2 * math.cos(math.pi * k / N)) for k in range(K_min, K_max + 1)])

# return True if matrix is unitary, False otherwise, O(len(matrix)^2)
def is_unitary(matrix):
    I = matrix.dot(np.conj(matrix).T)
    return I.shape[0] == I.shape[1] and np.allclose(I, np.eye(I.shape[0]))

# applies the Lambda matrix with the eigenvalues on the diagonal, this is different than the O_D oracle!
def apply_eigenvalue_matrix(qc, n, eig_gates, ancilla_gate):
    qc.h(ancilla_gate)

    # E_N^(plus)
    for i in range(n):
        bit_index = eig_gates[n - i - 1]
        plus_rotation_gate = U1Gate(math.pi / (math.pow(2,i+1))).control(1, ctrl_state = '0')
        qc.append(plus_rotation_gate, [ancilla_gate, bit_index])

    # E_N^(minus)
    for i in range(n):
        bit_index = n - i - 1
        minus_rotation_gate = U1Gate(-math.pi / (math.pow(2,i+1))).control(1)
        qc.append(minus_rotation_gate, [ancilla_gate, bit_index])

    qc.h(ancilla_gate)

# applies the matrix with cosines on the diagonal, this is different than the O_D oracle!
def apply_cosine_matrix(qc, n, eig_gates, ancilla_gate):
    qc.h(ancilla_gate)

    # E_N^(plus)
    for i in range(n+1):
        bit_index = eig_gates[n - i]
        plus_rotation_gate = U1Gate(math.pi / (math.pow(2,i))).control(1, ctrl_state = '0')
        qc.append(plus_rotation_gate, [ancilla_gate, bit_index])

    # E_N^(minus)
    for i in range(n+1):
        bit_index = eig_gates[n - i]
        minus_rotation_gate = U1Gate(-math.pi / (math.pow(2,i))).control(1)
        qc.append(minus_rotation_gate, [ancilla_gate, bit_index])

    qc.h(ancilla_gate)

# applies the Lambda matrix with the eigenvalues on the diagonal, this is different than the O_D oracle!
def apply_cosine_eigenvalue_matrix(qc, n, eig_gates, ancilla_gates):
    ag1 = ancilla_gates[0]
    ag2 = ancilla_gates[1]
    #qc.h(ag2)
    #qc.z(ag2)
    qc.x(ag2)

    control_H = HGate().control(1)
    qc.append(control_H,[ag2, ag1])

    # E_N^(plus)
    for i in range(n+1):
        bit_index = eig_gates[n - i]
        plus_rotation_gate = U1Gate(math.pi / (math.pow(2,i))).control(2, ctrl_state = '10')
        qc.append(plus_rotation_gate, ancilla_gates + [bit_index])

    # E_N^(minus)
    for i in range(n+1):
        bit_index = eig_gates[n - i]
        minus_rotation_gate = U1Gate(-math.pi / (math.pow(2,i))).control(2)
        qc.append(minus_rotation_gate, ancilla_gates + [bit_index])

    qc.append(control_H,[ag2, ag1])

    #qc.z(ag2)
    #qc.h(ag2)
    qc.x(ag2)

    qc.x(ag1)
    qc.z(ag1)
    qc.x(ag1)
    qc.z(ag1)


n_x = 3
N_x = int(math.pow(2,n_x))
input_file = 'input.txt'
x_range = 4


qc = QuantumCircuit(n_x + 3)

# put the quantum circuit in the coarse phi_0 state and do interpolation onto the fine grid
b_vector = np.zeros(N_x)
# set to 0 state
b_vector[0] = 1

b_vector /= np.linalg.norm(b_vector)

# b vector state preparation
eigvec_input_state = StatePreparation(b_vector)
qc.append(eigvec_input_state, list(range(n_x)))

apply_cosine_eigenvalue_matrix(qc, n_x, list(range(n_x+1)), [n_x+1, n_x+2])

# find the circuit unitary
circOp = Operator.from_circuit(qc)
circuit_unitary = circOp.data

qc.save_statevector()

# Run emulator
backend = QasmSimulator(method="statevector")
new_circuit = transpile(qc, backend)
print(dict(new_circuit.count_ops())) # print the counts of each type of gate
job = backend.run(new_circuit)
job_result = job.result()

# print statevector of non-junk qubits
state_vec = job_result.get_statevector(qc).data

qc.draw('mpl', filename="diagonal_eigenvalue_matrix_test.png")
