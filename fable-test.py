import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm
import time
import ProblemData
import LcuFunctions
import math

from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation
from QPE import PhaseEstimation
from qiskit import Aer
import fable

# from a vector of counts for each basis vector, return the normalized state representing the amplitudes for each of the basis vectors
def getStateFromCounts(counts_vec):
    norm = math.sqrt(np.sum(counts_vec))
    return np.sqrt(counts_vec) / norm

#sim_path = 'simulations/LCU_8G_diffusion/'
sim_path = 'simulations/test_1G_diffusion/'
input_file = 'input.txt'
data = ProblemData.ProblemData(sim_path + input_file)
A_mat_size = (data.n_x) * (data.n_y) * data.G
A_bits = math.ceil(math.log2(A_mat_size))
block_encode_bits = 2*A_bits + 1
n_eig_eval_bits = 5  # number of bits to represent the final eigenvalue
n_eig_eval_states = int(math.pow(2,n_eig_eval_bits))
A_matrix, B_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
num_iter = 10000000
# A_matrix and B_matrix will not necessarily be Hermitian for all problems, but I think for 1G problems they are
print("A is hermitian: ", ishermitian(A_matrix))
print("B is hermitian: ", ishermitian(B_matrix))

A_eigenvalues, A_eigenvectors = np.linalg.eig(A_matrix)
B_eigenvalues, B_eigenvectors = np.linalg.eig(B_matrix)

A_sing_vals = svdvals(A_matrix)
eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)

eigenvector_input = eigvecs[:,0]
eigenvector_input = eigenvector_input/np.linalg.norm(eigenvector_input) # normalize into quantum state

qc = QuantumCircuit(block_encode_bits, block_encode_bits)

eigvec_input_state = StatePreparation(eigenvector_input)
qc.append(eigvec_input_state,list(range(A_bits)))

fable.fable(A_matrix, circ=qc, epsilon=0)
qc.measure(list(range(block_encode_bits)), list(range(block_encode_bits)))

simulator = Aer.get_backend('qasm_simulator')
circ = transpile(qc, simulator)

# Run and get counts
result = simulator.run(circ, shots = num_iter).result()
counts = result.get_counts(circ)

#test = '{:b}'.format(10).zfill(block_encode_bits)
counts_dict = {'{:b}'.format(i).zfill(A_bits):counts['{:b}'.format(i).zfill(block_encode_bits)] if '{:b}'.format(i).zfill(block_encode_bits) in counts else 0 for i in range(A_mat_size)}
print(counts_dict)
qc.draw('mpl', filename="test_fable.png")

#np.linalg.norm(alpha * N * unitary.data[0:N, 0:N] - A)/np.linalg.norm(A)