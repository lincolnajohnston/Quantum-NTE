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
from scipy.linalg import eigh
import random
import ProblemData

# from a vector of counts for each basis vector, return the normalized state representing the amplitudes for each of the basis vectors
def getStateFromCounts(counts_vec):
    norm = math.sqrt(np.sum(counts_vec))
    return np.sqrt(counts_vec) / norm

#sim_path = 'simulations/LCU_8G_diffusion/'
coarse_sim_path = 'simulations/test_1G_diffusion_coarse/'
fine_sim_path = 'simulations/test_1G_diffusion_fine/'
coarse_input_file = 'input-exact-eigenvalue.txt'
fine_input_file = 'input-exact-eigenvalue.txt'
coarse_data = ProblemData.ProblemData(coarse_sim_path + coarse_input_file)
fine_data = ProblemData.ProblemData(fine_sim_path + fine_input_file)
A_coarse_mat_size = (coarse_data.n_x) * (coarse_data.n_y) * coarse_data.G
A_coarse_x_bits = math.ceil(math.log2(coarse_data.n_x))
A_coarse_y_bits = math.ceil(math.log2(coarse_data.n_y))
A_coarse_G_bits = math.ceil(math.log2(coarse_data.G))
A_coarse_bits = A_coarse_x_bits + A_coarse_y_bits + A_coarse_G_bits
A_mat_size = (fine_data.n_x) * (fine_data.n_y) * fine_data.G
A_x_bits = math.ceil(math.log2(fine_data.n_x))
A_y_bits = math.ceil(math.log2(fine_data.n_y))
A_G_bits = math.ceil(math.log2(fine_data.G))
A_bits = A_x_bits + A_y_bits + A_G_bits
interpolation_bits = A_bits - A_coarse_bits
n_eig_eval_bits = 5  # number of bits to represent the final eigenvalue
n_eig_eval_states = int(math.pow(2,n_eig_eval_bits))

x_bits_diff = A_x_bits - A_coarse_x_bits
y_bits_diff = A_y_bits - A_coarse_y_bits
G_bits_diff = A_G_bits - A_coarse_G_bits


# A_matrix and B_matrix will not necessarily be Hermitian for all problems, but I think for 1G problems they are
# If hermitian, then QPE will output their eigenvalues, if not, need to make A and B hermitian (using one more qubit)
A_matrix_coarse, B_matrix_coarse = coarse_data.diffusion_construct_L_F_matrices(A_coarse_mat_size)
A_matrix, B_matrix = fine_data.diffusion_construct_L_F_matrices(A_mat_size)

# find eigenvector and eigenvalues of the GEP classically (should use power iteration for real problems)
eigvals_coarse, eigvecs_coarse = eigh(A_matrix_coarse, B_matrix_coarse, eigvals_only=False)
eig_index = 0 # index of eigenvector/eigenvalue to use, 0 for fundamental eigenmode

########## Create matrices/vectors needed for the circuit ##########

# eigenvector_input is the state that is fed into the quantum circuit
# TODO: make eigenvector input the interpolation of the coarse mesh eigvec
eigenvector_input = eigvecs_coarse[:,eig_index]
eigenvector_input = eigenvector_input/np.linalg.norm(eigenvector_input) # normalize into quantum state


########## Run the circuit, output results ##########
# solve problem classically to compare to quantum results
eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
eigenvector_fine = eigvecs[:,eig_index] / np.linalg.norm(eigvecs[:,eig_index])
coarse_eigvec_expanded = math.pow((1/math.sqrt(2)), x_bits_diff + y_bits_diff + G_bits_diff) * np.kron(eigenvector_input.reshape(coarse_data.n_x, coarse_data.n_y, coarse_data.G), np.ones((int(math.pow(2,x_bits_diff)), int(math.pow(2,y_bits_diff)), int(math.pow(2,G_bits_diff))))).reshape(A_mat_size)
state_overlap = np.dot(coarse_eigvec_expanded.conj(), eigenvector_fine)
print("projected probability of success: ", state_overlap ** 2)