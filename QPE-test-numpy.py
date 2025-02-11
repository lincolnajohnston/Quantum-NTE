import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import time
import ProblemData
import LcuFunctions
import math
import cmath

from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator
from QPE import PhaseEstimation
from qiskit import Aer
import fable

# from a vector of counts for each basis vector, return the normalized state representing the amplitudes for each of the basis vectors
def getStateFromCounts(counts_vec):
    norm = math.sqrt(np.sum(counts_vec))
    return np.sqrt(counts_vec) / norm

# make a unitary matrix across n_qubits given a unitary that uses fewer qubits than n_qubits, fill the rest with unitaries
# the order of the qubits the unitary is applied on is new_order
def apply_unitary_to_qubits(n_qubits, new_order, unitary):
    n_unitary_qubits = int(math.log2(len(unitary)))
    identity_2x2 = np.eye(2)
    final_matrix_size = int(math.pow(2,n_qubits))

    expanded_matrix = np.kron(unitary, np.eye(int(math.pow(2,n_qubits-n_unitary_qubits))))
    final_matrix = np.zeros((final_matrix_size, final_matrix_size), dtype=np.complex_)

    # Integrate the unitary on target qubits into the overall system matrix
    # The matrix form for target qubits permutations
    #final_matrix = final_matrix.reshape((2,) * n_qubits * 2)
    for i in range(final_matrix_size):
        binary_i = '{:b}'.format(i).zfill(n_qubits)
        new_binary_i = np.zeros(n_qubits)
        for k in range(n_qubits):
            new_binary_i[new_order[k]]=binary_i[k]
        new_i = int(sum([new_binary_i[k] * math.pow(2,n_qubits - k - 1) for k in range(n_qubits)]))
        for j in range(final_matrix_size):
            binary_j = '{:b}'.format(j).zfill(n_qubits)
            new_binary_j = np.zeros(n_qubits)
            for k in range(n_qubits):
                new_binary_j[new_order[k]]=binary_j[k]
            new_j = int(sum([new_binary_j[k] * math.pow(2,n_qubits - k - 1) for k in range(n_qubits)]))
            final_matrix[new_i][new_j] = expanded_matrix[i][j]


    return final_matrix

# create the QFT unitary matrix then invert it (conjugate transpose it)
def get_IQFT_matrix(n_bits):
    mat_size = int(math.pow(2,n_bits))
    omega = cmath.exp(2j*math.pi/mat_size)
    final_mat = np.ones((mat_size,mat_size), dtype=np.complex_)
    for i in range(mat_size):
        final_mat[:,i] *= omega ** i
    for i in range(mat_size):
        final_mat[i,:]  = np.power(final_mat[i,:],i)
    return (1/math.sqrt(mat_size)) * final_mat.conj()


last_time = time.time()

#sim_path = 'simulations/LCU_8G_diffusion/'
sim_path = 'simulations/test_1G_diffusion/'
input_file = 'input-exact-eigenvalue.txt'
data = ProblemData.ProblemData(sim_path + input_file)
A_mat_size = (data.n_x) * (data.n_y) * data.G
A_bits = math.ceil(math.log2(A_mat_size))
n_eig_eval_bits = 5  # number of bits to represent the final eigenvalue
n_eig_eval_states = int(math.pow(2,n_eig_eval_bits))
A_matrix, B_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
# A_matrix and B_matrix will not necessarily be Hermitian for all problems, but I think for 1G problems they are
print("A is hermitian: ", ishermitian(A_matrix))
print("B is hermitian: ", ishermitian(B_matrix))

A_eigenvalues, A_eigenvectors = np.linalg.eig(A_matrix)
B_eigenvalues, B_eigenvectors = np.linalg.eig(B_matrix)

eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
eig_index = 1 # index of eigenvector/eigenvalue to use, 0 for fundamental eigenmode

print("B is positive definite: ", np.all(B_eigenvalues > 0))
#print(A_matrix)
#print(B_matrix)

# eigenvector corresponding to the smallest eigenvalue of the GEP
eigenvector_input = eigvecs[:,eig_index]
eigenvector_input = eigenvector_input/np.linalg.norm(eigenvector_input) # normalize into quantum state

sqrtB = sqrtm(B_matrix)
sqrtB_inv = np.linalg.inv(sqrtB)
A_squiggle = sqrtB_inv @ A_matrix @ sqrtB_inv
A_squiggle_pow = expm(2j*math.pi*A_squiggle)
A_squiggle_pow_eigvals, A_squiggle_pow_eigvecs = np.linalg.eig(A_squiggle_pow)
print("A_squiggle_pow is unitary: ", LcuFunctions.is_unitary(A_squiggle_pow))
n_bits = A_bits + n_eig_eval_bits
n_states = A_mat_size * n_eig_eval_states

# initialize intial state vector to all zeros in computational basis
state_vec = np.zeros(n_states, dtype=np.complex_)
state_vec[0] = 1

print("setup time: ", time.time() - last_time)
last_time = time.time()

# put the eigenvector register in the phi_0 state
state_prep_unitary = np.zeros((A_mat_size,A_mat_size))
state_prep_unitary[:,0] = eigenvector_input
state_prep_unitary = np.kron(state_prep_unitary, np.eye(n_eig_eval_states))
state_vec = state_prep_unitary @ state_vec

print("state prep time: ", time.time() - last_time)
last_time = time.time()

# apply sqrtB gate, normalize state_vec because sqrtB might not be unitary
state_vec = np.kron(sqrtB,np.eye(n_eig_eval_states)) @ state_vec
state_vec = state_vec / np.linalg.norm(state_vec)

print("sqrtB time: ", time.time() - last_time)
last_time = time.time()

########## do QPE on A_squiggle_pow ##########

# apply the Hadamards
H = 1/math.sqrt(2) * np.array([[1, 1],[1,-1]])
combined_hadamard_gate = np.eye(A_mat_size)
for i in range(n_eig_eval_bits):
    combined_hadamard_gate = np.kron(combined_hadamard_gate, H)
state_vec = combined_hadamard_gate @ state_vec

print("Hadamard time: ", time.time() - last_time)
last_time = time.time()

# apply the controlled unitaries
for c_i in range(n_bits-1,A_bits-1,-1):
    A_squiggle_pow_pow = np.linalg.matrix_power(A_squiggle_pow, int(math.pow(2,n_bits-c_i-1)))
    t0 = np.kron(np.eye(int(math.pow(2,c_i))), np.array([[1,0],[0,0]]))
    t1 = np.kron(A_squiggle_pow_pow, np.kron(np.eye(int(math.pow(2,c_i-A_bits))), np.array([[0,0],[0,1]])))
    full_control_unitary = np.kron(t0 + t1, np.eye(int(math.pow(2,n_bits - c_i - 1))))
    state_vec = full_control_unitary @ state_vec

print("controlled unitary time: ", time.time() - last_time)
last_time = time.time()

########## do IQFT ##########
IQFT_mat = np.kron(np.eye(A_mat_size), get_IQFT_matrix(n_eig_eval_bits))
state_vec = IQFT_mat @ state_vec

print("IQFT time: ", time.time() - last_time)
last_time = time.time()


# do B^(-1/2) on the eigenvector state to return it to its original state
state_vec = np.kron(sqrtB_inv,np.eye(n_eig_eval_states)) @ state_vec
state_vec = state_vec / np.linalg.norm(state_vec)
print("circuits made")

print("ssqrtB_inv time: ", time.time() - last_time)
last_time = time.time()

########## effectively do measurement of the register containing the eigenvalue and find the most likely eigenvalue the state will collapse to ##########
state_vec_collapsed = np.zeros(n_eig_eval_states)
for i in range(n_eig_eval_states):
    for j in range(A_mat_size):
        state_vec_collapsed[i] += state_vec[j*n_eig_eval_states + i] * state_vec[j*n_eig_eval_states + i].conj() # collpase state vec onto eigenvalue evaluation qubits (keeping only the 0 state for the block encode bits)
        #print('{:b}'.format(i).zfill(A_bits), "    ",'{:b}'.format(j).zfill(n_eig_eval_bits), "    ", round(state_vec[i*int(math.pow(2,A_bits)) + j],8))
    state_vec_collapsed[i] = math.sqrt(state_vec_collapsed[i])
index_max = max(range(len(state_vec_collapsed)), key=state_vec_collapsed.__getitem__)
print("eigenvalue found: ", (index_max) / n_eig_eval_states)
print("expected eigenvalue: ", eigvals[eig_index])
print("probability of getting this eigenvalue on one measurement: ", state_vec_collapsed[index_max] ** 2)
