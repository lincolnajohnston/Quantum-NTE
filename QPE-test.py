import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import time
import ProblemData
import LcuFunctions
import math

from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from QPE import PhaseEstimation
from qiskit import Aer
import fable
########## Comments/Thoughts ##########
# What needs to be done, in approximate order
# -solve coarse matrix, interpolate eigenvector to fine mesh, run the rest of the algorithm on it
# -do better block encoding on B^(1/2)
# -implement the block encoding of A_squiggle in the QPE correctly, apply the exponent of A_squiggle quantumly, not classically

########## Functions ##########

# from a vector of counts for each basis vector, return the normalized state representing the amplitudes for each of the basis vectors
def getStateFromCounts(counts_vec):
    norm = math.sqrt(np.sum(counts_vec))
    return np.sqrt(counts_vec) / norm

########## Read input, set up variables ##########

#sim_path = 'simulations/LCU_8G_diffusion/'
coarse_sim_path = 'simulations/test_1G_diffusion_coarse/'
fine_sim_path = 'simulations/test_1G_diffusion_fine/'
coarse_input_file = 'input.txt'
fine_input_file = 'input.txt'
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
method = "statevector" # either "counts" or "statevector" mode uses different methods of quantum simulation in Qiskit


# A_matrix and B_matrix will not necessarily be Hermitian for all problems, but I think for 1G problems they are
# If hermitian, then QPE will output their eigenvalues, if not, need to make A and B hermitian (using one more qubit)
A_matrix_coarse, B_matrix_coarse = coarse_data.diffusion_construct_L_F_matrices(A_coarse_mat_size)
'''A_eigenvalues_coarse, A_eigenvectors_coarse = np.linalg.eig(A_matrix_coarse)
B_eigenvalues_coarse, B_eigenvectors_coarse = np.linalg.eig(B_matrix_coarse)
print("A is hermitian: ", ishermitian(A_matrix_coarse))
print("B is hermitian: ", ishermitian(B_matrix_coarse))
print("B is positive definite: ", np.all(B_eigenvalues_coarse > 0))'''
A_matrix, B_matrix = fine_data.diffusion_construct_L_F_matrices(A_mat_size)
'''A_eigenvalues, A_eigenvectors = np.linalg.eig(A_matrix)
B_eigenvalues, B_eigenvectors = np.linalg.eig(B_matrix)
print("A is hermitian: ", ishermitian(A_matrix))
print("B is hermitian: ", ishermitian(B_matrix))
print("B is positive definite: ", np.all(B_eigenvalues > 0))'''

# find eigenvector and eigenvalues of the GEP classically (should use power iteration for real problems)
eigvals_coarse, eigvecs_coarse = eigh(A_matrix_coarse, B_matrix_coarse, eigvals_only=False)
eig_index = 0 # index of eigenvector/eigenvalue to use, 0 for fundamental eigenmode

########## Create matrices/vectors needed for the circuit ##########

# eigenvector_input is the state that is fed into the quantum circuit
# TODO: make eigenvector input the interpolation of the coarse mesh eigvec
eigenvector_input = eigvecs_coarse[:,eig_index]
eigenvector_input = eigenvector_input/np.linalg.norm(eigenvector_input) # normalize into quantum state

# Create the sqrtB, sqrtB_inv, and A_squiggle matrices. Cheating here for now by finding them classically.
# TODO: block encode these using the methods from the Changpeng paper
sqrtB = sqrtm(B_matrix)
sqrtB_inv = np.linalg.inv(sqrtB)
A_squiggle = sqrtB_inv @ A_matrix @ sqrtB_inv
A_squiggle_pow = expm(2j*math.pi*A_squiggle)
#A_squiggle_pow_eigvals, A_squiggle_pow_eigvecs = np.linalg.eig(A_squiggle_pow)
print("A_squiggle_pow is unitary: ", LcuFunctions.is_unitary(A_squiggle_pow))

########## Create Quantum Circuit ##########

logn = fable.get_logn(sqrtB)
block_encode_bits = 2*logn+1
qc = QuantumCircuit(block_encode_bits + n_eig_eval_bits, block_encode_bits + n_eig_eval_bits)

# put the quantum circuit in the coarse phi_0 state and do interpolation onto the fine grid
eigvec_input_state = StatePreparation(eigenvector_input)
#eigvec_input_state = StatePreparation(math.pow(1/math.sqrt(2),A_coarse_bits) * np.ones(A_coarse_mat_size))
#eigvec_input_state = StatePreparation(eigvecs_coarse[:,1] / np.linalg.norm(eigvecs_coarse[:,1]))
x_bits_diff = A_x_bits - A_coarse_x_bits
y_bits_diff = A_y_bits - A_coarse_y_bits
G_bits_diff = A_G_bits - A_coarse_G_bits
test = list(range(n_eig_eval_bits + G_bits_diff, n_eig_eval_bits + A_G_bits)) + list(range(n_eig_eval_bits + A_G_bits + y_bits_diff, n_eig_eval_bits + A_G_bits + A_y_bits)) + list(range(n_eig_eval_bits + A_G_bits + A_y_bits + x_bits_diff, n_eig_eval_bits + A_bits))
qc.append(eigvec_input_state, list(range(n_eig_eval_bits + x_bits_diff, n_eig_eval_bits + A_x_bits)) + list(range(n_eig_eval_bits + A_x_bits + y_bits_diff, n_eig_eval_bits + A_x_bits + A_y_bits)) + list(range(n_eig_eval_bits + A_x_bits + A_y_bits + G_bits_diff, n_eig_eval_bits + A_bits)))
for i in range(n_eig_eval_bits, n_eig_eval_bits + x_bits_diff):
    qc.h(i)
for i in range(n_eig_eval_bits + A_x_bits, n_eig_eval_bits + A_x_bits + y_bits_diff):
    qc.h(i)
for i in range(n_eig_eval_bits + A_x_bits + A_y_bits, n_eig_eval_bits + A_x_bits + A_y_bits + G_bits_diff):
    qc.h(i)

input_state = Statevector.from_instruction(qc).data
input_state_collapsed = input_state[:n_eig_eval_states*A_mat_size:n_eig_eval_states]

# block encoding of B^(1/2) so that when the most significant bits (bottom bits aka higher index) are 
# all 0, the state on the less significant (lower index) bits will resemble the state, c * B^(1/2) * phi_0
# TODO: FIX THIS, not done correctly here because I'm being lazy, more efficient way to block encode this from Changpeng paper
qc, alpha_sqrtB = fable.fable(sqrtB, qc, epsilon=0, max_i = qc.num_qubits-1)
state_0 = Statevector.from_instruction(qc).data
state_0_collapsed = state_0[:n_eig_eval_states*A_mat_size:n_eig_eval_states] / np.linalg.norm(state_0[:n_eig_eval_states*A_mat_size:n_eig_eval_states])

# block encoding of B^(-1/2)*A*B^(1/2) so that when the most significant bits (bottom bits aka higher index) are 
# all 0, the state on the less significant (lower index) bits will resemble the state, c * B^(-1/2)*A*B^(1/2) * (arbitrary_phi)
# TODO: FIX THIS, not done correctly here because I'm being lazy, more efficient way to block encode this from Changpeng paper
#A_squiggle_qc, alpha_A_squiggle = fable.fable(A_squiggle, epsilon=0, max_i = qc.num_qubits-1)

# do QPE
# TODO: change this to block encode A_squiggle, convert it to e^(2*pi*i*A_squiggle), then use that as the controlled unitary for QPE
qpe = PhaseEstimation(n_eig_eval_bits, A_squiggle_pow, A_bits, circuit=qc)
state_1 = Statevector.from_instruction(qc).data
state_1_collapsed = state_1[:n_eig_eval_states*A_mat_size] / np.linalg.norm(state_1[:n_eig_eval_states*A_mat_size])

# do B^(-1/2) on the eigenvector state to return it to its original state, the eigenvector of the GEP
#qc, alpha_sqrtB_inv = fable.fable(sqrtB_inv, qc, epsilon=0, max_i = qc.num_qubits-1)
#state_2 = Statevector.from_instruction(qc).data
#state_2_collapsed = state_2[:n_eig_eval_states*A_mat_size] / np.linalg.norm(state_2[:n_eig_eval_states*A_mat_size])

########## Run the circuit, output results ##########
# solve problem classically to compare to quantum results
eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
eigenvector_fine = eigvecs[:,eig_index] / np.linalg.norm(eigvecs[:,eig_index])
#coarse_eigvec_expanded = math.pow((1/math.sqrt(2)), x_bits_diff + y_bits_diff + G_bits_diff) * np.kron(eigenvector_input.reshape(coarse_data.n_x, coarse_data.n_y, coarse_data.G), np.ones((int(math.pow(2,x_bits_diff)), int(math.pow(2,y_bits_diff)), int(math.pow(2,G_bits_diff))))).reshape(A_mat_size)
#state_overlap = np.dot(coarse_eigvec_expanded.conj(), eigenvector_fine)

if method == "statevector":
    qc.save_statevector()

    # Run emulator
    backend = QasmSimulator(method="statevector")
    new_circuit = transpile(qc, backend)
    job = backend.run(new_circuit)
    job_result = job.result()

    # print statevector of non-junk qubits
    state_vec = job_result.get_statevector(qc).data
    eigvec_collapsed = state_vec[0:int(math.pow(2,n_eig_eval_bits+A_bits)):int(math.pow(2,n_eig_eval_bits))]
    state_vec_dict = {'{:b}'.format(i).zfill(qc.num_qubits):round(state_vec[i],5) for i in range(len(state_vec))}
    N_eig_eval = int(math.pow(2,n_eig_eval_bits))
    state_vec_collapsed = np.zeros(N_eig_eval)
    for i in range(N_eig_eval):
        for j in range(A_mat_size):
            state_vec_collapsed[i] += state_vec[j*N_eig_eval + i] * state_vec[j*N_eig_eval + i].conj() # collpase state vec onto eigenvalue evaluation qubits (keeping only the 0 state for the block encode bits)
        state_vec_collapsed[i] = math.sqrt(state_vec_collapsed[i])
    state_vec_collapsed = state_vec_collapsed / np.linalg.norm(state_vec_collapsed)

    state_vec_collapsed_eigvec = np.zeros(A_mat_size)
    for j in range(A_mat_size):
        for i in range(N_eig_eval):
            state_vec_collapsed_eigvec[j] += state_vec[j*N_eig_eval + i] * state_vec[j*N_eig_eval + i].conj() # collpase state vec onto eigenvalue evaluation qubits (keeping only the 0 state for the block encode bits)
        state_vec_collapsed_eigvec[j] = math.sqrt(state_vec_collapsed_eigvec[j])
    state_vec_collapsed_eigvec = state_vec_collapsed_eigvec / np.linalg.norm(state_vec_collapsed_eigvec)
    eigenvector_error = state_vec_collapsed_eigvec - eigenvector_fine

    index_max = max(range(len(state_vec_collapsed)), key=state_vec_collapsed.__getitem__)

    # when the build-in IQFT is used in the QPE script, bit reversal like this needs to be done, but I can't seem to get that working
    #max_index_binary = ('{:b}'.format(index_max).zfill(n_eig_eval_bits))
    #eig_result_i = 0
    #for i in range(n_eig_eval_bits):
    #    eig_result_i += int(max_index_binary[i]) * int(math.pow(2,i))
    print("expected eigenvalue: ", eigvals[eig_index])
    print("eigenvalue found: ", (index_max) / n_eig_eval_states)
    print("probability of getting this eigenvalue on one measurement: ", state_vec_collapsed[index_max] ** 2)
    print("square of inner product of input state and actual fine eigenvector: ", np.inner(input_state_collapsed, eigenvector_fine) ** 2)

elif method == "counts":
    num_iter = 1000 # should be able to only run this once if probablity of success if high and you do some amplitude amplification on the block encode bits
    # measure eigenvalue qubits
    qc.measure(list(range(n_eig_eval_bits)), list(range(n_eig_eval_bits)))

    # measure qubits containing eigenvector, don't run this line except for debugging
    #qc.measure(list(range(n_eig_eval_bits, n_eig_eval_bits + A_bits)), list(range(n_eig_eval_bits, n_eig_eval_bits + A_bits)))

    # measure qubits used for block encoding
    qc.measure(list(range(n_eig_eval_bits + A_bits, n_eig_eval_bits + block_encode_bits)), list(range(n_eig_eval_bits + A_bits, n_eig_eval_bits + block_encode_bits)))

    simulator = Aer.get_backend('qasm_simulator')
    circ = transpile(qc, simulator)

    # Run and get counts
    result = simulator.run(circ, shots = num_iter).result()
    counts = result.get_counts(circ)
    # make new dictionary with counts only from where the block encode bits are all 0
    counts_dict = {'{:b}'.format(i).zfill(n_eig_eval_bits):counts['{:b}'.format(i).zfill(qc.num_qubits)] if '{:b}'.format(i).zfill(qc.num_qubits) in counts else 0 for i in range(n_eig_eval_states)}

    counts_vec = np.array([counts['{:b}'.format(i).zfill(qc.num_qubits)] if '{:b}'.format(i).zfill(qc.num_qubits) in counts else 0 for i in range(n_eig_eval_states)]) # convert dictionary to array in binary order
    index_max = max(range(len(counts_vec)), key=counts_vec.__getitem__) # index of basis vector with highest counts
    #max_index_binary = ('{:b}'.format(index_max).zfill(n_eig_eval_bits)) # convert index_max to binary form

    # reverse order of bits to get actual correct index, eig_result_i
    #eig_result_i = 0
    #for i in range(n_eig_eval_bits):
    #    eig_result_i += int(max_index_binary[i]) * int(math.pow(2,i))

    # print results
    print("eigenvalue found: ", (index_max) / n_eig_eval_states)
    print("expected eigenvalue: ", eigvals[eig_index])

ax = sns.heatmap(np.abs(input_state_collapsed).reshape(fine_data.n_x, fine_data.n_y), linewidth=0.5)
ax.invert_yaxis()
plt.title("Input (Coarse) Solution")
plt.figure()

ax = sns.heatmap(np.abs(eigenvector_fine).reshape(fine_data.n_x, fine_data.n_y), linewidth=0.5)
ax.invert_yaxis()
plt.title("Actual Fine Solution")
plt.figure()

error_vector = np.abs(input_state_collapsed) - np.abs(eigenvector_fine)
ax = sns.heatmap(error_vector.reshape(fine_data.n_x, fine_data.n_y), linewidth=0.5)
ax.invert_yaxis()
plt.title("Error")
plt.show()

# draw circuit, can take a while to run
#qc.draw('mpl', filename="test_block_encoding.png")