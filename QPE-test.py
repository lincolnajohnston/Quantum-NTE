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
n_eig_eval_bits = 7  # number of bits to represent the final eigenvalue
n_eig_eval_states = int(math.pow(2,n_eig_eval_bits))
A_matrix, B_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
# A_matrix and B_matrix will not necessarily be Hermitian for all problems, but I think for 1G problems they are
print("A is hermitian: ", ishermitian(A_matrix))
print("B is hermitian: ", ishermitian(B_matrix))

A_eigenvalues, A_eigenvectors = np.linalg.eig(A_matrix)
B_eigenvalues, B_eigenvectors = np.linalg.eig(B_matrix)

A_sing_vals = svdvals(A_matrix)
eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
eig_index = 0 # index of eigenvector/eigenvalue to use, 0 for fundamental eigenmode

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
logn = fable.get_logn(sqrtB)
block_encode_bits = 2*logn+1
qc = QuantumCircuit(block_encode_bits + n_eig_eval_bits, block_encode_bits + n_eig_eval_bits)

# put the quantum circuit in the phi_0 state
eigvec_input_state = StatePreparation(eigenvector_input)
qc.append(eigvec_input_state, list(range(n_eig_eval_bits, n_eig_eval_bits + A_bits)))

#A_times_eigenvector = A_matrix @ (eigenvector_input) / A_mat_size
num_iter = 1000000

# block encoding of B^(1/2) so that when the most significant bits (bottom bits aka higher index) are 
# all 0, the state on the less significant (lower index) bits will resemble the state, c * B^(1/2) * phi_0
# TODO: FIX THIS, not done correctly here because I'm being lazy, more efficient way to block encode this from Changpeng paper
qc, alpha_sqrtB = fable.fable(sqrtB, qc, epsilon=0, max_i = qc.num_qubits-1)

# block encoding of B^(-1/2)*A*B^(1/2) so that when the most significant bits (bottom bits aka higher index) are 
# all 0, the state on the less significant (lower index) bits will resemble the state, c * B^(-1/2)*A*B^(1/2) * (arbitrary_phi)
# TODO: FIX THIS, not done correctly here because I'm being lazy, more efficient way to block encode this from Changpeng paper
#A_squiggle_qc, alpha_A_squiggle = fable.fable(A_squiggle, epsilon=0, max_i = qc.num_qubits-1)

# do QPE
qpe = PhaseEstimation(n_eig_eval_bits, A_squiggle_pow, A_bits, circuit=qc)

# do B^(-1/2) on the eigenvector state to return it to its original state
#qc, alpha_sqrtB_inv = fable.fable(sqrtB_inv, qc, epsilon=0, max_i = qc.num_qubits-1)
print("circuits made")

method = "statevector"
if method == "statevector":
    qc.save_statevector()

    # Run emulator
    backend = QasmSimulator(method="statevector")
    new_circuit = transpile(qc, backend)
    print("circuit transpiled")
    job = backend.run(new_circuit)
    job_result = job.result()
    print("job run")
    #counts = job_result.get_counts()

    # print statevector of non-junk qubits
    state_vec = job_result.get_statevector(qc).data
    state_vec_dict = {'{:b}'.format(i).zfill(qc.num_qubits):round(state_vec[i],5) for i in range(len(state_vec))}
    N_eig_eval = int(math.pow(2,n_eig_eval_bits))
    state_vec_collapsed = np.zeros(N_eig_eval)
    for i in range(N_eig_eval):
        for j in range(A_mat_size):
            state_vec_collapsed[i] += state_vec[j*N_eig_eval + i] * state_vec[j*N_eig_eval + i].conj() # collpase state vec onto eigenvalue evaluation qubits (keeping only the 0 state for the block encode bits)
            #print('{:b}'.format(i).zfill(A_bits), "    ",'{:b}'.format(j).zfill(n_eig_eval_bits), "    ", round(state_vec[i*int(math.pow(2,A_bits)) + j],8))
        state_vec_collapsed[i] = math.sqrt(state_vec_collapsed[i])
    '''state_vec_collapsed = np.zeros(A_mat_size)
    for i in range(A_mat_size):
        for j in range(n_eig_eval_states):
            state_vec_collapsed[i] += state_vec[i*N_eig_eval + j] * state_vec[i*N_eig_eval + j].conj() # collpase state vec onto eigenvector qubits (keeping only the 0 state for the block encode bits)
            #print('{:b}'.format(i).zfill(A_bits), "    ",'{:b}'.format(j).zfill(n_eig_eval_bits), "    ", round(state_vec[i*int(math.pow(2,A_bits)) + j],8))
        state_vec_collapsed[i] = math.sqrt(state_vec_collapsed[i])'''

    state_vec_collapsed = state_vec_collapsed / np.linalg.norm(state_vec_collapsed)
    index_max = max(range(len(state_vec_collapsed)), key=state_vec_collapsed.__getitem__)
    max_index_binary = ('{:b}'.format(index_max).zfill(n_eig_eval_bits))
    eig_result_i = 0
    for i in range(n_eig_eval_bits):
        eig_result_i += int(max_index_binary[i]) * int(math.pow(2,i))
    print("eigenvalue found: ", (eig_result_i) / n_eig_eval_states)
    print("expected eigenvalue: ", eigvals[eig_index])

elif method == "counts":
    # measure eigenvalue qubits
    qc.measure(list(range(n_eig_eval_bits)), list(range(n_eig_eval_bits)))

    # measure qubits containing eigenvector
    #qc.measure(list(range(n_eig_eval_bits, n_eig_eval_bits + A_bits)), list(range(n_eig_eval_bits, n_eig_eval_bits + A_bits)))

    # measure qubits used for block encoding
    qc.measure(list(range(n_eig_eval_bits + A_bits, n_eig_eval_bits + block_encode_bits)), list(range(n_eig_eval_bits + A_bits, n_eig_eval_bits + block_encode_bits)))

    simulator = Aer.get_backend('qasm_simulator')
    circ = transpile(qc, simulator)
    print("circuit transpiled")

    # Run and get counts
    result = simulator.run(circ, shots = num_iter).result()
    counts = result.get_counts(circ)
    print("job run")
    #plot_histogram(counts, title='Bell-State counts')

    # rearrange counts of eigenvector qubits to be in order
    #test = '{:b}'.format(10).zfill(block_encode_bits).ljust(qc.num_qubits,'0')
    #test2 = counts[test]
    #counts_dict = {'{:b}'.format(i).zfill(A_bits):counts['{:b}'.format(i).zfill(block_encode_bits).ljust(qc.num_qubits,'0')] if '{:b}'.format(i).zfill(block_encode_bits).ljust(qc.num_qubits,'0') in counts else 0 for i in range(A_mat_size)}
    
    #TODO: set up array/dictionary for measuring the eigenvalue evaluation qubits and compare to the actual fundamental eigenvalue, doesn't seem to work yet
    counts_dict = {'{:b}'.format(i).zfill(n_eig_eval_bits):counts['{:b}'.format(i).zfill(n_eig_eval_bits).ljust(qc.num_qubits,'0')] if '{:b}'.format(i).zfill(n_eig_eval_bits).ljust(qc.num_qubits,'0') in counts else 0 for i in range(n_eig_eval_states)}
    

qc.draw('mpl', filename="test_block_encoding.png")

print(counts_dict)
#test = counts['{:b}'.format(4).zfill(n_eig_eval_bits + A_bits).ljust(qc.num_qubits,'0')]
# TODO: See if I can reduce the difference between the input eigenvector and output eigenvector
# It only appears when doing the QPE step, probably from the approximation of the A_squiggle matrices with block encoding
counts_of_A_matrix = np.array([counts['{:b}'.format(i).zfill(block_encode_bits).ljust(qc.num_qubits,'0')] if '{:b}'.format(i).zfill(block_encode_bits).ljust(qc.num_qubits,'0') in counts else 0 for i in range(A_mat_size)])
predicted_state_from_counts = getStateFromCounts(counts_of_A_matrix)
error_vector = predicted_state_from_counts - eigenvector_input

print("predicted_state: ", predicted_state_from_counts)
print("eigenvector input: ", eigenvector_input)
print("Error vector: ", error_vector)
print("l2 norm of error vector: ", np.linalg.norm(error_vector))

ax = sns.heatmap(predicted_state_from_counts.reshape(data.n_x, data.n_y), linewidth=0.5)
ax.invert_yaxis()
plt.title("Predicted Solution")
plt.figure()

ax = sns.heatmap(eigenvector_input.reshape(data.n_x, data.n_y), linewidth=0.5)
ax.invert_yaxis()
plt.title("Actual Eigenvector")
plt.figure()

ax = sns.heatmap(error_vector.reshape(data.n_x, data.n_y), linewidth=0.5)
ax.invert_yaxis()
plt.title("Error")
plt.show()

#print("predicted_state norm: ", np.linalg.norm(predicted_state_from_counts))
#print("eigenvector input norm: ", np.linalg.norm(eigenvector_input))