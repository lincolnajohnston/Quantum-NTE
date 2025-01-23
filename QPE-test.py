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

#sim_path = 'simulations/LCU_8G_diffusion/'
sim_path = 'simulations/test_1G_diffusion/'
input_file = 'input.txt'
data = ProblemData.ProblemData(sim_path + input_file)
A_mat_size = (data.n_x) * (data.n_y) * data.G
A_bits = math.ceil(math.log2(A_mat_size))
n_eig_eval_bits = 5  # number of bits to represent the final eigenvalue
A_matrix, B_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
# A_matrix and B_matrix will not necessarily be Hermitian for all problems, but I think for 1G problems they are
print("A is hermitian: ", ishermitian(A_matrix))
print("B is hermitian: ", ishermitian(B_matrix))

A_eigenvalues, A_eigenvectors = np.linalg.eig(A_matrix)
B_eigenvalues, B_eigenvectors = np.linalg.eig(B_matrix)

A_sing_vals = svdvals(A_matrix)
eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)

print("B is positive definite: ", np.all(B_eigenvalues > 0))
#print(A_matrix)
#print(B_matrix)

# eigenvector corresponding to the smallest eigenvalue of the GEP
eigenvector_input = eigvecs[:,0]
eigenvector_input = eigenvector_input/np.linalg.norm(eigenvector_input) # normalize into quantum state

sqrtB = sqrtm(B_matrix)
sqrtB_inv = np.linalg.inv(sqrtB)
A_squiggle = sqrtB_inv @ A_matrix @ sqrtB
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
qc, alpha_sqrtB = fable.fable(sqrtB, qc, epsilon=0)

# block encoding of B^(-1/2)*A*B^(1/2) so that when the most significant bits (bottom bits aka higher index) are 
# all 0, the state on the less significant (lower index) bits will resemble the state, c * B^(-1/2)*A*B^(1/2) * (arbitrary_phi)
# TODO: FIX THIS, not done correctly here because I'm being lazy, more efficient way to block encode this from Changpeng paper
A_squiggle_qc, alpha_A_squiggle = fable.fable(A_squiggle, epsilon=0)

# do QPE
qpe = PhaseEstimation(n_eig_eval_bits, A_squiggle_qc, circuit=qc)

# do B^(-1/2) on the eigenvector state to return it to its original state
qc, alpha_sqrtB_inv = fable.fable(sqrtB_inv, qc, epsilon=0)


method = "counts"
if method == "statevector":
    qc.save_statevector()

    # Run emulator
    backend = QasmSimulator(method="statevector")
    new_circuit = transpile(qc, backend)
    job = backend.run(new_circuit)
    job_result = job.result()
    #counts = job_result.get_counts()

    # print statevector of non-junk qubits
    state_vec = job_result.get_statevector(qc).data
    state_vec_dict = {'{:b}'.format(i).zfill(qc.num_qubits):round(state_vec[i],5) for i in range(len(state_vec))}
    N_eig_eval = int(math.pow(2,n_eig_eval_bits))
    state_vec_collpased = np.zeros(N_eig_eval)
    for i in range(N_eig_eval):
        for j in range(A_mat_size):
            state_vec_collpased[i] += state_vec[i*A_mat_size + j] * state_vec[i*A_mat_size + j].conj()
            #print('{:b}'.format(i).zfill(A_bits), "    ",'{:b}'.format(j).zfill(n_eig_eval_bits), "    ", round(state_vec[i*int(math.pow(2,A_bits)) + j],8))
        state_vec_collpased[i] = math.sqrt(state_vec_collpased[i])

elif method == "counts":
    # measure qubits used for block encoding
    qc.measure(list(range(n_eig_eval_bits + A_bits, n_eig_eval_bits + block_encode_bits)), list(range(n_eig_eval_bits + A_bits, n_eig_eval_bits + block_encode_bits)))

    # measure qubits containing eigenvector
    qc.measure(list(range(n_eig_eval_bits, n_eig_eval_bits + A_bits)), list(range(n_eig_eval_bits, n_eig_eval_bits + A_bits)))

    simulator = Aer.get_backend('qasm_simulator')
    circ = transpile(qc, simulator)

    # Run and get counts
    result = simulator.run(circ, shots = num_iter).result()
    counts = result.get_counts(circ)
    #plot_histogram(counts, title='Bell-State counts')

    # rearrange counts of eigenvector qubits to be in order
    test = '{:b}'.format(10).zfill(n_eig_eval_bits + A_bits).ljust(qc.num_qubits,'0')
    test2 = counts[test]
    counts_dict = {'{:b}'.format(i).zfill(A_bits):counts['{:b}'.format(i).zfill(n_eig_eval_bits + A_bits).ljust(qc.num_qubits,'0')] if '{:b}'.format(i).zfill(n_eig_eval_bits + A_bits).ljust(qc.num_qubits,'0') in counts else 0 for i in range(A_mat_size)}
    

qc.draw('mpl', filename="test_block_encoding.png")

print(counts_dict)
test = counts['{:b}'.format(4).zfill(n_eig_eval_bits + A_bits).ljust(qc.num_qubits,'0')]
# TODO: Think I might have normalized this wrong, still getting answers slightly wrong
predicted_state_from_counts = [counts['{:b}'.format(i).zfill(n_eig_eval_bits + A_bits).ljust(qc.num_qubits,'0')] if '{:b}'.format(i).zfill(n_eig_eval_bits + A_bits).ljust(qc.num_qubits,'0') in counts else 0 for i in range(A_mat_size)]
normalizer = np.linalg.norm(np.sqrt(predicted_state_from_counts))
predicted_state_from_counts = np.sqrt(predicted_state_from_counts)/normalizer
print("predicted_state: ", predicted_state_from_counts)
print("eigenvector input: ", eigenvector_input)

print("predicted_state norm: ", np.linalg.norm(predicted_state_from_counts))
print("eigenvector input norm: ", np.linalg.norm(eigenvector_input))