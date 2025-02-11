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

# Code Description:
# Script for plotting the scaling of the theoretical upper limit on the probability of FEEN/QPE success based on the size of the coarse and fine mesh grids

# from a vector of counts for each basis vector, return the normalized state representing the amplitudes for each of the basis vectors
def getStateFromCounts(counts_vec):
    norm = math.sqrt(np.sum(counts_vec))
    return np.sqrt(counts_vec) / norm

def get_success_prob(coarse_data, fine_data):
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
        return state_overlap ** 2

coarse_input_file = 'simulations/Pu239_1G_diffusion_ANS_coarse/input.txt'
fine_input_file = 'simulations/Pu239_1G_diffusion_ANS_fine/input.txt'
coarse_data = ProblemData.ProblemData(coarse_input_file)
fine_data = ProblemData.ProblemData(fine_input_file)

# fixed fine mesh, varying coarse mesh inputs
'''fine_nxs = [64]
fine_nys = [64]
fine_dxs = 16.768 / np.array(fine_nxs) # exact eigenvalue
fine_dys = 16.768 / np.array(fine_nxs)
#fine_dxs = [1] # nearly exact eigenvalue
#fine_dys = [1]
#fine_dxs = [0.5] # in between discrete eigenvlaues
#fine_dys = [0.5]

coarse_nxs = [2,4,8,16,32,64]
coarse_nys = [2,4,8,16,32,64]
coarse_dxs = 16.768 / np.array(coarse_nxs) # exact eigenvalue
coarse_dys = 16.768 / np.array(coarse_nxs)
#coarse_dxs = [8,4,2,1] # nearly exact eigenvalue
#coarse_dys = [8,4,2,1]
#coarse_dxs = [4,2,1,0.5] # in between discrete eigenvlaues
#coarse_dys = [4,2,1,0.5]'''


# fixed coarse mesh, varying fine mesh inputs
fine_nxs = [8,16,32,64]
fine_nys = [8,16,32,64]
fine_dxs = 16.768 / np.array(fine_nxs) # exact eigenvalue
fine_dys = 16.768 / np.array(fine_nys)
#fine_dxs = [1] # nearly exact eigenvalue
#fine_dys = [1]
#fine_dxs = [0.5] # in between discrete eigenvlaues
#fine_dys = [0.5]

coarse_nxs = [8]
coarse_nys = [8]
coarse_dxs = 16.768 / np.array(coarse_nxs) # exact eigenvalue
coarse_dys = 16.768 / np.array(coarse_nys)
#coarse_dxs = [8,4,2,1] # nearly exact eigenvalue
#coarse_dys = [8,4,2,1]
#coarse_dxs = [4,2,1,0.5] # in between discrete eigenvlaues
#coarse_dys = [4,2,1,0.5]


success_probs = np.zeros(len(fine_nxs) * len(coarse_nxs))
for i in range(len(fine_nxs)):
    for j in range(len(coarse_nxs)):
        fine_data.n_x = fine_nxs[i]
        fine_data.n_pts_x = fine_nxs[i] + 2
        fine_data.n_y = fine_nys[i]
        fine_data.n_pts_y = fine_nys[i] + 2
        fine_data.delta_x = fine_dxs[i]
        fine_data.delta_y = fine_dys[i]
        fine_data.initialize_BC()
        fine_data.initialize_geometry()

        coarse_data.n_x = coarse_nxs[j]
        coarse_data.n_pts_x = coarse_nxs[j] + 2
        coarse_data.n_y = coarse_nys[j]
        coarse_data.n_pts_y = coarse_nys[j] + 2
        coarse_data.delta_x = coarse_dxs[j]
        coarse_data.delta_y = coarse_dys[j]
        coarse_data.initialize_BC()
        coarse_data.initialize_geometry()

        prob = get_success_prob(coarse_data, fine_data)

        success_probs[i*len(coarse_nxs) + j] = prob

print(success_probs)

'''plt.plot(coarse_nxs, success_probs, '-o')
plt.title("success probability vs " + r'$h_{coarse}$' + " for " r'$h_{fine}$' + "= " + str(fine_nxs[0]))
plt.xlabel(r'$h_{coarse}$')
plt.ylabel("success probability")
plt.show()'''

plt.plot(fine_nxs, success_probs, '-o')
plt.title("success probability vs " + r'$h_{fine}$' + " for " r'$h_{coarse}$' + "= " + str(coarse_nxs[0]))
plt.xlabel(r'$h_{fine}$')
plt.ylabel("success probability")
plt.show()