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
# -do better block encoding on B^(1/2)
# -implement the block encoding of A_squiggle in the QPE correctly, apply the exponent of A_squiggle quantumly, not classically

########## Functions ##########

# from a vector of counts for each basis vector, return the normalized state representing the amplitudes for each of the basis vectors
def getStateFromCounts(counts_vec):
    norm = math.sqrt(np.sum(counts_vec))
    return np.sqrt(counts_vec) / norm

########## Read input, set up variables ##########

# Fundamental Eigenvalue Estimator for the NTE
class FEEN():
    def __init__(
        self,
        n_eig_eval_bits: int, # number of bits to represent the final eigenvalue
        coarse_input_file,
        fine_input_file,
        sim_method="statevector", # either "counts" or "statevector" mode uses different methods of quantum simulation in Qiskit
        plot_results = False, # plot the coarse and fine eigenvectors after the simulation
    ) -> None:
        self.coarse_data = ProblemData.ProblemData(coarse_input_file)
        self.fine_data = ProblemData.ProblemData(fine_input_file)
        self.n_eig_eval_bits = n_eig_eval_bits
        self.sim_method = sim_method
        self.plot_results = plot_results

    def psuedo_invert_diagonal_matrix(self, in_matrix):
        M = np.array(in_matrix)
        for i in range(len(M)):
            M[i,i] = 0 if M[i,i]==0 else 1/M[i,i]
        return M

    def find_eigenvalue(self):
        A_coarse_mat_size = math.prod(self.coarse_data.n) * self.coarse_data.G
        A_coarse_bits_vec = [math.ceil(math.log2(self.coarse_data.G))] + [math.ceil(math.log2(self.coarse_data.n[i])) for i in range(self.coarse_data.dim)]
        A_coarse_bits = sum(A_coarse_bits_vec)
        A_mat_size = math.prod(self.fine_data.n) * self.fine_data.G
        A_bits_vec = [math.ceil(math.log2(self.fine_data.G))] + [math.ceil(math.log2(self.fine_data.n[i])) for i in range(self.fine_data.dim)]
        A_bits = sum(A_bits_vec)
        interpolation_bits = A_bits - A_coarse_bits
        n_eig_eval_states = int(math.pow(2,self.n_eig_eval_bits))
        reverse_order = False

        # A_matrix and B_matrix will not necessarily be Hermitian for all problems, but I think for 1G problems they are
        # If hermitian, then QPE will output their eigenvalues, if not, need to make A and B hermitian (using one more qubit)
        if self.coarse_data.sim_method == "sp3":
            A_matrix_coarse, B_matrix_coarse = self.coarse_data.sp3_construct_L_F_matrices(A_coarse_mat_size)
            A_matrix, B_matrix = self.fine_data.sp3_construct_L_F_matrices(A_mat_size)
        elif self.coarse_data.sim_method == "diffusion":
            '''if reverse_order:
                # NDE with operators in opposite order
                B_matrix_coarse, A_matrix_coarse = self.coarse_data.diffusion_construct_L_F_matrices(A_coarse_mat_size)
                B_matrix, A_matrix = self.fine_data.diffusion_construct_L_F_matrices(A_mat_size)
            else: 
                # NDE with operators in normal order
                A_matrix_coarse, B_matrix_coarse = self.coarse_data.diffusion_construct_L_F_matrices(A_coarse_mat_size)
                A_matrix, B_matrix = self.fine_data.diffusion_construct_L_F_matrices(A_mat_size)'''
            # NDE with operators in normal order
            A_matrix_coarse, B_matrix_coarse = self.coarse_data.diffusion_construct_L_F_matrices(A_coarse_mat_size)
            A_matrix, B_matrix = self.fine_data.diffusion_construct_L_F_matrices(A_mat_size)

        # find condition numbers and alphas of A and B matrices to help determine efficiency of implementing a block-encoding of them
        alpha_A = np.linalg.norm(np.ravel(A_matrix), np.inf)
        alpha_B = np.linalg.norm(np.ravel(B_matrix), np.inf)
        cond_A = np.linalg.cond(A_matrix)
        cond_B = np.linalg.cond(B_matrix)

        # find eigenvector and eigenvalues of the GEP classically (should use power iteration for real problems)
        if reverse_order:
            eigvals_coarse, eigvecs_coarse = eigh(B_matrix_coarse, A_matrix_coarse, eigvals_only=False)
        else:
            eigvals_coarse, eigvecs_coarse = eigh(A_matrix_coarse, B_matrix_coarse, eigvals_only=False)
        eig_index = -1 if reverse_order else 0 # index of eigenvector/eigenvalue to use, -1 for fundamental eigenvector of inverse equation, 0 for fundamental eigenvector of standard equation

        # solve problem classically to compare to quantum results
        if reverse_order:
            eigvals, eigvecs = eigh(B_matrix, A_matrix, eigvals_only=False)
        else:
            eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
        eigenvector_fine = eigvecs[:,eig_index] / np.linalg.norm(eigvecs[:,eig_index])

        ########## Create matrices/vectors needed for the circuit ##########

        # eigenvector_input is the state that is fed into the quantum circuit
        # TODO: make eigenvector input the interpolation of the coarse mesh eigvec
        eigenvector_input = eigvecs_coarse[:,eig_index]
        eigenvector_input = eigenvector_input/np.linalg.norm(eigenvector_input) # normalize into quantum state

        # Create the sqrtB, sqrtB_inv, and A_squiggle matrices. Cheating here for now by finding them classically.
        # TODO: block encode these using the methods from the Changpeng paper
        sqrtB = sqrtm(B_matrix)
        sqrtB_inv = self.psuedo_invert_diagonal_matrix(sqrtB) if reverse_order else np.linalg.inv(sqrtB)
        A_squiggle = sqrtB_inv @ A_matrix @ sqrtB_inv
        test = sqrtB @ sqrtB_inv
        A_squiggle_pow = expm(2j*math.pi*A_squiggle)
        #A_squiggle_pow_eigvals, A_squiggle_pow_eigvecs = np.linalg.eig(A_squiggle_pow)
        print("A_squiggle_pow is unitary: ", LcuFunctions.is_unitary(A_squiggle_pow))

        ########## Create Quantum Circuit ##########

        logn = fable.get_logn(sqrtB)
        block_encode_bits = 2*logn+1
        qc = QuantumCircuit(block_encode_bits + self.n_eig_eval_bits, block_encode_bits + self.n_eig_eval_bits)

        # put the quantum circuit in the coarse phi_0 state and do interpolation onto the fine grid
        eigvec_input_state = StatePreparation(eigenvector_input)
        #eigvec_input_state = StatePreparation(math.pow(1/math.sqrt(2),A_coarse_bits) * np.ones(A_coarse_mat_size))
        #eigvec_input_state = StatePreparation(eigvecs_coarse[:,1] / np.linalg.norm(eigvecs_coarse[:,1]))
        A_bits_diff_vec = np.array(A_bits_vec) - np.array(A_coarse_bits_vec)
        coarse_eig_bits = [q for i in range(self.fine_data.dim + 1) for q in list(range(self.n_eig_eval_bits + sum(A_bits_vec[:i]) + A_bits_diff_vec[i], self.n_eig_eval_bits + sum(A_bits_vec[:i+1])))] # qubits the coarse eigenvector solution will be put on
        qc.append(eigvec_input_state, coarse_eig_bits) # add coarse solution to the quantum circuit
        for i in list(set(range(self.n_eig_eval_bits, self.n_eig_eval_bits + A_bits)) - set(coarse_eig_bits)): # add in Hadamards to extend the coarse solution to a fine grid
            qc.h(i)

        # extract the interpolated input eigenvector from the quantum ciruit
        input_state = Statevector.from_instruction(qc).data
        input_state_collapsed = input_state[:n_eig_eval_states*A_mat_size:n_eig_eval_states]
        #input_state_collapsed = np.zeros(A_mat_size)

        # block encoding of B^(1/2) so that when the most significant bits (bottom bits aka higher index) are 
        # all 0, the state on the less significant (lower index) bits will resemble the state, c * B^(1/2) * phi_0
        # TODO: FIX THIS, not done correctly here because I'm being lazy, more efficient way to block encode this from Changpeng paper
        qc, alpha_sqrtB = fable.fable(sqrtB, qc, epsilon=0.01, max_i = qc.num_qubits-1)
        #state_0 = Statevector.from_instruction(qc).data
        #state_0_collapsed = state_0[:n_eig_eval_states*A_mat_size:n_eig_eval_states] / np.linalg.norm(state_0[:n_eig_eval_states*A_mat_size:n_eig_eval_states])

        # block encoding of B^(-1/2)*A*B^(1/2) so that when the most significant bits (bottom bits aka higher index) are 
        # all 0, the state on the less significant (lower index) bits will resemble the state, c * B^(-1/2)*A*B^(1/2) * (arbitrary_phi)
        # TODO: FIX THIS, not done correctly here because I'm being lazy, more efficient way to block encode this from Changpeng paper
        #A_squiggle_qc, alpha_A_squiggle = fable.fable(A_squiggle, epsilon=0, max_i = qc.num_qubits-1)

        # do QPE
        # TODO: change this to block encode A_squiggle, convert it to e^(2*pi*i*A_squiggle), then use that as the controlled unitary for QPE
        qpe = PhaseEstimation(self.n_eig_eval_bits, A_squiggle_pow, A_bits, circuit=qc)
        #state_1 = Statevector.from_instruction(qc).data
        #state_1_collapsed = state_1[:n_eig_eval_states*A_mat_size] / np.linalg.norm(state_1[:n_eig_eval_states*A_mat_size])

        # do B^(-1/2) on the eigenvector state to return it to its original state, the eigenvector of the GEP
        # seems to mess up the eigenvalue results for some reason, and is not necessary to get an eigenvalue result, so this is commented out for now
        #qc, alpha_sqrtB_inv = fable.fable(sqrtB_inv, qc, epsilon=0, max_i = qc.num_qubits-1)
        #state_2 = Statevector.from_instruction(qc).data
        #state_2_collapsed = state_2[:n_eig_eval_states*A_mat_size] / np.linalg.norm(state_2[:n_eig_eval_states*A_mat_size])

        ########## Run the circuit, output results ##########
        #coarse_eigvec_expanded = math.pow((1/math.sqrt(2)), x_bits_diff + y_bits_diff + G_bits_diff) * np.kron(eigenvector_input.reshape(coarse_data.n_x, coarse_data.n_y, coarse_data.G), np.ones((int(math.pow(2,x_bits_diff)), int(math.pow(2,y_bits_diff)), int(math.pow(2,G_bits_diff))))).reshape(A_mat_size)
        #state_overlap = np.dot(coarse_eigvec_expanded.conj(), eigenvector_fine)

        if self.sim_method == "statevector":
            qc.save_statevector()

            # Run emulator
            backend = QasmSimulator(method="statevector")
            new_circuit = transpile(qc, backend)
            print(dict(new_circuit.count_ops())) # print the counts of each type of gate
            job = backend.run(new_circuit)
            job_result = job.result()

            # print statevector of non-junk qubits
            state_vec = job_result.get_statevector(qc).data
            #eigvec_collapsed = state_vec[0:int(math.pow(2,self.n_eig_eval_bits+A_bits)):int(math.pow(2,self.n_eig_eval_bits))]
            #state_vec_dict = {'{:b}'.format(i).zfill(qc.num_qubits):round(state_vec[i],5) for i in range(len(state_vec))}
            N_eig_eval = int(math.pow(2,self.n_eig_eval_bits))
            state_vec_collapsed = np.zeros(N_eig_eval)
            for i in range(N_eig_eval):
                for j in range(A_mat_size):
                    state_vec_collapsed[i] += state_vec[j*N_eig_eval + i] * state_vec[j*N_eig_eval + i].conj() # collpase state vec onto eigenvalue evaluation qubits (keeping only the 0 state for the block encode bits)
                state_vec_collapsed[i] = math.sqrt(state_vec_collapsed[i])
            state_vec_collapsed = state_vec_collapsed / np.linalg.norm(state_vec_collapsed)

            '''state_vec_collapsed_eigvec = np.zeros(A_mat_size)
            for j in range(A_mat_size):
                for i in range(N_eig_eval):
                    state_vec_collapsed_eigvec[j] += state_vec[j*N_eig_eval + i] * state_vec[j*N_eig_eval + i].conj() # collpase state vec onto eigenvalue evaluation qubits (keeping only the 0 state for the block encode bits)
                state_vec_collapsed_eigvec[j] = math.sqrt(state_vec_collapsed_eigvec[j])
            state_vec_collapsed_eigvec = state_vec_collapsed_eigvec / np.linalg.norm(state_vec_collapsed_eigvec)
            eigenvector_error = state_vec_collapsed_eigvec - eigenvector_fine'''

            index_max = max(range(len(state_vec_collapsed)), key=state_vec_collapsed.__getitem__)

            # when the build-in IQFT is used in the QPE script, bit reversal like this needs to be done, but I can't seem to get that working
            #max_index_binary = ('{:b}'.format(index_max).zfill(n_eig_eval_bits))
            #eig_result_i = 0
            #for i in range(n_eig_eval_bits):
            #    eig_result_i += int(max_index_binary[i]) * int(math.pow(2,i))
            print("Simulation Run: ")
            print("expected eigenvalue: ", eigvals[eig_index])
            print("eigenvalue found: ", (index_max) / n_eig_eval_states)
            print("probability of getting this eigenvalue on one measurement: ", state_vec_collapsed[index_max] ** 2)
            print("square of inner product of input state and actual fine eigenvector: ", np.inner(input_state_collapsed, eigenvector_fine) ** 2)
            self.expected_eigenvalue = eigvals[eig_index]
            self.found_eigenvalue = index_max / n_eig_eval_states
            self.expected_fidelity = float(np.inner(input_state_collapsed, eigenvector_fine)) ** 2
            self.found_fidelity = state_vec_collapsed[index_max] ** 2
            #self.found_fidelity = np.sum([state_vec_collapsed[i] ** 2 for i in range(index_max - 5, index_max + 5)])

        elif self.sim_method == "counts":
            num_iter = 1000 # should be able to only run this once if probablity of success if high and you do some amplitude amplification on the block encode bits
            # measure eigenvalue qubits
            qc.measure(list(range(self.n_eig_eval_bits)), list(range(self.n_eig_eval_bits)))

            # measure qubits containing eigenvector, don't run this line except for debugging
            #qc.measure(list(range(n_eig_eval_bits, n_eig_eval_bits + A_bits)), list(range(n_eig_eval_bits, n_eig_eval_bits + A_bits)))

            # measure qubits used for block encoding
            qc.measure(list(range(self.n_eig_eval_bits + A_bits, self.n_eig_eval_bits + block_encode_bits)), list(range(self.n_eig_eval_bits + A_bits, self.n_eig_eval_bits + block_encode_bits)))

            simulator = Aer.get_backend('qasm_simulator')
            circ = transpile(qc, simulator)

            # Run and get counts
            result = simulator.run(circ, shots = num_iter).result()
            counts = result.get_counts(circ)
            # make new dictionary with counts only from where the block encode bits are all 0
            counts_dict = {'{:b}'.format(i).zfill(self.n_eig_eval_bits):counts['{:b}'.format(i).zfill(qc.num_qubits)] if '{:b}'.format(i).zfill(qc.num_qubits) in counts else 0 for i in range(n_eig_eval_states)}

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
            self.expected_eigenvalue = eigvals[eig_index]
            self.found_eigenvalue = index_max / n_eig_eval_states
            self.expected_fidelity = np.inner(input_state_collapsed, eigenvector_fine) ** 2
            #self.found_fidelity = np.sum([counts_vec[i] for i in range(index_max - 2, index_max + 2)])/sum(counts_vec)


        if(self.plot_results and self.fine_data.dim == 2):
            fig, (ax1, ax2) = plt.subplots(1,2)
            heatmap_min = min(np.min(np.abs(input_state_collapsed)), np.min(np.abs(eigenvector_fine)))
            heatmap_max = max(np.max(np.abs(input_state_collapsed)), np.max(np.abs(eigenvector_fine)))

            xticks = np.round(np.array(range(self.fine_data.n[0]))*self.fine_data.h[0] - (self.fine_data.n[0] - 1)*self.fine_data.h[0]/2,3)
            yticks = np.round(np.array(range(self.fine_data.n[1]))*self.fine_data.h[1] - (self.fine_data.n[1] - 1)*self.fine_data.h[1]/2,3)

            ax = sns.heatmap(np.abs(input_state_collapsed).reshape(self.fine_data.n[0], self.fine_data.n[1]), linewidth=0.5, ax=ax1, xticklabels=xticks, yticklabels=yticks, vmin=heatmap_min, vmax=heatmap_max)
            ax1.set_xlabel("x (cm)")
            ax1.set_ylabel("y (cm)")
            ax.invert_yaxis()
            #plt.title("Input (Coarse) Solution")
            #plt.figure()

            ax = sns.heatmap(np.abs(eigenvector_fine).reshape(self.fine_data.n[0], self.fine_data.n[1]), linewidth=0.5, ax=ax2, xticklabels=xticks, yticklabels=yticks, vmin=heatmap_min, vmax=heatmap_max)
            ax2.set_xlabel("x (cm)")
            ax2.set_ylabel("y (cm)")
            ax.invert_yaxis()
            #plt.title("Actual Fine Solution")
            plt.figure()

            error_vector = np.abs(input_state_collapsed) - np.abs(eigenvector_fine)
            ax = sns.heatmap(error_vector.reshape(self.fine_data.n[0], self.fine_data.n[1]), linewidth=0.5)
            ax.invert_yaxis()
            plt.title("Error")
            plt.show()

        if(self.plot_results and self.fine_data.dim == 1):
            fig, (ax1, ax2) = plt.subplots(1,2)
            heatmap_min = min(np.min(np.abs(input_state_collapsed)), np.min(np.abs(eigenvector_fine)))
            heatmap_max = max(np.max(np.abs(input_state_collapsed)), np.max(np.abs(eigenvector_fine)))


            ax = sns.heatmap(np.abs(input_state_collapsed.reshape(self.fine_data.n[0], 1)), linewidth=0.5, ax=ax1, vmin=heatmap_min, vmax=heatmap_max)
            ax1.set_xlabel("x (cm)")
            ax1.set_ylabel("y (cm)")
            ax.invert_yaxis()
            #plt.title("Input (Coarse) Solution")
            #plt.figure()

            ax = sns.heatmap(np.abs(eigenvector_fine.reshape(self.fine_data.n[0], 1)), linewidth=0.5, ax=ax2, vmin=heatmap_min, vmax=heatmap_max)
            ax2.set_xlabel("x (cm)")
            ax2.set_ylabel("y (cm)")
            ax.invert_yaxis()
            #plt.title("Actual Fine Solution")
            plt.figure()

            error_vector = np.abs(input_state_collapsed) - np.abs(eigenvector_fine)
            ax = sns.heatmap(error_vector.reshape(self.fine_data.n[0], 1), linewidth=0.5)
            ax.invert_yaxis()
            plt.title("Error")
            plt.show()

        # draw circuit, can take a while to run
        #qc.draw('mpl', filename="test_block_encoding.png")


# simulation to find eigenvector heatmaps, ANS plot 1
n_eig_eval_bits = 5
start_time = time.time()
FEEN1 = FEEN(n_eig_eval_bits,'simulations/ProblemData_1D_scaling_tests_fuel_pin/input-coarse.txt', 'simulations/ProblemData_1D_scaling_tests_fuel_pin/input-fine.txt', plot_results=True, sim_method="statevector")
FEEN1.find_eigenvalue() # uncomment this when I just want to run the QPE algorithm once

print("Found Eigenvalue: ", FEEN1.found_eigenvalue)
print("Expected Eigenvalue: ", FEEN1.expected_eigenvalue)

print("Runtime: ", time.time() - start_time)

#print("Found Inverse Eigenvalue: ", 1/FEEN1.found_eigenvalue)
#print("Expected Inverse Eigenvalue: ", 1/FEEN1.expected_eigenvalue)


################## ANS plot 2, fixed fine mesh (16), varying coarse mesh(2-16)  ##################
'''n_eig_eval_bits = 5
FEEN1 = FEEN(n_eig_eval_bits,'simulations/Pu239_1G_diffusion_ANS_coarse/input.txt', 'simulations/Pu239_1G_diffusion_ANS_fine/input.txt', plot_results=False)

# fixed fine mesh, varying coarse mesh inputs
x_range = 16
y_range = x_range
fine_nxs = [16]
fine_nys = [16]
fine_dxs = x_range / np.array(fine_nxs)
fine_dys = y_range / np.array(fine_nys)

coarse_nxs = [2,4,8,16]
coarse_nys = [2,4,8,16]
#coarse_nxs = [2]
#coarse_nys = [2]
coarse_dxs = x_range / np.array(coarse_nxs)
coarse_dys = y_range / np.array(coarse_nys)

success_probs = np.zeros(len(fine_nxs) * len(coarse_nxs))
expected_fidelity = np.zeros(len(fine_nxs) * len(coarse_nxs))
for i in range(len(fine_nxs)):
    for j in range(len(coarse_nxs)):
        FEEN1.fine_data.n_x = fine_nxs[i]
        FEEN1.fine_data.n_pts_x = fine_nxs[i] + 2
        FEEN1.fine_data.n_y = fine_nys[i]
        FEEN1.fine_data.n_pts_y = fine_nys[i] + 2
        FEEN1.fine_data.delta_x = fine_dxs[i]
        FEEN1.fine_data.delta_y = fine_dys[i]
        FEEN1.fine_data.initialize_BC()
        FEEN1.fine_data.initialize_geometry()


        FEEN1.coarse_data.n_x = coarse_nxs[j]
        FEEN1.coarse_data.n_pts_x = coarse_nxs[j] + 2
        FEEN1.coarse_data.n_y = coarse_nys[j]
        FEEN1.coarse_data.n_pts_y = coarse_nys[j] + 2
        FEEN1.coarse_data.delta_x = coarse_dxs[j]
        FEEN1.coarse_data.delta_y = coarse_dys[j]
        FEEN1.coarse_data.initialize_BC()
        FEEN1.coarse_data.initialize_geometry()

        FEEN1.find_eigenvalue()

        print("Found Eigenvalue: ", FEEN1.found_eigenvalue)
        print("Expected Eigenvalue: ", FEEN1.expected_eigenvalue)

        success_probs[i*len(coarse_nxs) + j] = FEEN1.found_fidelity if abs(FEEN1.found_eigenvalue - FEEN1.expected_eigenvalue) < (1/math.pow(2,n_eig_eval_bits)) else 0
        expected_fidelity[i*len(coarse_nxs) + j] = FEEN1.expected_fidelity if abs(FEEN1.found_eigenvalue - FEEN1.expected_eigenvalue) < (1/math.pow(2,n_eig_eval_bits)) else 0

# plot varying coarse mesh input with fixed fine mesh
plt.plot(coarse_nxs, success_probs, '-o')
plt.plot(coarse_nxs, expected_fidelity, '-o')
plt.title(r'$P_{s}$' + " vs " + r'$h_{c}$' + " for a Fixed " r'$h_{f}$' + "= " + str(fine_nxs[0]))
plt.legend(["Experimental " + r'$P_{s}$', "Theoretical Probability: " + r'$||\langle\phi_c,\phi_f\rangle || ^2$'])
plt.xlabel(r'$h_{c}$')
plt.ylabel(r'$P_{s}$')
plt.grid(True)
plt.show()'''

################## ANS plot 3, fixed fine mesh (16), varying coarse mesh(2-16), exact eigenvalues  ##################
'''n_eig_eval_bits = 5
FEEN1 = FEEN(n_eig_eval_bits,'simulations/Pu239_1G_diffusion_ANS_coarse/input.txt', 'simulations/Pu239_1G_diffusion_ANS_fine/input.txt', plot_results=False)

# fixed fine mesh, varying coarse mesh inputs
x_range = 16.6
y_range = x_range
fine_nxs = [16]
fine_nys = [16]
fine_dxs = x_range / np.array(fine_nxs)
fine_dys = y_range / np.array(fine_nys)

coarse_nxs = [2,4,8,16]
coarse_nys = [2,4,8,16]
#coarse_nxs = [2]
#coarse_nys = [2]
coarse_dxs = x_range / np.array(coarse_nxs)
coarse_dys = y_range / np.array(coarse_nys)

success_probs = np.zeros(len(fine_nxs) * len(coarse_nxs))
expected_fidelity = np.zeros(len(fine_nxs) * len(coarse_nxs))
for i in range(len(fine_nxs)):
    for j in range(len(coarse_nxs)):
        FEEN1.fine_data.n_x = fine_nxs[i]
        FEEN1.fine_data.n_pts_x = fine_nxs[i] + 2
        FEEN1.fine_data.n_y = fine_nys[i]
        FEEN1.fine_data.n_pts_y = fine_nys[i] + 2
        FEEN1.fine_data.delta_x = fine_dxs[i]
        FEEN1.fine_data.delta_y = fine_dys[i]
        FEEN1.fine_data.initialize_BC()
        FEEN1.fine_data.initialize_geometry()


        FEEN1.coarse_data.n_x = coarse_nxs[j]
        FEEN1.coarse_data.n_pts_x = coarse_nxs[j] + 2
        FEEN1.coarse_data.n_y = coarse_nys[j]
        FEEN1.coarse_data.n_pts_y = coarse_nys[j] + 2
        FEEN1.coarse_data.delta_x = coarse_dxs[j]
        FEEN1.coarse_data.delta_y = coarse_dys[j]
        FEEN1.coarse_data.initialize_BC()
        FEEN1.coarse_data.initialize_geometry()

        FEEN1.find_eigenvalue()

        print("Found Eigenvalue: ", FEEN1.found_eigenvalue)
        print("Expected Eigenvalue: ", FEEN1.expected_eigenvalue)

        success_probs[i*len(coarse_nxs) + j] = FEEN1.found_fidelity if abs(FEEN1.found_eigenvalue - FEEN1.expected_eigenvalue) < (1/math.pow(2,n_eig_eval_bits)) else 0
        expected_fidelity[i*len(coarse_nxs) + j] = FEEN1.expected_fidelity if abs(FEEN1.found_eigenvalue - FEEN1.expected_eigenvalue) < (1/math.pow(2,n_eig_eval_bits)) else 0

# plot varying coarse mesh input with fixed fine mesh
plt.plot(coarse_nxs, success_probs, '-o')
plt.plot(coarse_nxs, expected_fidelity, '-o')
plt.title(r'$P_{s}$' + " vs " + r'$h_{c}$' + " for a Fixed " r'$h_{f}$' + "= " + str(fine_nxs[0]))
plt.legend(["Experimental " + r'$P_{s}$', "Theoretical Probability: " + r'$||\langle\phi_c,\phi_f\rangle || ^2$'])
plt.xlabel(r'$h_{c}$')
plt.ylabel(r'$P_{s}$')
plt.grid(True)
plt.show()'''





# fixed fine mesh, varying coarse mesh inputs
'''fine_nxs = [8]
fine_nys = [8]
#fine_dxs = [1.048] # exact eigenvalue
#fine_dys = [1.048]
fine_dxs = [2] # nearly exact eigenvalue
fine_dys = [2]
#fine_dxs = [0.5] # in between discrete eigenvlaues
#fine_dys = [0.5]

coarse_nxs = [2,4,8]
coarse_nys = [2,4,8]
#coarse_dxs = [8.384, 4.192, 2.096, 1.048] # exact eigenvalue
#coarse_dys = [8.384, 4.192, 2.096, 1.048]
coarse_dxs = [8,4,2] # nearly exact eigenvalue
coarse_dys = [8,4,2]
#coarse_dxs = [4,2,1,0.5] # in between discrete eigenvlaues
#coarse_dys = [4,2,1,0.5]'''

# fixed coarse mesh, varying fine mesh inputs
'''fine_nxs = [2,4,8]
fine_nys = [2,4,8]
fine_dxs = [8.384, 4.192, 2.096, 1.048] # exact eigenvalue
fine_dys = [8.384, 4.192, 2.096, 1.048]
#fine_dxs = [1] # nearly exact eigenvalue
#fine_dys = [1]
#fine_dxs = [0.5] # in between discrete eigenvlaues
#fine_dys = [0.5]

coarse_nxs = [2]
coarse_nys = [2]
coarse_dxs = [8.384] # exact eigenvalue
coarse_dys = [8.384]
#coarse_dxs = [8,4,2,1] # nearly exact eigenvalue
#coarse_dys = [8,4,2,1]
#coarse_dxs = [4,2,1,0.5] # in between discrete eigenvlaues
#coarse_dys = [4,2,1,0.5]'''


'''success_probs = np.zeros(len(fine_nxs) * len(coarse_nxs))
expected_fidelity = np.zeros(len(fine_nxs) * len(coarse_nxs))
for i in range(len(fine_nxs)):
    for j in range(len(coarse_nxs)):
        FEEN1.fine_data.n_x = fine_nxs[i]
        FEEN1.fine_data.n_pts_x = fine_nxs[i] + 2
        FEEN1.fine_data.n_y = fine_nys[i]
        FEEN1.fine_data.n_pts_y = fine_nys[i] + 2
        FEEN1.fine_data.delta_x = fine_dxs[i]
        FEEN1.fine_data.delta_y = fine_dys[i]
        FEEN1.fine_data.initialize_BC()
        FEEN1.fine_data.initialize_geometry()


        FEEN1.coarse_data.n_x = coarse_nxs[j]
        FEEN1.coarse_data.n_pts_x = coarse_nxs[j] + 2
        FEEN1.coarse_data.n_y = coarse_nys[j]
        FEEN1.coarse_data.n_pts_y = coarse_nys[j] + 2
        FEEN1.coarse_data.delta_x = coarse_dxs[j]
        FEEN1.coarse_data.delta_y = coarse_dys[j]
        FEEN1.coarse_data.initialize_BC()
        FEEN1.coarse_data.initialize_geometry()

        FEEN1.find_eigenvalue()

        print("Found Eigenvalue: ", FEEN1.found_eigenvalue)
        print("Expected Eigenvalue: ", FEEN1.expected_eigenvalue)

        success_probs[i*len(coarse_nxs) + j] = FEEN1.found_fidelity if abs(FEEN1.found_eigenvalue - FEEN1.expected_eigenvalue) < (1/math.pow(2,n_eig_eval_bits)) else 0
        expected_fidelity[i*len(coarse_nxs) + j] = FEEN1.expected_fidelity if abs(FEEN1.found_eigenvalue - FEEN1.expected_eigenvalue) < (1/math.pow(2,n_eig_eval_bits)) else 0'''

# plot varying coarse mesh input with fixed fine mesh
'''plt.plot(coarse_nxs, success_probs, '-o')
plt.plot(coarse_nxs, expected_fidelity, '-o')
plt.title("success probability vs " + r'$h_{coarse}$' + " for " r'$h_{fine}$' + "= " + str(fine_nxs[0]))
plt.legend(["experimental probability", "square of inner product of states"])
plt.xlabel(r'$h_{coarse}$')
plt.ylabel("success probability")
plt.show()'''

# plot varying fine mesh input with fixed coarse mesh input
'''plt.plot(fine_nxs, success_probs, '-o')
plt.plot(fine_nxs, expected_fidelity, '-o')
plt.title("success probability vs " + r'$h_{fine}$' + " for " r'$h_{coarse}$' + "= " + str(coarse_nxs[0]))
plt.legend(["experimental probability", "square of inner product of states"])
plt.xlabel(r'$h_{fine}$')
plt.ylabel("success probability")
plt.show()'''

#print(success_probs)
#print(expected_fidelity)