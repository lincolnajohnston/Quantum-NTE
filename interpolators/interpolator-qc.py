import sys
import os
sys.path.append(os.getcwd())
import math
import numpy as np
import matplotlib.pyplot as plt

from helpers.ProblemData import ProblemData

from qiskit import transpile
from qiskit_aer import QasmSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation

# Code Description: 
# Interpolates a coarse mesh represented as a quantum state over 'nc' qubits to a quantum state representing the same solution on a fine mesh over 'nf' qubits
# Uses 1-qubit gates to perform the inteprolation. Intention of this algorithm is to use in the FEEN/QPE algorithm to improve the overlap between the coarse mesh input
# and the fundamental eigenvector of the fine mesh matrix.

# Linear Combination of states reference: https://arxiv.org/pdf/2112.12307

############## Functions ##############

# Creates a quantum state where each term is approxmiately a sequential integer [1,2,3,...] (and then scaled)
def get_interpolater_state(nf, nc, qc1):
    N = nf - nc
    total_norm = 1
    last_val = 1
    dot_prod = 1
    for i in range(N):
        q_n = [1,math.pow(N,1/(N))-math.pow(i,1/(N))+1]
        q_n_norm = np.linalg.norm(q_n)
        q_n = q_n / q_n_norm
        last_val *= q_n[1]
        state = StatePreparation(q_n)
        qc1.append(state,[2*nf-nc-i])
        total_norm = total_norm * q_n_norm
        dot_prod *= sum(q_n) / math.sqrt(2)

    return last_val, dot_prod

# Creates a quantum state where each term is a sequential integer (and then scaled), used to find error of quantum solution
def get_perfect_interpolater_state(N):
    perfect_interp = [i for i in range(int(math.pow(2,N)))] # the state we are aiming to approximate
    return perfect_interp / np.linalg.norm(perfect_interp)

############## Inputs ##############
nc = 4
Nc = int(math.pow(2,nc))
nf = 7
Nf = int(math.pow(2,nf))
dn = nf-nc
num_iter = 1000000

# TODO: get the coarse solution from solving the diffusion equation, then also get the fine solution from solving the same diffusion equation on a finer grid
# look at the scaling of the error between the fine solution and the perfectly interpolated coarse solution then the error between the fine solution and
# the approximately interpolated solution, this will give us the O(h^x) scaling to find the overall success probability of QPE

# TODO 1: L2 norm of difference between true solution and interpolated coarse solution
# TODO 2: L2 norm of difference between normalized coefficients on basis functions for true solution and interpolated coarse solution
# TODO 3: L2 norm of difference between normalized coefficients on basis functions for true solution and repeated coarse solution
# TODO 4: Do this with a diffusion equation solution instead of a test function

# get classical solution to coarse diffusion equation
coarse_sim_path = 'simulations/Pu239_1G_1D_diffusion_coarse/'
input_file_coarse = 'input.txt'
coarse_data = ProblemData(coarse_sim_path + input_file_coarse)
A_mat_size_coarse = math.prod(coarse_data.n) * coarse_data.G
A_matrix_coarse, b_vector_coarse = coarse_data.diffusion_construct_A_matrix(A_mat_size_coarse)
classical_sol_vec_coarse = np.abs(np.linalg.solve(A_matrix_coarse, b_vector_coarse))
solution_coarse_normed = classical_sol_vec_coarse / np.linalg.norm(classical_sol_vec_coarse)

# get classical solution to fine diffusion equation
fine_sim_path = 'simulations/Pu239_1G_1D_diffusion_fine/'
input_file_fine = 'input.txt'
fine_data = ProblemData(fine_sim_path + input_file_fine)
A_mat_size_fine = math.prod(fine_data.n) * fine_data.G
A_matrix_fine, b_vector_fine = fine_data.diffusion_construct_A_matrix(A_mat_size_fine)
classical_sol_vec_fine = np.abs(np.linalg.solve(A_matrix_fine, b_vector_fine))
solution_fine_normed = classical_sol_vec_fine / np.linalg.norm(classical_sol_vec_fine)

#coarse_sol = np.array([random.uniform(0, 1) for _ in range(Nc)])
#coarse_sol = np.array([0.1,0.2,0.3,0.4,0.45,0.5,0.8,0.9])
#coarse_sol = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.43, 0.45, 0.47, 0.5, 0.6, 0.8, 0.82, 0.9, 0.99])
#coarse_sol = np.sin(0.3 * np.array(list(range(2**nc)))) + 1
coarse_sol = solution_coarse_normed
print("coarse solution: ", coarse_sol)
coarse_sol_diff = np.array(coarse_sol)
coarse_sol_diff[:Nc-1] -= coarse_sol_diff[1:Nc]
coarse_sol_diff *= -1
coarse_sol_diff[Nc-1] = 0 

coarse_sol_norm = np.linalg.norm(coarse_sol)
coarse_sol_diff_norm = np.linalg.norm(coarse_sol_diff)
coarse_sol = coarse_sol / coarse_sol_norm
coarse_sol_diff = coarse_sol_diff / coarse_sol_diff_norm

############## Quantum Circuit Setup ##############

qc1 = QuantumCircuit(2*nf+1,2*nf+1)

last_val, interp_dot_prod = get_interpolater_state(nf, nc, qc1)
interpolater_norm = (math.pow(2,dn)-1) / last_val # norm that assures that the last value of interp_state is math.pow(2,dn)-1, which is directly on the linear interpolation line
alpha_0 = math.pow(2,dn/2) * coarse_sol_norm
alpha_1 = interpolater_norm * coarse_sol_diff_norm / math.pow(2,dn)

perf_interp_state = get_perfect_interpolater_state(dn)
perf_interpolater_norm = (math.pow(2,dn)-1) / perf_interp_state[-1] # norm that assures that the last value of interp_state is math.pow(2,dn)-1, which is directly on the linear interpolation line
perf_alpha_1 = perf_interpolater_norm * coarse_sol_diff_norm / math.pow(2,dn)

beta_0 = 1/math.sqrt(3)
beta_1 = math.sqrt(2)/math.sqrt(3)
beta = np.array([beta_0, beta_1])
sigma_matrix = np.outer(beta, beta.conj())
phi_dot_product = np.dot(coarse_sol,coarse_sol_diff) * interp_dot_prod # the closer the dot product is to 0, the more variance there will be in the result

# matrix that allows the two states to be linearly combined
M_matrix = np.array([[(alpha_0 * alpha_0) / (beta_0 * beta_0 * 1), (alpha_1 * alpha_0) / (beta_1 * beta_0 * phi_dot_product)],
              [(alpha_0 * alpha_1) / (beta_0 * beta_1 * phi_dot_product), (alpha_1 * alpha_1) / (beta_1 * beta_1 * 1)]])

# desired state: used for testing and finding error of quantum state
phi_goal = alpha_0 * np.kron(coarse_sol, 1/math.pow(2,(dn)/2) * np.ones(int(Nf/Nc))) + perf_alpha_1 * np.kron(coarse_sol_diff, perf_interp_state)

M_eigenvalues, M_eigenvectors = np.linalg.eig(M_matrix)

# ancilla qubit
sigma_stateprep = StatePreparation(beta)
qc1.append(sigma_stateprep, [nf])

# term 1
stateprep_phi_c = StatePreparation(coarse_sol)
qc1.append(stateprep_phi_c, list(range(nf-nc, nf)))
for i in range(nf-nc):
    qc1.h(i)

#term 2
stateprep_phi_0 = StatePreparation(coarse_sol_diff)
qc1.append(stateprep_phi_0, list(range(2 * nf + 1 - nc, 2 * nf + 1)))

# put cswap gates in quantum circuit
for i in range(nf):
    qc1.cswap(nf,i,nf+1+i)

############## Perform Measurements, Record Counts ##############

# T transforms the state so that it can be measured in the computational basis and retain the same probabilities of each eigenvalue being measured
T = np.outer([1,0],M_eigenvectors[:,0]) + np.outer([0,1],M_eigenvectors[:,1])
T_gate = UnitaryGate(T)
qc1.append(T_gate, [nf])
qc1.measure([nf],[nf])
qc1.measure(list(range(nf)), list(range(nf)))
backend = QasmSimulator(method="statevector")
new_circuit = transpile(qc1, backend)
job = backend.run(new_circuit, shots=num_iter)
job_result = job.result()
counts = job_result.get_counts()

############## Post-Processing ##############

# weight the counts by the eigenvalue associated with the measurement on the sigma qubit
weighted_counts = {}
for i, (bitkey, n) in enumerate(counts.items()):
    eig = M_eigenvalues[int(bitkey[nf])]
    if bitkey[nf+1:2*nf+1] in weighted_counts:
        weighted_counts[bitkey[nf+1:2*nf+1]] += eig * n
    else:
        weighted_counts[bitkey[nf+1:2*nf+1]] = eig * n
predicted_state = [math.sqrt(abs(weighted_counts['{:b}'.format(i).zfill(nf)]/num_iter)) if '{:b}'.format(i).zfill(nf) in weighted_counts else 0 for i in range(Nf)]

############## Show Results ##############

# print difference between measured and desired state and L2
print("predicted_state: ", predicted_state)    
print("desired_state: ", phi_goal)
phi_goal_normed = phi_goal / np.linalg.norm(phi_goal)
predicted_state_normed = predicted_state / np.linalg.norm(predicted_state)  
interp_error = np.linalg.norm(phi_goal_normed - predicted_state_normed)
print("Interpolated L2 error: ", interp_error)

repeated_phi = np.kron(coarse_sol, coarse_sol_norm * np.ones(int(Nf/Nc)))
repeated_phi_normed = repeated_phi / np.linalg.norm(repeated_phi) 
repeated_error = np.linalg.norm(repeated_phi_normed - predicted_state_normed)
print("Repeated L2 error: ", repeated_error)


# plot results
'''plt.plot(list(range(0, int(Nf/Nc) * len(coarse_sol), int(Nf/Nc))), coarse_sol_norm * coarse_sol, 'o')
plt.plot(predicted_state)
plt.plot(phi_goal)
plt.plot(repeated_phi)
#plt.plot(predicted_state - phi_goal)

plt.title("Desired vs Measured States")
plt.legend(['Input Coarse Grid', 'Measured State', 'Desired State', 'Repeated Phi'])
plt.show()'''

# just plot coarse and fine solutions, not quantum circuit results
plt.plot()


qc1.draw('mpl', filename="1d-interpolator-circuit.png")
