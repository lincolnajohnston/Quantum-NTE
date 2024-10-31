import numpy as np
import math
import cmath
from scipy.linalg import expm
from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian
import time
import ProblemData
import random
import os

#from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.visualization import plot_histogram

# References:
# [1] https://arxiv.org/pdf/1805.10549
# [2] https://arxiv.org/pdf/quant-ph/0001106

# Return the A(s) operator from equation 3 of [1]
def get_A_s(s, A):
    X = np.array([[0, 1],[1, 0]])
    Z = np.array([[1, 0],[0, -1]])
    return np.kron((1-s) * Z, np.eye(len(A))) + np.kron(s * X, A)
    
def adiabatic_solver(A_matrix, b_vec, M, plot_evolution=False, verbose=False, qiskit_solve=False):
    # Normalize
    A_matrix = A_matrix / np.linalg.norm(A_matrix)
    b_vec = b_vec / np.linalg.norm(b_vec)
    
    # Set up matrices for Hamiltonian
    b_bar_plus = np.kron(np.array([1/math.sqrt(2), 1/math.sqrt(2)]), b_vec) # b vector with ancilla "+" qubit inserted at the beginning
    psi = np.kron(np.array([1/math.sqrt(2), -1/math.sqrt(2)]), b_vec) # set the initial state to be a "-" qubit and the b vector state, which is the eigenvector corresponding to the 0-eigenvalue of H(s) from equation 3 of [1] 
    P_b = np.eye(len(A_matrix) * 2) - np.outer(b_bar_plus, b_bar_plus) # the P_b operator from equation 3 of [1]
    A_B = get_A_s(0,A_matrix) # initial A(s) when s=0
    A_P = get_A_s(1,A_matrix) # final A(s) when s=1
    H_B = np.matmul(np.matmul(A_B, P_b), A_B) # initial Hamiltonian matrix
    H_P = np.matmul(np.matmul(A_P, P_b), A_P) # final Hamiltonian matrix

    # Initialize quantum circuit
    '''set the initial state to be a "-" qubit and the b vector state,
    which is the eigenvector corresponding to the 0-eigenvalue of H(s) from equation 3 of [1]'''
    if(qiskit_solve):
        n_qubits = int(np.log2(len(A_matrix))) + 1
        qc = QuantumCircuit(n_qubits)
        qc.initialize(psi, qc.qubits)

    # initialize data structures to store the system state as evolution progresses
    #state_evolution = np.zeros((M, 2 ** n_bits), dtype=np.complex_)
    #expected_state_evolution = np.zeros((M, 2 ** n_bits), dtype=np.complex_)
    #eigenvector_error = np.zeros((M, 1), dtype=np.complex_)
    #eigenvector_error_abs = np.zeros((M, 1), dtype=np.complex_)
    #eigenvalue_evolution = np.zeros((2 * len(A_matrix), M))

    lastH = H_B
    dAds = (A_P - A_B) / M
    A = A_B

    # Use randomization method from Subasi to get f_vals
    kappa = np.linalg.cond(A_matrix)
    v_a = math.sqrt(2)*kappa / math.sqrt(1+kappa**2) * math.log(kappa*math.sqrt(1+kappa**2) - kappa**2)
    v_b = math.sqrt(2)*kappa / math.sqrt(1+kappa**2) * math.log(math.sqrt(1+kappa**2) + 1)
    v = np.linspace(v_a,v_b,num=M)
    f_vals = (np.exp(v * math.sqrt(1+kappa**2) / (math.sqrt(2)*kappa)) + 2*kappa**2 - kappa**2*np.exp(-v * math.sqrt(1+kappa**2) / (math.sqrt(2)*kappa))) / (2 * (1+kappa**2))
    T = 0

    for l in range(M):
        s = f_vals[l]
        A = A_B + s * (A_P - A_B)
        H = np.matmul(np.matmul(A, P_b), A)

        delta_prime = (1-s)**2 + (s/kappa)**2
        dt = random.random() * 2 * math.pi / delta_prime # sample dt uniformly from range of times
        T += dt
        
        # using equation 5.4 in [2], and from algorithm after eq. 9 in [1]
        U = expm(-1j * dt * H)

        if(qiskit_solve):
            # Add to circuit
            U_gate = UnitaryGate(U)
            qc.append(U_gate, qc.qubits)
        else:
            psi = U.dot(psi)

        if(l % 100 == 0 and verbose):
            print("l = ", l)
        
        lastH = H
    
    if(qiskit_solve):
        # Simulate the circuit
        simulator = Aer.get_backend('aer_simulator')
        qc.save_statevector()
        compiled_circuit = transpile(qc, simulator)
        result = simulator.run(compiled_circuit).result()
        final_state = result.get_statevector(qc)

        # Plotting the results
        if plot_evolution:
            plot_histogram(final_state.probabilities_dict())
            
        return final_state.data, T
    else:
        return psi, T

    return false

# manually input linear system
#A_matrix = np.array([[-1, -4,  0,  3], [-4, -1,  0,  0],  [0,  0,  2,  0], [ 3,  0,  0, -1]])
#b_vec = np.array([5,6,7,8])

# Use ProblemData class to create a linear system corresponding to a neutron transport equation discretization
sim_path = 'simulations/1G_sp3_small/'
input_file = '1G_sp3.txt'
data = ProblemData.ProblemData(sim_path + input_file)
# make A matrix and b vector
if data.sim_method == "sp3":
    A_mat_size = 2 * data.G * (data.n_x) * (data.n_y)
    A_matrix, b_vec = data.sp3_construct_A_matrix(A_mat_size) 
elif data.sim_method == "diffusion":
    A_mat_size = data.G * (data.n_x) * (data.n_y)
    A_matrix, b_vec = data.diffusion_construct_A_matrix(A_mat_size)

# Input which T and M values to test
#T_vec = np.power(10,range(11))
#M_vec = np.power(10,range(2,6))
#T_vec = [1000000]
M_vec = [1]
#M_vec = np.ones(40) * 500
n_bits = 1 + int(math.log2(len(A_matrix)))

# real answer to linear system
print("real answer (scaled):")
real_psi_solution = np.linalg.inv(A_matrix).dot(b_vec)
real_psi_solution = real_psi_solution/np.linalg.norm(real_psi_solution)
print(real_psi_solution)

# parametric solutions, run solver for many M and T values
#psi_solutions = np.zeros((len(T_vec), len(M_vec), len(A_matrix)), dtype=np.complex_)
#psi_error = np.zeros((len(T_vec), len(M_vec), len(A_matrix)), dtype=np.complex_)
time1 = time.perf_counter()
for j, M in enumerate(M_vec):
    # Use Qiskit for updating state
    #solution = adiabatic_solver_qiskit(A_matrix, b_vec, T, M, plot_evolution=False, verbose=False)
    #psi = solution.data

    # just use numpy arrays for quantum states
    psi, T = adiabatic_solver(A_matrix, b_vec, int(M), plot_evolution=False, verbose=True, qiskit_solve=False)

    # get state resulting from measuring 0 on ancilla qubit (first qubit)
    psi = psi[0:int(len(psi)/2)]
    psi = psi / np.linalg.norm(psi)

    # just get magnitude of each variable by removing imaginary and negative parts, this part is hacky a little
    #psi = np.abs(psi)

    #psi_solutions[i,j,:] = psi
    #psi_error[i,j,:] = psi - real_psi_solution

    print("T: ", T)
    print("M: ", M)
    #print("psi: ", psi)
    #print("psi absolute value: ", np.abs(psi))
    #print("real psi: ", real_psi_solution)
    #print("psi error: ", psi - real_psi_solution)
    precision = np.linalg.norm(np.abs(psi)-real_psi_solution)
    print("precision: ", precision)

    # save data vectors to files
    i = 0
    while os.path.exists(sim_path + 'saved_data/stats' + str(i) + '.txt'):
        i += 1
    f = open(sim_path + 'saved_data/stats' + str(i) + '.txt', "w")
    f.write("precision: " +  str(precision))
    f.close()
    np.savetxt(sim_path + 'saved_data/psi' + str(i) + '.npy', psi)
    np.savetxt(sim_path + 'saved_data/real_psi_solution' + str(i) + '.npy', real_psi_solution)
time2 = time.perf_counter()
print("solver run time: ", time2 - time1)

# plot parametric results
'''for i,M in enumerate(M_vec):
    plt.loglog(T_vec, np.linalg.norm(psi_error,axis=2)[:,i])
    plt.title("norm of error in psi vs. time T with M=" + str(M))
    plt.figure()
plt.show()'''

# get plotting ranges and tickmark locations
min_val = min(np.min(np.abs(psi)),np.min(real_psi_solution))
max_val = min(np.max(np.abs(psi)),np.max(real_psi_solution))
xticks = np.round(np.array(range(data.n_x))*data.delta_x - (data.n_x - 1)*data.delta_x/2,3)
yticks = np.round(np.array(range(data.n_y))*data.delta_y - (data.n_y - 1)*data.delta_y/2,3)

for g in range(data.G):
    psi.resize((data.G,data.n_x,data.n_y))
    ax = sns.heatmap(np.abs(psi[g,:,:]), linewidth=0.5, cmap="jet", vmin=min_val, vmax=max_val, xticklabels=xticks, yticklabels=yticks)
    ax.invert_yaxis()
    plt.title("Quantum Solution")
    plt.savefig('q_sol.png')
    plt.figure()

    real_psi_solution.resize((data.G, data.n_x,data.n_y))
    ax = sns.heatmap(real_psi_solution[g,:,:], linewidth=0.5, cmap="jet", vmin=min_val, vmax=max_val, xticklabels=xticks, yticklabels=yticks)
    ax.invert_yaxis()
    plt.title("Real Solution")
    plt.savefig('real_sol.png')
    plt.figure()

    ax = sns.heatmap(np.abs(psi[g,:,:]) - real_psi_solution[g,:,:], linewidth=0.5, cmap="jet", xticklabels=xticks, yticklabels=yticks)
    ax.invert_yaxis()
    plt.title("Error Between Correct Psi and Quantum Solution of Psi")
    plt.savefig('sol_error.png')
    plt.figure()

plt.show()

