import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import math
import cmath
from scipy.linalg import expm
from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian
from scipy.linalg import eigh
import time
import ProblemData
import LcuFunctions

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate

# Use AQC methods to evolve a quantum state from the fundamental eigenvalue of an initial to a final Hamiltonian
def adiabatic_solver(L_matrix, F_matrix, b_vec, T, M, plot_evolution=False, verbose=False):
    # setup sections
    X = np.array([[0, 1],[1, 0]])
    Z = np.array([[1, 0],[0, -1]])
    n_bits = int(math.log2(len(A_matrix)))
    num_LCU_bits = 6
    b_vec = b_vec / np.linalg.norm(b_vec)

    # set up matrices for Hamiltonian
    P_b = np.eye(len(A_matrix)) - np.outer(b_vec, b_vec)
    #H_B = np.matmul(np.matmul(np.eye(len(A_matrix)), P_b), np.eye(len(A_matrix)))
    L_B = P_b
    L_P = L_matrix

    U_B = np.eye(int(math.pow(2,n_bits + num_LCU_bits)))
    best_j = 1
    best_y_max = 1.5
    best_z_max = 1.5
    U_P, alphas, error_norm = LcuFunctions.get_fourier_unitaries(pow(2,best_j), pow(2,num_LCU_bits-best_j-1), best_y_max, best_z_max, F_matrix, True, len(F_matrix))
    psi = b_vec
    V = LcuFunctions.gram_schmidt_ortho(np.sqrt(alphas))

    F_inv_B = np.kron(np.conj(V).T,np.eye(int(math.pow(2,n_bits)))) @ U_B @ np.kron(V,np.eye(int(math.pow(2,n_bits))))
    F_inv_P = np.kron(np.conj(V).T,np.eye(int(math.pow(2,n_bits)))) @ U_P @ np.kron(V,np.eye(int(math.pow(2,n_bits))))

    # testing the LCU inverse matrix
    F_inv = np.linalg.inv(F_matrix)
    real_F_inv_P = np.kron(np.eye(int(math.pow(2,num_LCU_bits))), F_inv)
    test_phi = np.ones(len(F_inv_P)) / math.sqrt(len(F_inv_P))

    real_F_inv_phi = real_F_inv_P @ test_phi
    LCU_F_inv_phi = F_inv_P @ test_phi

    print("real: ", real_F_inv_phi)
    print("LCU: ", LCU_F_inv_phi)

    H_B = F_inv_B @ np.kron(np.eye(int(math.pow(2,num_LCU_bits))), L_B)
    H_P = F_inv_P @ np.kron(np.eye(int(math.pow(2,num_LCU_bits))), L_P)

    # print eigenvalues of H_B and H_P
    '''if(verbose):
        print("H_B", U_B)
        H_B_eigenvalues, H_B_eigenvectors = np.linalg.eig(H_B)
        print("H_B eigenvalues: ", H_B_eigenvalues)
        print("H_B eigenvectors: ", H_B_eigenvectors)
        print("H_P", U_P)
        H_P_eigenvalues, H_P_eigenvectors = np.linalg.eig(H_P)
        print("H_P eigenvalues: ", H_P_eigenvalues)
        print("H_P eigenvectors: ", H_P_eigenvectors)'''

    dt = T/M
    #print("delta-t * delta-H = ", np.linalg.norm(dt * (H_P - H_B))) # test whether time steps are small enough

    # initialize vectors containing the evolution of the state over time
    state_evolution = np.zeros((M,int(math.pow(2,n_bits + num_LCU_bits))),dtype=np.complex_)
    expected_state_evolution = np.zeros((M,int(math.pow(2,n_bits + num_LCU_bits))),dtype=np.complex_)
    eigenvector_error = np.zeros((M,1),dtype=np.complex_)
    eigenvector_error_abs = np.zeros((M,1),dtype=np.complex_)
    eigenvalue_evolution = np.zeros((len(A_matrix),M))

    #lastH = H_B
    U_T = np.eye(int(math.pow(2,n_bits))) # matrix representing the multiplication of all other matrices applied in the evolution process
    qb = QuantumRegister(n_bits)  # right hand side and solution
    ql = QuantumRegister(num_LCU_bits)  # LCU ancilla zero bits
    qc = QuantumCircuit(qb, ql)

    qc.append(LcuFunctions.get_b_setup_gate(psi, n_bits), qb[:])

    for l in range(M):
        s = l/M
        H = H_B + (H_P - H_B) * s # slowly change Hamiltonian

        # using equation 5.4 in Farhi
        U = expm(-(1j) * dt * H)
        U_gate = UnitaryGate(U, 'U', False)
        qc.append(U_gate, qb[:] + ql[:])
        #U_T = np.matmul(U,U_T)
        #n[l,:] = psi

        # expected state evolution to check answers
        '''H_eig, eigenvectors = np.linalg.eig(H)
        min_eig_id = H_eig.tolist().index(min(H_eig))
        if(eigenvectors[0,min_eig_id].real < 0):
            eigenvectors = eigenvectors * -1
        if(psi[0].real < 0):
            psi = psi * -1
        expected_state_evolution[l,:] =  eigenvectors[:,min_eig_id]
        eigenvalue_evolution[:,l] = H_eig
        eigenvector_error[l] = np.linalg.norm(psi - expected_state_evolution[l,:])
        eigenvector_error_abs[l] = np.linalg.norm(abs(psi) - abs(expected_state_evolution[l,:]))
        if (verbose and l % 100 == 0):
            print(", delta-t * delta-H = ", np.linalg.norm(dt * (lastH - H)), ", 1/M = ", 1/M)
            print("l = ", l, " psi = ", np.round(psi, decimals=8))
            print("expected psi: ", expected_state_evolution[l,:])
            print("psi error: ", psi - expected_state_evolution[l,:])

        lastH = H'''

        # using equation 5.7 in Farhi, will allow for faster process when more qubits are used, but needs more work
        '''v = l*dt/T
        u = 1-v
        K_min = M * math.pow((1 + dt*np.linalg.norm(H_B) + dt*np.linalg.norm(H_P)), 2)
        K = int(100 * K_min)
        H_B_exponent = expm(-(1j) * dt * u * H_B / K)
        H_P_exponent = expm(-(1j) * dt * v * H_P / K)
        for i in range(K):
            psi = H_P_exponent.dot(psi)
            psi = H_B_exponent.dot(psi)'''

    qc.save_statevector()


    # Run quantum algorithm
    backend = QasmSimulator(method="statevector")
    new_circuit = transpile(qc, backend)
    job = backend.run(new_circuit)
    job_result = job.result()
    state_vec = job_result.get_statevector(qc).data

    # PLOTTING RESULTS
    if(plot_evolution):
        start_index = 0
        end_index = M
        colors = ['-.r','-.b','-.g','-.c','-.m','-.y','-.k','-.w',]
        legend_vec = [str(format(i,'b')) + " state" for i in range(int(math.pow(2,n_bits)))]

        # plot the evolution of the magnitude of each of the possible states
        n_plots_x = int(pow(2,math.floor(n_bits/2)))
        n_plots_y = int(pow(2,math.ceil(n_bits/2)))
        fig, axs = plt.subplots(n_plots_x, n_plots_y)
        for i in range(int(math.pow(2,n_bits))):
            x = state_evolution[start_index:end_index,i].real
            y = state_evolution[start_index:end_index,i].imag
            if (n_plots_x == 1):
                axs[i].plot(x,y,colors[i%len(colors)])
                axs[i].set_title(legend_vec[i] + " state")
            else:
                axs[i % n_plots_x, math.floor(i/n_plots_x)].plot(x,y,colors[i%len(colors)])
                axs[i % n_plots_x, math.floor(i/n_plots_x)].set_title(legend_vec[i] + " state")


        # plot the evolved states of H(s) for each s
        plt.figure()
        for i in range(M):
            x = np.array(range(int(math.pow(2,n_bits)))) * 0 + i
            y = abs(state_evolution[i,:])
            plt.plot(x,y,'.')
        plt.legend(["eig 1", "eig 2", "eig 3", "eig 4", "eig 5", "eig 6", "eig 7", "eig 8"])
        plt.title('state magnitudes of H(s)')

        # plot the actual eigenvectors of H(s) for each s
        plt.figure()
        for i in range(M):
            x = np.array(range(int(math.pow(2,n_bits)))) * 0 + i
            y = abs(state_evolution[i,:])
            plt.plot(x,y,'.')
        plt.legend(["eig 1", "eig 2", "eig 3", "eig 4", "eig 5", "eig 6", "eig 7", "eig 8"])
        plt.title('eigenvector magnitudes of H(s)')

        # plot the eigenvalues of H(s) for each s
        plt.figure()
        for i in range(int(math.pow(2,n_bits))):
            x = range(M)
            y = eigenvalue_evolution[i,:]
            plt.plot(x,y,'.')
        plt.legend(["eig 1", "eig 2", "eig 3", "eig 4", "eig 5", "eig 6", "eig 7", "eig 8"])
        plt.title('eigenvalues of H(s)')

        # plot the error betwee psi and the actual base eigenvector for all s
        plt.figure()
        plt.plot(range(M),eigenvector_error,'.')
        plt.title('error on psi')

        plt.figure()
        plt.plot(range(M),eigenvector_error_abs,'.')
        plt.title('error on absolute value of psi')

        # 
        '''plt.figure()
        for i in range(int(math.pow(2,n_bits))):
            x = np.abs(state_evolution[start_index:end_index,i])
            y = state_evolution[start_index:end_index,i] * 0
            plt.plot(x,y,colors[i])
        plt.legend(legend_vec)
        plt.title('magnitude of actual state evolution')
        plt.figure()

        # plot expected eigenvectors throughout the evolution if that is known
        for i in range(int(math.pow(2,n_bits))):
            x = expected_state_evolution[start_index:end_index,i].real
            y = expected_state_evolution[start_index:end_index,i].imag
            plt.plot(x,y,colors[i])
        plt.legend(legend_vec)
        plt.title('expected state evolution')'''

        plt.figure()
        for i in range(int(math.pow(2,n_bits))):
            x = np.abs(state_evolution[start_index:end_index,i]) - np.abs(expected_state_evolution[start_index:end_index,i])
            plt.plot(x,colors[i%len(colors)])
        plt.legend(['0 state', '1 state'])
        plt.title('difference between magnitude of actual state evolution and expected')

        plt.show()

    state_vec = state_vec[0:len(b_vec)]
    return state_vec

# input section for parametric simulations
input_path = "simulations/AQC_1G_diffusion_small/1G_diffusion.txt"
data = ProblemData.ProblemData(input_path)

# create the vectors holding the material data at each discretized point
data.read_input(input_path)
data.initialize_BC()
data.initialize_materials()
data.initialize_geometry()
# make A matrix and b vector
if data.sim_method == "sp3":
    print("aaaa noooo I didn't implement this :(")
    #A_mat_size = 2 * (data.n_x) * (data.n_y)
    #A_matrix, b_vec = data.sp3_construct_A_matrix(A_mat_size) 
elif data.sim_method == "diffusion":
    A_mat_size = math.prod(data.n) * data.G
    L_matrix, F_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)

eigvals, eigvecs = eigh(L_matrix, F_matrix, eigvals_only=False)
F_inv = np.linalg.inv(F_matrix) # Can't actually do this matrix inversion in the quantum algorithm, can replace this with LCU method for approximating the inverse of an operator
A_matrix = F_inv @ L_matrix

#A_matrix = np.array([[-1, -4,  0,  3], [-4, -1,  0,  0],  [0,  0,  2,  0], [ 3,  0,  0, -1]])
#b_vec = np.array([5,6,7,8])
#T_vec = np.power(10,range(11))
#M_vec = np.power(10,range(2,6))
T_vec = [100]
M_vec = [10]
n_bits = 1 + int(math.log2(len(A_matrix)))

# real answer to linear system
print("real answer (scaled):")
A_eigenvalues, A_eigenvectors = np.linalg.eig(A_matrix)
real_psi_solution = A_eigenvectors[:,np.argmin(A_eigenvalues)]
real_psi_solution = real_psi_solution/np.linalg.norm(real_psi_solution)
if(real_psi_solution[0].real < 0):
    real_psi_solution = real_psi_solution * -1
print(real_psi_solution)

psi_initial = np.ones(A_mat_size) / math.sqrt(A_mat_size)

# parametric solutions, run solver for many M and T values
psi_solutions = np.zeros((len(T_vec), len(M_vec), len(A_matrix)), dtype=np.complex_)
psi_error = np.zeros((len(T_vec), len(M_vec), len(A_matrix)), dtype=np.complex_)
time1 = time.perf_counter()
for i, T in enumerate(T_vec):
    for j, M in enumerate(M_vec):
        psi = adiabatic_solver(L_matrix, F_matrix, psi_initial, T, M, plot_evolution=False, verbose=False)
        psi_solutions[i,j,:] = psi
        psi_error[i,j,:] = psi - real_psi_solution

        print("T: ", T)
        print("M: ", M)
        print("psi: ", psi)
        print("psi absolute value: ", np.abs(psi))
        print("real psi: ", real_psi_solution)
        print("psi error: ", psi - real_psi_solution)
time2 = time.perf_counter()
print("solver run time: ", time2 - time1)

min_val = min(np.min(np.abs(psi)),np.min(real_psi_solution))
max_val = min(np.max(np.abs(psi)),np.max(real_psi_solution))

psi.resize((data.n))
ax = sns.heatmap(np.abs(psi), linewidth=0.5, cmap="jet", vmin=min_val, vmax=max_val)
plt.title("Quantum Solution")
plt.savefig('q_sol.png')
plt.figure()

real_psi_solution.resize((data.n))
ax = sns.heatmap(real_psi_solution, linewidth=0.5, cmap="jet", vmin=min_val, vmax=max_val)
plt.title("Real Solution")
plt.savefig('real_sol.png')
plt.figure()
plt.show()

# plot parametric results
'''for i,M in enumerate(M_vec):
    plt.loglog(T_vec, np.linalg.norm(psi_error,axis=2)[:,i])
    plt.title("norm of error in psi vs. time T with M=" + str(M))
    plt.figure()
plt.show()'''

