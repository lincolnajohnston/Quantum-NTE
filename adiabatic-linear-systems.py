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

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate

# Testing the QLSP solver from https://arxiv.org/pdf/1805.10549

def get_A_s(s, A):
    X = np.array([[0, 1],[1, 0]])
    Z = np.array([[1, 0],[0, -1]])
    return np.kron((1-s) * Z, np.eye(len(A))) + np.kron(s * X, A)
    

def adiabatic_solver(A_matrix, b_vec, T, M, plot_evolution=False, verbose=False):
    # setup sections
    X = np.array([[0, 1],[1, 0]])
    Z = np.array([[1, 0],[0, -1]])
    n_bits = 1 + int(math.log2(len(A_matrix)))
    A_matrix = A_matrix / np.linalg.norm(A_matrix)
    b_vec = b_vec / np.linalg.norm(b_vec)

    # set up matrices for Hamiltonian
    b_bar_plus = np.kron(np.array([1/math.sqrt(2), 1/math.sqrt(2)]), b_vec)
    psi = np.kron(np.array([1/math.sqrt(2), -1/math.sqrt(2)]), b_vec)
    P_b = np.eye(len(A_matrix) * 2) - np.outer(b_bar_plus, b_bar_plus)
    A_B = get_A_s(0,A_matrix)
    A_P = get_A_s(1,A_matrix)
    H_B = np.matmul(np.matmul(A_B, P_b), A_B)
    H_P = np.matmul(np.matmul(A_P, P_b), A_P)

    # print eigenvalues of H_B and H_P
    '''print("A: ", A_matrix)
    print("b_bar_plus: ", b_bar_plus)
    A_eigenvalues, A_eigenvectors = np.linalg.eig(A_matrix)
    print("A eigenvalues: ", A_eigenvalues)
    print("A eigenvectors: ", A_eigenvectors)
    print("A(0): ", get_A_s(0,A_matrix))
    A_eigenvalues, A_eigenvectors = np.linalg.eig(get_A_s(0,A_matrix))
    print("A(0) eigenvalues: ", A_eigenvalues)
    print("A(0) eigenvectors: ", A_eigenvectors)
    print("A(1): ", get_A_s(1,A_matrix))
    A_eigenvalues, A_eigenvectors = np.linalg.eig(get_A_s(1,A_matrix))
    print("A(1) eigenvalues: ", A_eigenvalues)
    print("A(1) eigenvectors: ", A_eigenvectors)
    print("H_B", H_B)
    H_B_eigenvalues, H_B_eigenvectors = np.linalg.eig(H_B)
    print("H_B eigenvalues: ", H_B_eigenvalues)
    print("H_B eigenvectors: ", H_B_eigenvectors)
    print("H_P", H_P)
    H_P_eigenvalues, H_P_eigenvectors = np.linalg.eig(H_P)
    print("H_P eigenvalues: ", H_P_eigenvalues)
    print("H_P eigenvectors: ", H_P_eigenvectors)
    H_B_eig, eigenvectors = np.linalg.eig(H_B)'''

    dt = T/M
    dH = H_P - H_B
    #print("psi = ", psi)
    #print("delta-t * delta-H = ", np.linalg.norm(dt * (H_P - H_B)))

    state_evolution = np.zeros((M,int(math.pow(2,n_bits))),dtype=np.complex_)
    expected_state_evolution = np.zeros((M,int(math.pow(2,n_bits))),dtype=np.complex_)
    eigenvector_error = np.zeros((M,1),dtype=np.complex_)
    eigenvector_error_abs = np.zeros((M,1),dtype=np.complex_)

    lastH = H_B
    U_T = np.eye(int(math.pow(2,n_bits)))
    A0 = get_A_s(0,A_matrix)
    A1 = get_A_s(1,A_matrix)
    dAds = (A1 - A0)/M
    A = A0
    eigenvalue_evolution = np.zeros((2*len(A_matrix),M))
    for l in range(M):
        s = l/M
        A = A + dAds
        H = np.matmul(np.matmul(A, P_b), A)

        # using equation 5.4 in Farhi
        U = expm(-(1j) * dt * H)
        U_T = np.matmul(U,U_T)
        psi = U.dot(psi)
        state_evolution[l,:] = psi
        #expected_state_evolution[l,:] = [1,-s/(s-1)-math.sqrt(1-2*s*(1-s))/(s-1)] # one qubit case

        # expected state evolution to check answers
        H_eig, eigenvectors = np.linalg.eig(H)
        min_eig_id = H_eig.tolist().index(min(H_eig))
        if(eigenvectors[0,min_eig_id].real < 0):
            eigenvectors = eigenvectors * -1
        expected_state_evolution[l,:] =  eigenvectors[:,min_eig_id]
        eigenvalue_evolution[:,l] = H_eig
        eigenvector_error[l] = np.linalg.norm(psi - expected_state_evolution[l,:])
        eigenvector_error_abs[l] = np.linalg.norm(abs(psi) - abs(expected_state_evolution[l,:]))
        if(psi[0].real < 0):
            psi = psi * -1
        if (verbose and l % 100 == 0):
            print(", delta-t * delta-H = ", np.linalg.norm(dt * (lastH - H)), ", 1/M = ", 1/M)
            print("l = ", l, " psi = ", np.round(psi, decimals=8))
            print("expected psi: ", expected_state_evolution[l,:])
            print("psi error: ", psi - expected_state_evolution[l,:])

        lastH = H

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

        state_evolution[l,:] = psi

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
                axs[i].plot(x,y,colors[i])
                axs[i].set_title(legend_vec[i] + " state")
            else:
                axs[i % n_plots_x, math.floor(i/n_plots_x)].plot(x,y,colors[i])
                axs[i % n_plots_x, math.floor(i/n_plots_x)].set_title(legend_vec[i] + " state")


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
        plt.title('error on psi')

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
            plt.plot(x,colors[i])
        plt.legend(['0 state', '1 state'])
        plt.title('difference between magnitude of actual state evolution and expected')

        plt.show()

    return psi

# input section for parametric simulations
A_matrix = np.array([[-1, -4,  0,  3], [-4, -1,  0,  0],  [0,  0,  2,  0], [ 3,  0,  0, -1]])
b_vec = np.array([5,6,7,8])
T_vec = np.power(10,range(11))
M_vec = np.power(10,range(2,6))
n_bits = 1 + int(math.log2(len(A_matrix)))

# real answer to linear system
print("real answer (scaled):")
real_psi_solution = np.linalg.inv(A_matrix).dot(b_vec)
real_psi_solution = real_psi_solution/np.linalg.norm(real_psi_solution)
print(real_psi_solution)

# parametric solutions, run solver for many M and T values
psi_solutions = np.zeros((len(T_vec), len(M_vec), len(A_matrix)), dtype=np.complex_)
psi_error = np.zeros((len(T_vec), len(M_vec), len(A_matrix)), dtype=np.complex_)
time1 = time.perf_counter()
for i, T in enumerate(T_vec):
    for j, M in enumerate(M_vec):
        psi = adiabatic_solver(A_matrix, b_vec, T, M, plot_evolution=False, verbose=False)
        psi = psi[0:int(len(psi)/2)]
        psi = psi / np.linalg.norm(psi)
        psi_solutions[i,j,:] = psi
        psi_error[i,j,:] = psi - real_psi_solution

        print("T: ", T)
        print("M: ", M)
        print("psi: ", psi)
        print("psi error: ", psi - real_psi_solution)
time2 = time.perf_counter()
print("solver run time: ", time2 - time1)

# plot parametric results
for i,M in enumerate(M_vec):
    plt.loglog(T_vec, np.linalg.norm(psi_error,axis=2)[:,i])
    plt.title("norm of error in psi vs. time T with M=" + str(M))
    plt.figure()
plt.show()

