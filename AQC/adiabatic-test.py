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
import time

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate

# Testing the simple problems from https://arxiv.org/pdf/quant-ph/0001106 using their method of applying adiabatic methods to gate-based systems

# get the incremental U unitary operator to apply to adiabatic evolution of a quantum state
def getU(l, T, M, dH):
    assert(l<=M)
    dt = T/M

def get_H_B(i,n):
    H_B = 1/2 * np.array([[1 , -1],[-1, 1]])
    return_vec = np.eye(int(math.pow(2,i-1)))
    return_vec = np.kron(return_vec,H_B)
    return_vec = np.kron(return_vec,np.eye(int(math.pow(2,n-i))))
    return return_vec
    

# setup section
n_bits = 3
T = 100
M = 10000
X = np.array([[0, 1],[1, 0]])
#H_B = np.array([[1/2, -1/2],[-1/2, 1/2]]) # one qubit case
#H_P = np.array([[1, 0],[0, 0]]) # one qubit case
#H_B = np.array(np.eye(4) - (1/2) * np.kron(X, np.eye(2))- (1/2) * np.kron(np.eye(2), X)) # two qubit case
#H_B = get_H_B(1,2) + get_H_B(2,2)
#H_P = np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]) # 2 qubit case
H_B = get_H_B(1,3) + get_H_B(2,3) + get_H_B(1,3) + get_H_B(3,3) + get_H_B(2,3) + get_H_B(3,3) # 3 qubit case
H_P = np.diag([1,1,2,0,1,3,1,1]) # 3 qubit case
#H_B = get_H_B(1,3) + get_H_B(2,3) + get_H_B(3,3) # simpler form of 3 qubit case (3-SAT instead of 2-SAT with 3 qubits)
#H_P = np.diag([1,1,1,0,1,1,1,1]) # simpler form of 3 qubit case (3-SAT instead of 2-SAT with 3 qubits)


# print eigenvalues of H_B and H_P
print("H_B", H_B)
H_B_eigenvalues, H_B_eigenvectors = np.linalg.eig(H_B)
print("H_B eigenvalues: ", H_B_eigenvalues)
print("H_B eigenvectors: ", H_B_eigenvectors)
print("H_P", H_P)
H_P_eigenvalues, H_P_eigenvectors = np.linalg.eig(H_P)
print("H_P eigenvalues: ", H_P_eigenvalues)
print("H_P eigenvectors: ", H_P_eigenvectors)
H_B_eig, eigenvectors = np.linalg.eig(H_B)

# put psi in the ground state of H_B
min_eig_id = H_B_eig.tolist().index(min(H_B_eig))
psi = eigenvectors[:,min_eig_id] # put this in the ground state
#psi = np.array(np.ones(int(math.pow(2,n_bits)))) * 1/math.sqrt(math.pow(2,n_bits))


dt = T/M
dH = H_P - H_B
print("psi = ", psi)
print("delta-t * delta-H = ", np.linalg.norm(dt * (H_P - H_B)))

state_evolution = np.zeros((M,int(math.pow(2,n_bits))),dtype=np.complex_)
expected_state_evolution = np.zeros((M,int(math.pow(2,n_bits))),dtype=np.complex_)

lastH = H_B
U_T = np.eye(int(math.pow(2,n_bits)))
for l in range(M):
    s = l/M
    H = (1-s) * H_B + s * H_P

    # using equation 5.4 in Farhi
    U = expm(-(1j) * dt * H)
    U_T = np.matmul(U,U_T)
    psi = U.dot(psi)
    state_evolution[l,:] = psi
    #expected_state_evolution[l,:] = [1,-s/(s-1)-math.sqrt(1-2*s*(1-s))/(s-1)] # one qubit case
    #expected_state_evolution[l,:] = [0,0,0,0]
    #expected_state_evolution[l,:] /= np.linalg.norm(expected_state_evolution[l,:])
    if (l % 100 == 0):
        print(", delta-t * delta-H = ", np.linalg.norm(dt * (lastH - H)), ", 1/M = ", 1/M)
        print("l = ", l, " psi = ", np.round(psi, decimals=8))
        print("expected psi: ", expected_state_evolution[l,:])

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
    #expected_state_evolution[l,:] = [1,-s/(s-1)-math.sqrt(1-2*s*(1-s))/(s-1)]
    #expected_state_evolution[l,:] /= np.linalg.norm(expected_state_evolution[l,:])

print("U_T", U_T)

print("psi: ", psi)

# for one qubit case
'''theta = 2 * math.acos(abs(psi[0]))
print("theta: ", theta)
gamma = math.atan2(psi[0].imag,psi[0].real)
gamma_plus_phi = math.atan2(psi[1].imag,psi[1].real)
phi = gamma_plus_phi - gamma
print("gamma: ", gamma)
print("phi: ", phi)
print("zero state: ", cmath.exp(1j*gamma)*math.cos(theta/2))
print("one state: ", cmath.exp(1j*gamma_plus_phi)*math.sin(theta/2))'''

start_index = 0
end_index = 10000
colors = ['-.r','-.b','-.g','-.c','-.m','-.y','-.k','-.w',]
legend_vec = [str(format(i,'b')) + " state" for i in range(int(math.pow(2,n_bits)))]

# all basis state magnitudes on a single plot
'''for i in range(int(math.pow(2,n_bits))):
    x = state_evolution[start_index:end_index,i].real
    y = state_evolution[start_index:end_index,i].imag
    plt.plot(x,y,colors[i])
plt.legend(legend_vec)
plt.title('actual state evolution')
plt.figure()'''

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
plt.figure()

for i in range(int(math.pow(2,n_bits))):
    x = np.abs(state_evolution[start_index:end_index,i])
    y = state_evolution[start_index:end_index,i] * 0
    plt.plot(x,y,colors[i])
plt.legend(legend_vec)
plt.title('magnitude of actual state evolution')
plt.figure()

# plot expected eigenvectors throughout the evolution if that is known
'''for i in range(int(math.pow(2,n_bits))):
    x = expected_state_evolution[start_index:end_index,i].real
    y = expected_state_evolution[start_index:end_index,i].imag
    plt.plot(x,y,colors[i])
plt.legend(legend_vec)
plt.title('expected state evolution')
plt.figure()

for i in range(int(math.pow(2,n_bits))):
    x = np.abs(state_evolution[start_index:end_index,i]) - np.abs(expected_state_evolution[start_index:end_index,i])
    plt.plot(x,colors[i])
plt.legend(['0 state', '1 state'])
plt.title('difference between magnitude of actual state evolution and expected')'''
plt.show()


