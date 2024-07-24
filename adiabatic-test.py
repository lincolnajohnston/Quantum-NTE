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
    
n_bits = 1 
T = 100
M = 10000
dt = T/M
X = np.array([[0, 1],[1, 0]])
H_B = np.array([[1/2, -1/2],[-1/2, 1/2]]) # one qubit case
#H_B = np.array(np.eye(4) - (1/2) * np.kron(X, np.eye(2))- (1/2) * np.kron(np.eye(2), X)) # two qubit case
H_B_eigenvalues, H_B_eigenvectors = np.linalg.eig(H_B)
print("H_B eigenvalues: ", H_B_eigenvalues)
print("H_B eigenvectors: ", H_B_eigenvectors)
H_P = np.array([[1, 0],[0, 0]]) # one qubit case
#H_P = np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]) # 2 qubit case
H_P_eigenvalues, H_P_eigenvectors = np.linalg.eig(H_P)
print("H_P eigenvalues: ", H_P_eigenvalues)
print("H_P eigenvectors: ", H_P_eigenvectors)
dH = H_P - H_B
_, eigenvectors = np.linalg.eig(H_B)
psi = eigenvectors[0]
#psi = np.array(np.ones(int(math.pow(2,n_bits)))) * 1/math.sqrt(math.pow(2,n_bits))
print("psi = ", psi)
print("delta-t * delta-H = ", np.linalg.norm(dt * (H_P - H_B)))

state_evolution = np.zeros((M,int(math.pow(2,n_bits))),dtype=np.complex_)
expected_state_evolution = np.zeros((M,int(math.pow(2,n_bits))),dtype=np.complex_)

lastH = H_B
U_T = np.eye(2)
for l in range(M):
    s = l/M
    H = (1-s) * H_B + s * H_P
    U = expm(-(1j) * dt * H)
    U_T = np.matmul(U,U_T)
    psi = U.dot(psi)
    state_evolution[l,:] = psi
    expected_state_evolution[l,:] = [1,-s/(s-1)-math.sqrt(1-2*s*(1-s))/(s-1)]
    expected_state_evolution[l,:] /= np.linalg.norm(expected_state_evolution[l,:])
    print(", delta-t * delta-H = ", np.linalg.norm(dt * (lastH - H)), ", 1/M = ", 1/M)
    print("l = ", l, " psi = ", np.round(psi, decimals=8))
    lastH = H

print("U_T", U_T)

theta = 2 * math.acos(abs(psi[0]))
print("theta: ", theta)
gamma = math.atan2(psi[0].imag,psi[0].real)
gamma_plus_phi = math.atan2(psi[1].imag,psi[1].real)
phi = gamma_plus_phi - gamma
print("gamma: ", gamma)
print("phi: ", phi)
print("zero state: ", cmath.exp(1j*gamma)*math.cos(theta/2))
print("one state: ", cmath.exp(1j*gamma_plus_phi)*math.sin(theta/2))

start_index = 0
end_index = 1000
colors = ['-.r','-.b','-.g','-.c','-.m','-.y','-.k','-.w',]

for i in range(int(math.pow(2,n_bits))):
    x = state_evolution[start_index:end_index,i].real
    y = state_evolution[start_index:end_index,i].imag
    plt.plot(x,y,colors[i])
plt.title('actual state evolution')
plt.figure()

for i in range(int(math.pow(2,n_bits))):
    x = expected_state_evolution[start_index:end_index,i].real
    y = expected_state_evolution[start_index:end_index,i].imag
    plt.plot(x,y,colors[i])
plt.title('expected state evolution')
plt.show()