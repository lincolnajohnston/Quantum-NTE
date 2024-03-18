import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import transpile, execute
from qiskit.providers.aer import QasmSimulator
from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, expm
import time
import Helpers

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.quantum_info.operators import Operator
from linear_solvers.matrices.numpy_matrix import NumPyMatrix
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
np.set_printoptions(threshold=np.inf)

# code to implement the Linear Combination of Unitaries method described in the link below
# https://arxiv.org/pdf/1511.02306.pdf

start = time.perf_counter()
# number of variable points in each direction
n_x = int(Helpers.input_variable('input.txt', 2))
n_y = int(Helpers.input_variable('input.txt', 2))

# number of points (including boundary values) in each direction
n_pts_x = n_x + 2
n_pts_y = n_y + 2

# distance between finite difference points
delta_x = float(Helpers.input_variable('input.txt', 4))
delta_y = float(Helpers.input_variable('input.txt', 4))

# create boundary conditions
top_I_1 = 0
bottom_I_1 = 0
right_I_1 = 0
left_I_1 = 0
top_x_I_1 = np.ones(n_pts_x) * top_I_1
bottom_x_I_1 = np.zeros(n_pts_x) * bottom_I_1
right_y_I_1 = np.zeros(n_pts_y) * right_I_1
left_y_I_1 = np.zeros(n_pts_y) * left_I_1

top_I_3 = 0
bottom_I_3 = 0
right_I_3 = 0
left_I_3 = 0
top_x_I_3 = np.ones(n_pts_x) * top_I_3
bottom_x_I_3 = np.zeros(n_pts_x) * bottom_I_3
right_y_I_3 = np.zeros(n_pts_y) * right_I_3
left_y_I_3 = np.zeros(n_pts_y) * left_I_3

# material data initialization, O(N)
sigma_t = np.zeros(n_x*n_y)
sigma_s0 = np.zeros(n_x*n_y)
sigma_s2 = np.zeros(n_x*n_y)
nu_sigma_f = np.zeros(n_x*n_y)
D0 = np.zeros(n_x*n_y)
D2 = np.zeros(n_x*n_y)
Q = np.zeros(n_x*n_y)

A_mat_size = 2 * (n_x) * (n_y)

#Choose method, either diffusion or sp3
method = Helpers.input_variable('input.txt', 6)
# make A matrix
Helpers.initialize_XSs(n_pts_x, n_pts_y, delta_x, delta_y, sigma_t, sigma_s0, sigma_s2, nu_sigma_f, D0, D2, Q, n_x, n_y) 
# create the vectors holding the material data at each discretized point
if method == "sp3":
    A_matrix, b_vector = Helpers.sp3_construct_A_matrix(n_x, n_y, A_mat_size, delta_x, delta_y, D0, D2, sigma_t, nu_sigma_f, sigma_s0, sigma_s2, Q, left_y_I_1, right_y_I_1, bottom_x_I_1, top_x_I_1, left_y_I_3, right_y_I_3, bottom_x_I_3, top_x_I_3) # use the material data (like XSs) to make the A matrix for the equation being solved

'''print("A matrix:")
print(A_matrix)
print("\n b vector: ")
print(b_vector)
eigenvalues, eigenvectors = np.linalg.eig(A_matrix)
print("A eigenvalues: ", eigenvalues)
print("A condition number: ", max(eigenvalues) / min(eigenvalues))'''

material_initialization_time = time.perf_counter()
print("Initialization Time: ", material_initialization_time - start)

# Do LCU routine (https://arxiv.org/pdf/1511.02306.pdf), equation 18
num_LCU_bits = 4
num_unitaries = pow(2,num_LCU_bits)
last_error_norm = np.inf

A_mat_size = len(A_matrix)
if(not ishermitian(A_matrix)): # make sure the matrix is hermitian
    quantum_mat = np.zeros((2*A_mat_size,2*A_mat_size))
    quantum_mat[A_mat_size:2*A_mat_size, 0:A_mat_size] = np.conj(A_matrix).T
    quantum_mat[0:A_mat_size, A_mat_size:2*A_mat_size] = A_matrix
    quantum_b_vector = np.zeros(2*len(b_vector))
    quantum_b_vector[0:len(b_vector)] = b_vector
    quantum_b_vector[len(b_vector):2*len(b_vector)] = b_vector
    A_mat_size *= 2
else:
    quantum_mat = A_matrix
    quantum_b_vector = b_vector

# select optimal J, K, y_max, and z_max in just about the least efficient way possible
'''best_j = 0
best_y_max = 0
best_z_max = 0
best_error_norm = np.inf
for j in range(int(num_LCU_bits/4),num_LCU_bits - int(num_LCU_bits/4)):
    J = pow(2,j)
    K = pow(2,num_LCU_bits-j-1)
    for y_max in np.linspace(0.5,5,10):
        for z_max in np.linspace(0.5,5,10):
            U, alphas, error_norm = get_fourier_unitaries(J, K, y_max, z_max, quantum_mat, False, A_mat_size, delta_x, delta_y, delta_z)
            print("J: ", J)
            print("K: ", K)
            print("y_max: ", y_max)
            print("z_max: ", z_max)
            print("Error: ", error_norm)
            if(last_error_norm < error_norm):
                break
            if error_norm < best_error_norm:
                best_j = j
                best_y_max = y_max
                best_z_max = z_max
                best_error_norm = error_norm
            last_error_norm = error_norm'''

# a little quicker way to get best parameters for LCU, but gets worse answers
'''best_j = math.floor(num_LCU_bits/2)
best_y_max = 4
best_z_max = 3
best_error_norm = np.inf
J = pow(2, best_j)
K = pow(2,num_LCU_bits-best_j-1)
for y_max in np.linspace(0.1,6,30):
    U, alphas, error_norm = get_fourier_unitaries(J, K, y_max, best_z_max, A_matrix, False, A_mat_size, delta_x, delta_y, delta_z)
    print("y_max: ", y_max)
    print("Error: ", error_norm)
    if error_norm < best_error_norm:
        best_y_max = y_max
        best_error_norm = error_norm
    else:
        break
best_error_norm = np.inf
for z_max in np.linspace(0.1,5,30):
    U, alphas, error_norm = get_fourier_unitaries(J, K, best_y_max, z_max, A_matrix, False, A_mat_size, delta_x, delta_y, delta_z)
    print("z_max: ", z_max)
    print("Error: ", error_norm)
    if(last_error_norm < error_norm):
        break
    if error_norm < best_error_norm:
        best_z_max = z_max
        best_error_norm = error_norm
    else:
        break'''


# manually input paremters for LCU
best_j = 1
best_y_max = 1
best_z_max = 1

U, alphas, error_norm = Helpers.get_fourier_unitaries(pow(2,best_j), pow(2,num_LCU_bits-best_j-1), best_y_max, best_z_max, quantum_mat, True, A_mat_size)
print("Error Norm: ", error_norm)

unitary_construction_time = time.perf_counter()
print("Unitary Construction Time: ", unitary_construction_time - material_initialization_time)

# Initialise the quantum registers
nb = int(np.log2(len(quantum_b_vector)))
qb = QuantumRegister(nb)  # right hand side and solution
ql = QuantumRegister(num_LCU_bits)  # LCU ancilla zero bits
cl = ClassicalRegister(num_LCU_bits)  # right hand side and solution

qc = QuantumCircuit(qb, ql)

# b vector State preparation
qc.append(Helpers.get_b_setup_gate(quantum_b_vector, nb), qb[:])

circuit_setup_time = time.perf_counter()
print("Circuit Setup Time: ", circuit_setup_time - unitary_construction_time)

alpha = np.sum(alphas)

V = Helpers.gram_schmidt_ortho(np.sqrt(alphas))
v_mat_time = time.perf_counter()
print("Construction of V matrix time: ", v_mat_time - circuit_setup_time)

op_time = time.perf_counter()
print("Operator Construction Time: ", op_time - v_mat_time)

V_gate = UnitaryGate(V, 'V', False)
U_gate = UnitaryGate(U, 'U', False)
V_inv_gate = UnitaryGate(np.conj(V).T, 'V_inv', False)

qc.append(V_gate, ql[:])
qc.append(U_gate, qb[:] + ql[:])
qc.append(V_inv_gate, ql[:])

gate_time = time.perf_counter()
print("Gate U and V Application Time: ", gate_time - op_time)

qc.save_statevector()


# Run quantum algorithm
backend = QasmSimulator(method="statevector")
job = execute(qc, backend)
job_result = job.result()
state_vec = job_result.get_statevector(qc).data
#print(state_vec[0:A_mat_size])
state_vec = np.real(state_vec[len(quantum_b_vector) - len(b_vector):len(quantum_b_vector)])
classical_sol_vec = np.linalg.solve(A_matrix, b_vector)
print(classical_sol_vec)

# find scalar flux value from 0th and 2nd moments in SP3 equations
for i in range(int(len(state_vec)/2)):
    state_vec[i] -= 2 * state_vec[int(i + len(state_vec)/2)]
    classical_sol_vec[i] -= 2 * classical_sol_vec[int(i + len(classical_sol_vec)/2)]
state_vec = state_vec[:int(len(state_vec)/2)]
classical_sol_vec = classical_sol_vec[:int(len(classical_sol_vec)/2)]

state_vec = state_vec * np.linalg.norm(classical_sol_vec) / np.linalg.norm(state_vec) # scale result to match true answer


# Print results
print("quantum solution estimate: ", state_vec)
#print("expected quantum solution: ", np.matmul(M, vector))

print('classical solution vector:          ', classical_sol_vec)

sol_rel_error = (state_vec - classical_sol_vec) / classical_sol_vec
#print("Relative solution error: ", sol_rel_error)

sol_error = state_vec - classical_sol_vec
#print("Solution error: ", sol_error)

solve_time = time.perf_counter()
print("Circuit Solve Time: ", solve_time - gate_time)
print("Total time: ", solve_time - start)


# Make graphs of results
state_vec.resize((n_x,n_y))
ax = sns.heatmap(state_vec, linewidth=0.5)
plt.title("Quantum Solution")
plt.savefig('q_sol.png')
plt.figure()

classical_sol_vec.resize((n_x,n_y))
ax = sns.heatmap(classical_sol_vec, linewidth=0.5)
plt.title("Real Solution")
plt.savefig('real_sol.png')
plt.figure()

sol_rel_error.resize((n_x,n_y))
ax = sns.heatmap(sol_rel_error, linewidth=0.5)
plt.title("Relative error between quantum and real solution")
plt.savefig('rel_error.png')
plt.figure()

sol_error.resize((n_x,n_y))
ax = sns.heatmap(sol_error, linewidth=0.5)
plt.title("Actual error between quantum and real solution")
plt.savefig('error.png')
plt.show()