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

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.quantum_info.operators import Operator
from linear_solvers.matrices.numpy_matrix import NumPyMatrix
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
np.set_printoptions(threshold=np.inf)

# code to implement the Linear Combination of Unitaries method described in the link below
# https://arxiv.org/pdf/1511.02306.pdf

start = time.perf_counter()
# number of variable points in each direction
n_x = 32
n_y = 32

# number of points (including boundary values) in each direction
n_pts_x = n_x + 2
n_pts_y = n_y + 2

# distance between finite difference points
delta_x = 0.5
delta_y = 0.5

# create simple, constant boundary flux values for now, O(N)
top_J_minus = 0
bottom_J_minus = 0
right_J_minus = 0
left_J_minus = 0
top_x_BCs = np.ones(n_pts_x) * top_J_minus
bottom_x_BCs = np.zeros(n_pts_x) * bottom_J_minus
right_y_BCs = np.zeros(n_pts_y) * right_J_minus
left_y_BCs = np.zeros(n_pts_y) * left_J_minus

# material data initialization, O(N)
sigma_a = np.zeros(n_x*n_y)
nu_sigma_f = np.zeros(n_x*n_y)
D = np.zeros(n_x*n_y)
Q = np.zeros(n_x*n_y)

A_mat_size = (n_x) * (n_y)

# from quantum state after passing through algorithm, extract solution to differential equation, O(2^(N+L))
def get_solution_vector(solution, n):
    """Extracts and normalizes simulated state vector
    from LinearSolverResult."""
    solution_vector = Statevector(solution.state).data.real
    solution_length = len(solution_vector)
    solution_vector = solution_vector[int(solution_length/2):int(solution_length/2) + n]
    norm = solution.euclidean_norm
    return norm * solution_vector / np.linalg.norm(solution_vector)

# convert 2D (x,y) index to 1D index
def unroll_index(index_vec):
    return index_vec[0]*n_y + index_vec[1]

# convert 1D index to 2D (x,y) index
def roll_index(index):
    return np.array([math.floor(index/n_y), index % n_y])

# return flux at B.C.s outside of valid index range
def get_BC_value(index):
    i = index[0]
    j = index[1]
    if (i == 0):
        return left_y_BCs[j+1]
    if (i == n_x-1):
        return right_y_BCs[j+1]
    if (j == 0):
        return bottom_x_BCs[i+1]
    if (j == n_y-1):
        return top_x_BCs[i+1]
    raise Exception("tried to get BC on non-boundary node")

# Set material data at each finite difference point, O(N)
def initialize_XSs():
    x_range = (n_pts_x - 1) * delta_x
    y_range = (n_pts_y - 1) * delta_y

    fuel_radius = min(x_range,y_range)/8
    #fuel_radius = 9999

    for i in range(n_x):
        for j in range(n_y):
            x_val = (i + 1) * delta_x - x_range/2
            y_val = (j + 1) * delta_y - y_range/2

            # fuel at center
            '''if (math.sqrt(x_val * x_val + y_val * y_val) < fuel_radius):
                # use fuel XSs
                sigma_a[i * n_y + j] = 4
                nu_sigma_f[i * n_y + j] = 3
                D[i * n_y + j] = 1
                Q[i * n_y + j] = 5
            else:
                # use moderator XSs
                sigma_a[i * n_y + j] = 2
                D[i * n_y + j] = 1'''

            # 4 fuel pins
            if (math.sqrt(math.pow(abs(x_val)-x_range/4,2) + math.pow(abs(y_val)-y_range/4,2)) < fuel_radius):
                # use fuel XSs
                sigma_a[i * n_y + j] = 4
                nu_sigma_f[i * n_y + j] = 3
                D[i * n_y + j] = 1
                Q[i * n_y + j] = 5
            else:
                # use moderator XSs
                sigma_a[i * n_y + j] = 2
                D[i * n_y + j] = 1


# Perform Gram-Schmidt orthogonalization to return V vector from LCU paper.
# The first column is the normalized set of sqrt(alpha) values,
# all other columns' actual values are irrelevant to the algorithm but are set to be
# orthogonal to the alpha vector to ensure V is unitary.
# Function modified from ChatGPT suggestion, O(2^(2L))
def gram_schmidt_ortho(vector):
    num_dimensions = len(vector)
    ortho_basis = np.zeros((num_dimensions, num_dimensions), dtype=float)
    
    
    # Normalize the input vector and add it to the orthogonal basis
    ortho_basis[0] = vector / np.linalg.norm(vector)
    #print(ortho_basis)

    # Gram-Schmidt orthogonalization
    for i in range(1, num_dimensions):
        ortho_vector = np.random.rand(num_dimensions)  # random initialization
        #if(abs(np.log2(i) - np.ceil(np.log2(i))) < 0.00001):
        #    print("dimension: ", i)
        for j in range(i):
            ortho_vector -= np.dot(ortho_basis[j], ortho_vector) / np.dot(ortho_basis[j], ortho_basis[j]) * ortho_basis[j]
        ortho_basis[i] = ortho_vector / np.linalg.norm(ortho_vector)
    
    return ortho_basis.T

# return True if matrix is unitary, False otherwise, O(len(matrix)^2)
def is_unitary(matrix):
    I = matrix.dot(np.conj(matrix).T)
    return I.shape[0] == I.shape[1] and np.allclose(I, np.eye(I.shape[0]))


# get averaged diffusion coefficient in either the "x" or "y" direction for interior points
# lower_index is lower index in the direction of the averaged diffusion coefficient
# set_index is the other dimension index
def get_av_D(direction, lower_index, set_index):
    if direction == "x":
        D_lower = D[unroll_index([lower_index, set_index])]
        D_upper = D[unroll_index([lower_index+1, set_index])]
        delta = delta_x
    elif direction == "y":
        D_lower = D[unroll_index([set_index, lower_index])]
        D_upper = D[unroll_index([set_index, lower_index+1])]
        delta = delta_y
    return 2 * (D_lower/delta) * (D_upper/delta) / (D_lower/delta + D_upper/delta)

def get_edge_D(beta, x_i, y_i, delta):
    return 2 * (beta/2) * (D[unroll_index([x_i, y_i])]/delta) / (beta/2 + (D[unroll_index([x_i, y_i])]/delta))



# use finite volume method to contruct the A matrix reprenting the diffusion equation in the form Ax=b, O(N)
def construct_A_matrix():
    fd_order = 2
    A_matrix = np.zeros((A_mat_size, A_mat_size))
    for x_i in range(n_x):
        for y_i in range(n_y):
            i = unroll_index([x_i, y_i])
            if(x_i == 0): # left BC, normal vector = (-1,0)
                J_x_minus = get_edge_D(0.5, x_i ,y_i,delta_x) * delta_y
            else:
                J_x_minus = get_av_D("x",x_i-1,y_i) * delta_y
                A_matrix[i,unroll_index([x_i-1, y_i])] =  -J_x_minus # (i-1,j) terms
            if(x_i == n_x - 1): # right BC, normal vector = (1,0)
                J_x_plus = get_edge_D(0.5, x_i ,y_i,delta_x) * delta_y
            else:
                J_x_plus = get_av_D("x",x_i,y_i) * delta_y
                A_matrix[i,unroll_index([x_i+1, y_i])] =  -J_x_plus # (i+1,j) terms
            if(y_i == 0): # bottom BC, normal vector = (0,-1)
                J_y_minus = get_edge_D(0.5, x_i ,y_i,delta_y) * delta_x
            else:
                J_y_minus = get_av_D("y",y_i-1,x_i) * delta_x
                A_matrix[i,unroll_index([x_i, y_i-1])] =  -J_y_minus # (i,j-1) terms
            if(y_i == n_y - 1): # right BC, normal vector = (0,1)
                J_y_plus = get_edge_D(0.5, x_i ,y_i,delta_y) * delta_x
            else:
                J_y_plus = get_av_D("x",y_i,x_i) * delta_x
                A_matrix[i,unroll_index([x_i, y_i+1])] =  -J_y_plus # (i,j+1) terms
            A_matrix[i,i] = J_x_minus + J_x_plus + J_y_minus + J_y_plus + (sigma_a[i] - nu_sigma_f[i]) * delta_x * delta_y
    return A_matrix

# return gate to transform 0 state to vector b represented as a quantum state
def get_b_setup_gate(vector, nb):
    if isinstance(vector, list):
        vector = np.array(vector)
    vector_circuit = QuantumCircuit(nb)
    vector_circuit.isometry(
        vector / np.linalg.norm(vector), list(range(nb)), None
    )
    return vector_circuit

# return the U gate in the LCU process as well as a vector of the alpha values.
# Use the Fourier process from the LCU paper to approximate the inverse of A, O(2^L * N^2) ??, need to verify this
def get_fourier_unitaries(J, K, y_max, z_max, matrix, doFullSolution):
    A_mat_size = len(matrix)
    delta_y = y_max / J
    delta_z = z_max / K
    if doFullSolution:
        U = np.zeros((A_mat_size * 2 * J * K, A_mat_size * 2 * J * K), dtype=complex) # matrix from 
        alphas = np.zeros(2 * J * K)
    M = np.zeros((A_mat_size, A_mat_size)) # approximation of inverse of A matrix

    for j in range(J):
        y = j * delta_y
        for k in range(-K,K):
            z = k * delta_z
            alpha_temp = (1) / math.sqrt(2 * math.pi) * delta_y * delta_z * z * math.exp(-z*z/2)
            uni_mat = (1j) * expm(-(1j) * matrix * y * z)
            assert(is_unitary(uni_mat))
            M_temp = alpha_temp * uni_mat
            if doFullSolution:
                if(alpha_temp < 0): # if alpha is negative, incorporate negative phase into U unitary
                    alpha_temp *= -1
                    U[A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1),A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1)] = -1 * uni_mat
                else:
                    alpha_temp *= 1
                    U[A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1),A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1)] = uni_mat
                alphas[2 * j * K + (k + K)] = alpha_temp
            M = M + M_temp

    matrix_invert = np.linalg.inv(matrix)
    error_norm = np.linalg.norm(M - matrix_invert)
    if doFullSolution:
        #print("real matrix inverse: ", matrix_invert)
        #print("probability of algorithm success: ", math.pow(np.linalg.norm(np.matmul(matrix, vector/ np.linalg.norm(vector))),2))
        #print("estimated matrix inverse: ", M)
        #print("Matrix inverse error: ", (M - matrix_invert) / matrix_invert)
        
        #print("norm of inverse error: ", error_norm)

        return U, alphas, error_norm
    return 0, 0, error_norm

# make A matrix
initialize_XSs() # create the vectors holding the material data at each discretized point
A_matrix = construct_A_matrix() # use the material data (like XSs) to make the A matrix for the equation being solved

# make b vector
b_vector = Q * delta_x * delta_y

#print("A matrix:")
#print(A_matrix)
'''print("\n b vector: ")
print(b_vector)
eigenvalues, eigenvectors = np.linalg.eig(A_matrix)
print("A eigenvalues: ", eigenvalues)
print("A condition number: ", max(eigenvalues) / min(eigenvalues))'''

material_initialization_time = time.perf_counter()
print("Initialization Time: ", material_initialization_time - start)

# Do LCU routine (https://arxiv.org/pdf/1511.02306.pdf), equation 18
num_LCU_bits = 3
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
            U, alphas, error_norm = get_fourier_unitaries(J, K, y_max, z_max, quantum_mat, False)
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
    U, alphas, error_norm = get_fourier_unitaries(J, K, y_max, best_z_max, A_matrix, False)
    print("y_max: ", y_max)
    print("Error: ", error_norm)
    if error_norm < best_error_norm:
        best_y_max = y_max
        best_error_norm = error_norm
    else:
        break
best_error_norm = np.inf
for z_max in np.linspace(0.1,5,30):
    U, alphas, error_norm = get_fourier_unitaries(J, K, best_y_max, z_max, A_matrix, False)
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


U, alphas, error_norm = get_fourier_unitaries(pow(2,best_j), pow(2,num_LCU_bits-best_j-1), best_y_max, best_z_max, quantum_mat, True)
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
qc.append(get_b_setup_gate(quantum_b_vector, nb), qb[:])

circuit_setup_time = time.perf_counter()
print("Circuit Setup Time: ", circuit_setup_time - unitary_construction_time)

alpha = np.sum(alphas)

V = gram_schmidt_ortho(np.sqrt(alphas))
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

