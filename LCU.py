import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import transpile, execute
from qiskit.providers.aer import QasmSimulator
from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, expm

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister


from qiskit.quantum_info.operators import Operator
from linear_solvers.matrices.numpy_matrix import NumPyMatrix
np.set_printoptions(threshold=np.inf)

# code to implement the Linear Combination of Unitaries method described in the link below
# https://arxiv.org/pdf/1511.02306.pdf

# number of variable points in each direction
n_x = 2
n_y = 2

# number of points (including boundary values) in each direction
n_pts_x = n_x + 2
n_pts_y = n_y + 2

# distance between finite difference points
delta_x = 0.5
delta_y = 0.5

# create simple, constant boundary flux values for now
top_flux = 0.5
bottom_flux = 1.5
right_flux = 2.5
left_flux = 3.5
top_x_BCs = np.ones(n_pts_x) * top_flux
bottom_x_BCs = np.zeros(n_pts_x) * bottom_flux
right_y_BCs = np.zeros(n_pts_y) * right_flux
left_y_BCs = np.zeros(n_pts_y) * left_flux

# material data initialization
sigma_a = np.zeros(n_x*n_y)
nu_sigma_f = np.zeros(n_x*n_y)
D = np.zeros(n_x*n_y)
Q = np.zeros(n_x*n_y)

A_mat_size = (n_x) * (n_y)

# from quantum state after passing through algorithm, extract solution to differential equation
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
def get_BC_flux(index):
    i = index[0]
    j = index[1]
    if (i == -1):
        return left_y_BCs[j+1]
    if (i == n_x):
        return right_y_BCs[j+1]
    if (j == -1):
        return bottom_x_BCs[i+1]
    if (j == n_y):
        return top_x_BCs[i+1]
    raise Exception("tried to get BC on non-boundary node")

# Set material data at each finite difference point
def initialize_XSs():
    x_range = (n_pts_x - 1) * delta_x
    y_range = (n_pts_y - 1) * delta_y

    fuel_radius = min(x_range,y_range)/3

    for i in range(n_x):
        for j in range(n_y):
            x_val = (i + 1) * delta_x - x_range/2
            y_val = (j + 1) * delta_y - y_range/2
            if (math.sqrt(x_val * x_val + y_val * y_val) < fuel_radius):
                # use fuel XSs
                sigma_a[i * n_y + j] = 4
                nu_sigma_f[i * n_y + j] = 3
                D[i * n_y + j] = 1
                Q[i * n_y + j] = 5
            else:
                # use moderator XSs
                sigma_a[i * n_y + j] = 2
                D[i * n_y + j] = 1

# return average of diffusion coefficient between two points (defined with 1D index)
def get_mean_D(p1, p2):
    return (D[p1] + D[p2])/2

# Perform Gram-Schmidt orthogonalization to return V vector from LCU paper.
# The first column is the normalized set of sqrt(alpha) values,
# all other columns' actual values are irrelevant to the algorithm but are set to be
# orthogonal to the alpha vector to ensure V is unitary.
# Function modified from ChatGPT suggestion
def gram_schmidt_ortho(vector):
    num_dimensions = len(vector)
    ortho_basis = np.zeros((num_dimensions, num_dimensions), dtype=float)
    
    
    # Normalize the input vector and add it to the orthogonal basis
    ortho_basis[0] = vector / np.linalg.norm(vector)
    #print(ortho_basis)

    # Gram-Schmidt orthogonalization
    for i in range(1, num_dimensions):
        ortho_vector = np.random.rand(num_dimensions)  # random initialization
        if(abs(np.log2(i) - np.ceil(np.log2(i))) < 0.00001):
            print("dimension: ", i)
        for j in range(i):
            ortho_vector -= np.dot(ortho_basis[j], ortho_vector) / np.dot(ortho_basis[j], ortho_basis[j]) * ortho_basis[j]
        ortho_basis[i] = ortho_vector / np.linalg.norm(ortho_vector)
    
    return ortho_basis.T

# return True if matrix is unitary, False otherwise
def is_unitary(matrix):
    I = matrix.dot(np.conj(matrix).T)
    return I.shape[0] == I.shape[1] and np.allclose(I, np.eye(I.shape[0]))

# use finite difference method to contruct the A matrix reprenting the diffusion equation in the form Ax=b
def construct_A_matrix():
    A_matrix = np.zeros((A_mat_size, A_mat_size))
    for i in range(A_mat_size):
        A_matrix[i,i] = (2*D[i]/(delta_x*delta_x) + 2*D[i]/(delta_y*delta_y) + sigma_a[i] - nu_sigma_f[i])
        current_index = roll_index(i)
        if(current_index[0] > 0):
            A_matrix[i,unroll_index(current_index + np.array([-1, 0]))] = -get_mean_D(i,unroll_index(current_index + np.array([-1, 0]))) / (delta_x*delta_x) # (i-1,j) term
        else:
            b_vector[i] += D[i] / (delta_x*delta_x) * get_BC_flux(current_index + np.array([-1, 0]))
        if(current_index[0] < n_x - 1):
            A_matrix[i,unroll_index(current_index + np.array([1, 0]))] = -get_mean_D(i,unroll_index(current_index + np.array([1, 0]))) / (delta_x*delta_x) # (i+1,j) term
        else:
            b_vector[i] += D[i] / (delta_x*delta_x) * get_BC_flux(current_index + np.array([1, 0]))
        if(current_index[1] > 0):
            A_matrix[i,unroll_index(current_index + np.array([0, -1]))] = -get_mean_D(i,unroll_index(current_index + np.array([0, -1]))) / (delta_y*delta_y) # (i,j-1) term
        else:
            b_vector[i] += D[i] / (delta_y*delta_y) * get_BC_flux(current_index + np.array([0, -1]))
        if(current_index[1] < n_y - 1):
            A_matrix[i,unroll_index(current_index + np.array([0, 1]))] = -get_mean_D(i,unroll_index(current_index + np.array([0, 1]))) / (delta_y*delta_y) # (i,j+1) term
        else:
            b_vector[i] += D[i] / (delta_y*delta_y) * get_BC_flux(current_index + np.array([0, 1]))
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
# Use the Fourier process from the LCU paper to approximate the inverse of A
def get_fourier_unitaries(J, K, matrix):
    A_mat_size = len(matrix)
    y_max = 3
    z_max = 3
    delta_y = y_max / J
    delta_z = z_max / K
    U = np.zeros((A_mat_size * 2 * J * K, A_mat_size * 2 * J * K), dtype=complex) # matrix from 
    alphas = np.zeros(2 * J * K)

    for j in range(J):
        print("J: ", j)
        y = j * delta_y
        for k in range(-K,K):
            z = k * delta_z
            alpha_temp = (1) / math.sqrt(2 * math.pi) * delta_y * delta_z * z * math.exp(-z*z/2)
            uni_mat = (1j) * expm(-(1j) * matrix * y * z)
            assert(is_unitary(uni_mat))
            #M_temp = alpha_temp * uni_mat
            if(alpha_temp < 0): # if alpha is negative, incorporate negative phase into U unitary
                alpha_temp *= -1
                U[A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1),A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1)] = -1 * uni_mat
            else:
                alpha_temp *= 1
                U[A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1),A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1)] = uni_mat
            alphas[2 * j * K + (k + K)] = alpha_temp
            #M = M + M_temp
    alphas = np.sqrt(alphas)
    return U, alphas

# make b vector
b_vector = Q

# make A matrix
initialize_XSs() # create the vectors holding the material data at each discretized point
A_matrix = construct_A_matrix() # use the material data (like XSs) to make the A matrix for the equation being solved

print("A matrix:")
print(A_matrix)
print("\n b vector: ")
print(b_vector)
eigenvalues, eigenvectors = np.linalg.eig(A_matrix)
print("A eigenvalues: ", eigenvalues)
print("A condition number: ", max(eigenvalues) / min(eigenvalues))

# Do LCU routine (https://arxiv.org/pdf/1511.02306.pdf), equation 18
J = 64 # number of discrete points in the y direction for approximating A^(-1)
K = 16 # half the number of discrete points in the z direction for approximating A^(-1)
num_unitaries = 2 * J * K # number of separate unitaries used to approximate the inverse of the A matrix
num_LCU_bits = math.ceil(np.log2(num_unitaries))
nb = int(np.log2(len(b_vector)))

# Initialise the quantum registers
qb = QuantumRegister(nb)  # right hand side and solution
ql = QuantumRegister(num_LCU_bits)  # LCU ancilla zero bits
cl = ClassicalRegister(num_LCU_bits)  # right hand side and solution

qc = QuantumCircuit(qb, ql)

# b vector State preparation
qc.append(get_b_setup_gate(b_vector, nb), qb[:])

#M = np.zeros((A_mat_size, A_mat_size)) # approximation of inverse of A matrix
U, alphas = get_fourier_unitaries(J, K, A_matrix)
alpha = np.dot(alphas, alphas)

'''A_matrix_invert = np.linalg.inv(A_matrix)
print(A_matrix_invert)
print("probability of algorithm success: ", math.pow(np.linalg.norm(np.matmul(A_matrix, vector/ np.linalg.norm(vector))),2))
print(M)
print("A_Matrix inverse error: ", (M - A_matrix_invert) / A_matrix_invert)
print("norm of diff A_matrix: ", np.linalg.norm(M - A_matrix_invert))'''

V = gram_schmidt_ortho(alphas)

# check if operator matrices are unitary
#print("V is unitary: ", is_unitary(V))
#print("U is unitary: ", is_unitary(U))

V_op = Operator(V)
U_op = Operator(U)
V_inv_op = Operator(np.conj(V).T)


qc.unitary(V_op, ql[:], label='V') # further to the right registers are the more major ones (ql has most major qubits here)
qc.unitary(U_op, qb[:] + ql[:], label='U')
qc.unitary(V_inv_op, ql[:], label='V_inv')

qc.save_statevector()

backend = QasmSimulator(method="statevector")
job = execute(qc, backend)
job_result = job.result()
state_vec = job_result.get_statevector(qc).data
print(state_vec[0:A_mat_size])
state_vec = np.absolute(state_vec[0:A_mat_size])
state_vec = state_vec * math.sqrt(np.dot(b_vector, b_vector)) * alpha
print("quantum solution estimate: ", state_vec)

#print("expected quantum solution: ", np.matmul(M, vector))

classical_sol_vec = np.linalg.solve(A_matrix, b_vector)
print('classical solution vector:          ', classical_sol_vec)

