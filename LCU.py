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
n_x = 16
n_y = 16

# number of points (including boundary values) in each direction
n_pts_x = n_x + 2
n_pts_y = n_y + 2

# distance between finite difference points
delta_x = 0.5
delta_y = 0.5

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

def get_I_1_value(index):
    i = index[0]
    j = index[1]
    if (i == 0):
        return left_y_I_1[j]
    if (i == n_x-1):
        return right_y_I_1[j]
    if (j == 0):
        return bottom_x_I_1[i]
    if (j == n_y-1):
        return top_x_I_1[i]
    raise Exception("tried to get BC on non-boundary node")

# return flux at B.C.s at edge of problem domain
def get_I_3_value(index):
    i = index[0]
    j = index[1]
    if (i == 0):
        return left_y_I_3[j]
    if (i == n_x-1):
        return right_y_I_3[j]
    if (j == 0):
        return bottom_x_I_3[i]
    if (j == n_y-1):
        return top_x_I_3[i]
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
                sigma_t[i * n_y + j] = 5
                sigma_s0[i * n_y + j] = 4
                sigma_s2[i * n_y + j] = 0.1
                nu_sigma_f[i * n_y + j] = 3
                D0[i * n_y + j] = 1
                D2[i * n_y + j] = 2
                Q[i * n_y + j] = 5
            else:
                # use moderator XSs
                sigma_t[i * n_y + j] = 5
                sigma_s0[i * n_y + j] = 5
                sigma_s2[i * n_y + j] = 2
                D0[i * n_y + j] = 1
                D2[i * n_y + j] = 2'''

            # 4 fuel pins
            if (math.sqrt(math.pow(abs(x_val)-x_range/4,2) + math.pow(abs(y_val)-y_range/4,2)) < fuel_radius):
                # use fuel XSs

                sigma_t[i * n_y + j] = 5
                sigma_s0[i * n_y + j] = 4
                sigma_s2[i * n_y + j] = 0.1
                nu_sigma_f[i * n_y + j] = 0
                D0[i * n_y + j] = 1
                D2[i * n_y + j] = 2
                Q[i * n_y + j] = 5
            else:
                # use moderator XSs
                sigma_t[i * n_y + j] = 5
                sigma_s0[i * n_y + j] = 5
                sigma_s2[i * n_y + j] = 2
                D0[i * n_y + j] = 1
                D2[i * n_y + j] = 2


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
def get_av_D0(direction, lower_index, set_index):
    if direction == "x":
        D_lower = D0[unroll_index([lower_index, set_index])]
        D_upper = D0[unroll_index([lower_index+1, set_index])]
        delta = delta_x
    elif direction == "y":
        D_lower = D0[unroll_index([set_index, lower_index])]
        D_upper = D0[unroll_index([set_index, lower_index+1])]
        delta = delta_y
    return 2 * (D_lower/delta) * (D_upper/delta) / (D_lower/delta + D_upper/delta)

def get_av_D2(direction, lower_index, set_index):
    if direction == "x":
        D_lower = D2[unroll_index([lower_index, set_index])]
        D_upper = D2[unroll_index([lower_index+1, set_index])]
        delta = delta_x
    elif direction == "y":
        D_lower = D2[unroll_index([set_index, lower_index])]
        D_upper = D2[unroll_index([set_index, lower_index+1])]
        delta = delta_y
    return 2 * (D_lower/delta) * (D_upper/delta) / (D_lower/delta + D_upper/delta)

#def get_edge_D(beta, x_i, y_i, delta):
#    return 2 * (beta/2) * (D[unroll_index([x_i, y_i])]/delta) / (beta/2 + (D[unroll_index([x_i, y_i])]/delta))



# use finite volume method to contruct the A matrix reprenting the diffusion equation in the form Ax=b, O(N)
def construct_A_matrix():
    fd_order = 2
    beta = 0.5
    phi_2_offset = n_x * n_y
    A_matrix = np.zeros((A_mat_size, A_mat_size))
    b_vector = np.zeros((A_mat_size))
    for x_i in range(n_x):
        for y_i in range(n_y):
            i = unroll_index([x_i, y_i])
            if(x_i == 0): # left BC, normal vector = (-1,0)
                # these coefficients will be the same for all cells with the same material and mesh size so
                # need to imporove efficiency of this by storing these values beforehand instead of recalculating for each B.C. cell
                a1 = (1 + 4 * D0[unroll_index([x_i, y_i])]/delta_x)
                a2 = (-3/4) * D0[unroll_index([x_i, y_i])]/D2[unroll_index([x_i, y_i])]
                a3 = 2 * D0[unroll_index([x_i, y_i])]/delta_x
                a4 = (-3/4) * 2 * D0[unroll_index([x_i, y_i])] / delta_x
                a5 =  4 * D0[unroll_index([x_i, y_i])]/delta_x * 2 * get_I_1_value([x_i,y_i])

                b2 = (1 + (80/21) * D2[unroll_index([x_i, y_i])]/delta_x)
                b1 = (-1/7) * D2[unroll_index([x_i, y_i])]/D0[unroll_index([x_i, y_i])]
                b4 = 2 * D2[unroll_index([x_i, y_i])]/delta_x
                b3 = (-2/7) * D2[unroll_index([x_i, y_i])] / delta_x
                b5 =  (6/5) * (80/21) * D2[unroll_index([x_i, y_i])]/delta_x * get_I_3_value([x_i,y_i])

                denom = (a1 - a2 * b1 / b2)
                c1 = (a5 - a2 * b5 / b2) / denom
                c2 = (a2 * b3 / b2 - a3) / denom
                c3 = (a2 * b4 / b2 - a4) / denom

                b_vector[i] += c1 * delta_y
                A_matrix[i,i] -= c2 * delta_y
                A_matrix[i,i+phi_2_offset] -= c3 * delta_y

                b_vector[i + phi_2_offset] += (b5 - b1 * c1) / b2 * delta_y
                A_matrix[i+phi_2_offset,i] -= (-b1 * c2 - b3) / b2 * delta_y
                A_matrix[i+phi_2_offset,i+phi_2_offset] -= (-b1 * c3 - b4) / b2 * delta_y
            else:
                # Phi_0 equations
                x_minus_term_0 = get_av_D0("x",x_i-1,y_i) * delta_y
                A_matrix[i,unroll_index([x_i-1, y_i])] =  -x_minus_term_0 # phi_0, (i-1,j) term
                A_matrix[i,i] +=  x_minus_term_0 # phi_0, (i,j) term

                # Phi_2 equations
                x_minus_term_2 = get_av_D2("x",x_i-1,y_i) * delta_y
                A_matrix[i+phi_2_offset,unroll_index([x_i-1, y_i]) + phi_2_offset] =  -x_minus_term_2 # phi_2, (i-1,j) term
                A_matrix[i+phi_2_offset,i+phi_2_offset] +=  x_minus_term_2 # phi_2, (i,j) term
            if(x_i == n_x - 1): # right BC, normal vector = (1,0)
                d1 = (-1 - 4 * D0[unroll_index([x_i, y_i])]/delta_x)
                d2 = (3/4) * D0[unroll_index([x_i, y_i])]/D2[unroll_index([x_i, y_i])]
                d3 = 2 * D0[unroll_index([x_i, y_i])]/delta_x
                d4 = (-3/4) * 2 * D0[unroll_index([x_i, y_i])] / delta_x
                d5 =  4 * D0[unroll_index([x_i, y_i])]/delta_x * 2 * get_I_1_value([x_i,y_i])

                e2 = (-1 - (80/21) * D2[unroll_index([x_i, y_i])]/delta_x)
                e1 = (1/7) * D2[unroll_index([x_i, y_i])]/D0[unroll_index([x_i, y_i])]
                e4 = 2 * D2[unroll_index([x_i, y_i])]/delta_x
                e3 = (-2/7) * D2[unroll_index([x_i, y_i])] / delta_x
                e5 =  (6/5) * (80/21) * D2[unroll_index([x_i, y_i])]/delta_x * get_I_3_value([x_i,y_i])

                denom = (d1 - d2 * e1 / e2)
                f1 = (d5 - d2 * e5 / e2) / denom
                f2 = (d2 * e3 / e2 - d3) / denom
                f3 = (d2 * e4 / e2 - d4) / denom

                b_vector[i] -= f1 * delta_y
                A_matrix[i,i] += f2 * delta_y
                A_matrix[i,i+phi_2_offset] += f3 * delta_y

                b_vector[i + phi_2_offset] -= (e5 - e1 * f1) / e2 * delta_y
                A_matrix[i+phi_2_offset,i] += (-e1 * f2 - e3) / e2 * delta_y
                A_matrix[i+phi_2_offset,i+phi_2_offset] += (-e1 * f3 - e4) / e2 * delta_y
            else:
                # Phi_0 equations
                x_plus_term_0 = get_av_D0("x",x_i,y_i) * delta_y
                A_matrix[i,unroll_index([x_i+1, y_i])] =  -x_plus_term_0 # phi_0, (i+1,j) term
                A_matrix[i,i] +=  x_plus_term_0 # phi_0, (i,j) term

                # Phi_2 equations
                x_plus_term_2 = get_av_D2("x",x_i,y_i) * delta_y
                A_matrix[i+phi_2_offset,unroll_index([x_i+1, y_i]) + phi_2_offset] =  -x_plus_term_2 # phi_2, (i+1,j) term
                A_matrix[i+phi_2_offset,i+phi_2_offset] +=  x_plus_term_2 # phi_2, (i,j) term
            if(y_i == 0): # bottom BC, normal vector = (0,-1)
                a1 = (1 + 4 * D0[unroll_index([x_i, y_i])]/delta_y)
                a2 = (-3/4) * D0[unroll_index([x_i, y_i])]/D2[unroll_index([x_i, y_i])]
                a3 = 2 * D0[unroll_index([x_i, y_i])]/delta_y
                a4 = (-3/4) * 2 * D0[unroll_index([x_i, y_i])] / delta_y
                a5 =  4 * D0[unroll_index([x_i, y_i])]/delta_y * 2 * get_I_1_value([x_i,y_i])

                b2 = (1 + (80/21) * D2[unroll_index([x_i, y_i])]/delta_y)
                b1 = (-1/7) * D2[unroll_index([x_i, y_i])]/D0[unroll_index([x_i, y_i])]
                b4 = 2 * D2[unroll_index([x_i, y_i])]/delta_y
                b3 = (-2/7) * D2[unroll_index([x_i, y_i])] / delta_y
                b5 =  (6/5) * (80/21) * D2[unroll_index([x_i, y_i])]/delta_y * 2 * get_I_3_value([x_i,y_i])

                denom = (a1 - a2 * b1 / b2)
                c1 = (a5 - a2 * b5 / b2) / denom
                c2 = (a2 * b3 / b2 - a3) / denom
                c3 = (a2 * b4 / b2 - a4) / denom

                b_vector[i] += c1 * delta_x
                A_matrix[i,i] -= c2 * delta_x
                A_matrix[i,i+phi_2_offset] -= c3 * delta_x

                b_vector[i + phi_2_offset] += (b5 - b1 * c1) / b2 * delta_x
                A_matrix[i+phi_2_offset,i] -= (-b1 * c2 - b3) / b2 * delta_x
                A_matrix[i+phi_2_offset,i+phi_2_offset] -= (-b1 * c3 - b4) / b2 * delta_x
            else:
                # Phi_0 equations
                y_minus_term_0 = get_av_D0("y",x_i,y_i-1) * delta_x
                A_matrix[i,unroll_index([x_i, y_i-1])] =  -y_minus_term_0 # phi_0, (i,j-1) term
                A_matrix[i,i] +=  y_minus_term_0 # phi_0, (i,j) term

                # Phi_2 equations
                y_minus_term_2 = get_av_D2("y",x_i,y_i-1) * delta_x
                A_matrix[i+phi_2_offset,unroll_index([x_i, y_i-1]) + phi_2_offset] =  -y_minus_term_2 # phi_2, (i,j-1) term
                A_matrix[i+phi_2_offset,i+phi_2_offset] +=  y_minus_term_2 # phi_2, (i,j) term
            if(y_i == n_y - 1): # right BC, normal vector = (0,1)
                d1 = (-1 - 4 * D0[unroll_index([x_i, y_i])]/delta_y)
                d2 = (3/4) * D0[unroll_index([x_i, y_i])]/D2[unroll_index([x_i, y_i])]
                d3 = 2 * D0[unroll_index([x_i, y_i])]/delta_y
                d4 = (-3/4) * 2 * D0[unroll_index([x_i, y_i])] / delta_y
                d5 =  4 * D0[unroll_index([x_i, y_i])]/delta_y * 2 * get_I_1_value([x_i,y_i])

                e2 = (-1 - (80/21) * D2[unroll_index([x_i, y_i])]/delta_y)
                e1 = (1/7) * D2[unroll_index([x_i, y_i])]/D0[unroll_index([x_i, y_i])]
                e4 = 2 * D2[unroll_index([x_i, y_i])]/delta_y
                e3 = (-2/7) * D2[unroll_index([x_i, y_i])] / delta_y
                e5 =  (6/5) * (80/21) * D2[unroll_index([x_i, y_i])]/delta_y * get_I_3_value([x_i,y_i])

                denom = (d1 - d2 * e1 / e2)
                f1 = (d5 - d2 * e5 / e2) / denom
                f2 = (d2 * e3 / e2 - d3) / denom
                f3 = (d2 * e4 / e2 - d4) / denom

                b_vector[i] -= f1 * delta_x
                A_matrix[i,i] += f2 * delta_x
                A_matrix[i,i+phi_2_offset] += f3 * delta_x

                b_vector[i + phi_2_offset] -= (e5 - e1 * f1) / e2 * delta_x
                A_matrix[i+phi_2_offset,i] += (-e1 * f2 - e3) / e2 * delta_x
                A_matrix[i+phi_2_offset,i+phi_2_offset] += (-e1 * f3 - e4) / e2 * delta_x
            else:
                # Phi_0 equations
                y_plus_term_0 = get_av_D0("y",x_i,y_i) * delta_x
                A_matrix[i,unroll_index([x_i, y_i+1])] =  -y_plus_term_0 # phi_0, (i,j+1) term
                A_matrix[i,i] +=  y_plus_term_0 # phi_0, (i,j) term

                # Phi_2 equations
                y_plus_term_2 = get_av_D2("y",x_i,y_i) * delta_x
                A_matrix[i+phi_2_offset,unroll_index([x_i, y_i+1]) + phi_2_offset] =  -y_plus_term_2 # phi_2, (i,j+1) term
                A_matrix[i+phi_2_offset,i+phi_2_offset] +=  y_plus_term_2 # phi_2, (i,j) term
            A_matrix[i,i] += (sigma_t[i] - nu_sigma_f[i] - sigma_s0[i]) * delta_x * delta_y
            A_matrix[i,i + phi_2_offset] += (-2) * (sigma_t[i] - nu_sigma_f[i] - sigma_s0[i]) * delta_x * delta_y
            b_vector[i] += Q[i] * delta_x * delta_y

            A_matrix[i+phi_2_offset,i] += (-2/5) * (sigma_t[i] - nu_sigma_f[i] - sigma_s0[i]) * delta_x * delta_y
            A_matrix[i+phi_2_offset,i + phi_2_offset] += ((sigma_t[i] - nu_sigma_f[i] - sigma_s2[i]) + (4/5) * (sigma_t[i] - nu_sigma_f[i] - sigma_s0[i])) * delta_x * delta_y
            b_vector[i+phi_2_offset] += (-2/5) * Q[i] * delta_x * delta_y
    return A_matrix, b_vector

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
A_matrix, b_vector = construct_A_matrix() # use the material data (like XSs) to make the A matrix for the equation being solved

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

