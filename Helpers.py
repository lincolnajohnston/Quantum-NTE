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
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
np.set_printoptions(threshold=np.inf)


def input_variable(filename, line_number):
    with open(filename, 'r') as file:
        lines = file.readlines()
    #check if line number is within range [1 to size]
    if 0 < line_number <= len(lines):
        return lines[line_number - 1].strip()
    else:
        return None


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
def unroll_index(index_vec, n_y):
    return index_vec[0]*n_y + index_vec[1]

# convert 1D index to 2D (x,y) index
def roll_index(index, n_y):
    return np.array([math.floor(index/n_y), index % n_y])


def get_I_1_value(index, n_x, n_y, left_y_I_1, right_y_I_1, bottom_x_I_1, top_x_I_1):
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
def get_I_3_value(index, n_x, n_y, left_y_I_3, right_y_I_3, bottom_x_I_3, top_x_I_3):
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
def initialize_XSs(n_pts_x, n_pts_y, delta_x, delta_y, sigma_t, sigma_s0, sigma_s2, nu_sigma_f, D0, D2, Q, n_x, n_y):
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
def get_av_D0(direction, lower_index, set_index, D0, delta_x, delta_y, n_y):
    if direction == "x":
        D_lower = D0[unroll_index([lower_index, set_index], n_y)]
        D_upper = D0[unroll_index([lower_index+1, set_index], n_y)]
        delta = delta_x
    elif direction == "y":
        D_lower = D0[unroll_index([set_index, lower_index], n_y)]
        D_upper = D0[unroll_index([set_index, lower_index+1], n_y)]
        delta = delta_y
    return 2 * (D_lower/delta) * (D_upper/delta) / (D_lower/delta + D_upper/delta)


def get_av_D2(direction, lower_index, set_index, D2, delta_x, delta_y, n_y):
    if direction == "x":
        D_lower = D2[unroll_index([lower_index, set_index], n_y)]
        D_upper = D2[unroll_index([lower_index+1, set_index], n_y)]
        delta = delta_x
    elif direction == "y":
        D_lower = D2[unroll_index([set_index, lower_index], n_y)]
        D_upper = D2[unroll_index([set_index, lower_index+1], n_y)]
        delta = delta_y
    return 2 * (D_lower/delta) * (D_upper/delta) / (D_lower/delta + D_upper/delta)


# use finite volume method to contruct the A matrix reprenting the diffusion equation in the form Ax=b, O(N)
def sp3_construct_A_matrix(n_x, n_y, A_mat_size, delta_x, delta_y, D0, D2, sigma_t, nu_sigma_f, sigma_s0, sigma_s2, Q, left_y_I_1, right_y_I_1, bottom_x_I_1, top_x_I_1, left_y_I_3, right_y_I_3, bottom_x_I_3, top_x_I_3):
    fd_order = 2
    beta = 0.5
    phi_2_offset = n_x * n_y
    A_matrix = np.zeros((A_mat_size, A_mat_size))
    b_vector = np.zeros((A_mat_size))
    for x_i in range(n_x):
        for y_i in range(n_y):
            i = unroll_index([x_i, y_i], n_y)
            if(x_i == 0): # left BC, normal vector = (-1,0)
                # these coefficients will be the same for all cells with the same material and mesh size so
                # need to imporove efficiency of this by storing these values beforehand instead of recalculating for each B.C. cell
                a1 = (1 + 4 * D0[unroll_index([x_i, y_i], n_y)]/delta_x)
                a2 = (-3/4) * D0[unroll_index([x_i, y_i], n_y)]/D2[unroll_index([x_i, y_i], n_y)]
                a3 = 2 * D0[unroll_index([x_i, y_i], n_y)]/delta_x
                a4 = (-3/4) * 2 * D0[unroll_index([x_i, y_i], n_y)] / delta_x
                a5 =  4 * D0[unroll_index([x_i, y_i], n_y)]/delta_x * 2 * get_I_1_value([x_i,y_i], n_x, n_y, left_y_I_1, right_y_I_1, bottom_x_I_1, top_x_I_1)

                b2 = (1 + (80/21) * D2[unroll_index([x_i, y_i], n_y)]/delta_x)
                b1 = (-1/7) * D2[unroll_index([x_i, y_i], n_y)]/D0[unroll_index([x_i, y_i], n_y)]
                b4 = 2 * D2[unroll_index([x_i, y_i], n_y)]/delta_x
                b3 = (-2/7) * D2[unroll_index([x_i, y_i], n_y)] / delta_x
                b5 =  (6/5) * (80/21) * D2[unroll_index([x_i, y_i], n_y)]/delta_x * get_I_3_value([x_i,y_i], n_x, n_y, left_y_I_3, right_y_I_3, bottom_x_I_3, top_x_I_3)

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
                x_minus_term_0 = get_av_D0("x",x_i-1,y_i, D0, delta_x, delta_y, n_y) * delta_y
                A_matrix[i,unroll_index([x_i-1, y_i], n_y)] =  -x_minus_term_0 # phi_0, (i-1,j) term
                A_matrix[i,i] +=  x_minus_term_0 # phi_0, (i,j) term

                # Phi_2 equations
                x_minus_term_2 = get_av_D2("x",x_i-1,y_i, D2, delta_x, delta_y, n_y) * delta_y
                A_matrix[i+phi_2_offset,unroll_index([x_i-1, y_i], n_y) + phi_2_offset] =  -x_minus_term_2 # phi_2, (i-1,j) term
                A_matrix[i+phi_2_offset,i+phi_2_offset] +=  x_minus_term_2 # phi_2, (i,j) term
            if(x_i == n_x - 1): # right BC, normal vector = (1,0)
                d1 = (-1 - 4 * D0[unroll_index([x_i, y_i], n_y)]/delta_x)
                d2 = (3/4) * D0[unroll_index([x_i, y_i], n_y)]/D2[unroll_index([x_i, y_i], n_y)]
                d3 = 2 * D0[unroll_index([x_i, y_i], n_y)]/delta_x
                d4 = (-3/4) * 2 * D0[unroll_index([x_i, y_i], n_y)] / delta_x
                d5 =  4 * D0[unroll_index([x_i, y_i], n_y)]/delta_x * 2 * get_I_1_value([x_i,y_i], n_x, n_y, left_y_I_1, right_y_I_1, bottom_x_I_1, top_x_I_1)

                e2 = (-1 - (80/21) * D2[unroll_index([x_i, y_i], n_y)]/delta_x)
                e1 = (1/7) * D2[unroll_index([x_i, y_i], n_y)]/D0[unroll_index([x_i, y_i], n_y)]
                e4 = 2 * D2[unroll_index([x_i, y_i], n_y)]/delta_x
                e3 = (-2/7) * D2[unroll_index([x_i, y_i], n_y)] / delta_x
                e5 =  (6/5) * (80/21) * D2[unroll_index([x_i, y_i], n_y)]/delta_x * get_I_3_value([x_i,y_i], n_x, n_y, left_y_I_3, right_y_I_3, bottom_x_I_3, top_x_I_3)

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
                x_plus_term_0 = get_av_D0("x",x_i,y_i, D0, delta_x, delta_y, n_y) * delta_y
                A_matrix[i,unroll_index([x_i+1, y_i], n_y)] =  -x_plus_term_0 # phi_0, (i+1,j) term
                A_matrix[i,i] +=  x_plus_term_0 # phi_0, (i,j) term

                # Phi_2 equations
                x_plus_term_2 = get_av_D2("x",x_i,y_i, D2, delta_x, delta_y, n_y) * delta_y
                A_matrix[i+phi_2_offset,unroll_index([x_i+1, y_i], n_y) + phi_2_offset] =  -x_plus_term_2 # phi_2, (i+1,j) term
                A_matrix[i+phi_2_offset,i+phi_2_offset] +=  x_plus_term_2 # phi_2, (i,j) term
            if(y_i == 0): # bottom BC, normal vector = (0,-1)
                a1 = (1 + 4 * D0[unroll_index([x_i, y_i], n_y)]/delta_y)
                a2 = (-3/4) * D0[unroll_index([x_i, y_i], n_y)]/D2[unroll_index([x_i, y_i], n_y)]
                a3 = 2 * D0[unroll_index([x_i, y_i], n_y)]/delta_y
                a4 = (-3/4) * 2 * D0[unroll_index([x_i, y_i], n_y)] / delta_y
                a5 =  4 * D0[unroll_index([x_i, y_i], n_y)]/delta_y * 2 * get_I_1_value([x_i,y_i], n_x, n_y, left_y_I_1, right_y_I_1, bottom_x_I_1, top_x_I_1)

                b2 = (1 + (80/21) * D2[unroll_index([x_i, y_i], n_y)]/delta_y)
                b1 = (-1/7) * D2[unroll_index([x_i, y_i], n_y)]/D0[unroll_index([x_i, y_i], n_y)]
                b4 = 2 * D2[unroll_index([x_i, y_i], n_y)]/delta_y
                b3 = (-2/7) * D2[unroll_index([x_i, y_i], n_y)] / delta_y
                b5 =  (6/5) * (80/21) * D2[unroll_index([x_i, y_i], n_y)]/delta_y * 2 * get_I_3_value([x_i,y_i], n_x, n_y, left_y_I_3, right_y_I_3, bottom_x_I_3, top_x_I_3)

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
                y_minus_term_0 = get_av_D0("y",x_i,y_i-1, D0, delta_x, delta_y, n_y) * delta_x
                A_matrix[i,unroll_index([x_i, y_i-1], n_y)] =  -y_minus_term_0 # phi_0, (i,j-1) term
                A_matrix[i,i] +=  y_minus_term_0 # phi_0, (i,j) term

                # Phi_2 equations
                y_minus_term_2 = get_av_D2("y",x_i,y_i-1, D2, delta_x, delta_y, n_y) * delta_x
                A_matrix[i+phi_2_offset,unroll_index([x_i, y_i-1], n_y) + phi_2_offset] =  -y_minus_term_2 # phi_2, (i,j-1) term
                A_matrix[i+phi_2_offset,i+phi_2_offset] +=  y_minus_term_2 # phi_2, (i,j) term
            if(y_i == n_y - 1): # right BC, normal vector = (0,1)
                d1 = (-1 - 4 * D0[unroll_index([x_i, y_i], n_y)]/delta_y)
                d2 = (3/4) * D0[unroll_index([x_i, y_i], n_y)]/D2[unroll_index([x_i, y_i], n_y)]
                d3 = 2 * D0[unroll_index([x_i, y_i], n_y)]/delta_y
                d4 = (-3/4) * 2 * D0[unroll_index([x_i, y_i], n_y)] / delta_y
                d5 =  4 * D0[unroll_index([x_i, y_i], n_y)]/delta_y * 2 * get_I_1_value([x_i,y_i], n_x, n_y, left_y_I_1, right_y_I_1, bottom_x_I_1, top_x_I_1)

                e2 = (-1 - (80/21) * D2[unroll_index([x_i, y_i], n_y)]/delta_y)
                e1 = (1/7) * D2[unroll_index([x_i, y_i], n_y)]/D0[unroll_index([x_i, y_i], n_y)]
                e4 = 2 * D2[unroll_index([x_i, y_i], n_y)]/delta_y
                e3 = (-2/7) * D2[unroll_index([x_i, y_i], n_y)] / delta_y
                e5 =  (6/5) * (80/21) * D2[unroll_index([x_i, y_i], n_y)]/delta_y * get_I_3_value([x_i,y_i], n_x, n_y, left_y_I_3, right_y_I_3, bottom_x_I_3, top_x_I_3)

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
                y_plus_term_0 = get_av_D0("y",x_i,y_i, D0, delta_x, delta_y, n_y) * delta_x
                A_matrix[i,unroll_index([x_i, y_i+1], n_y)] =  -y_plus_term_0 # phi_0, (i,j+1) term
                A_matrix[i,i] +=  y_plus_term_0 # phi_0, (i,j) term

                # Phi_2 equations
                y_plus_term_2 = get_av_D2("y",x_i,y_i, D2, delta_x, delta_y, n_y) * delta_x
                A_matrix[i+phi_2_offset,unroll_index([x_i, y_i+1], n_y) + phi_2_offset] =  -y_plus_term_2 # phi_2, (i,j+1) term
                A_matrix[i+phi_2_offset,i+phi_2_offset] +=  y_plus_term_2 # phi_2, (i,j) term
            A_matrix[i,i] += (sigma_t[i] - nu_sigma_f[i] - sigma_s0[i]) * delta_x * delta_y
            A_matrix[i,i + phi_2_offset] += (-2) * (sigma_t[i] - nu_sigma_f[i] - sigma_s0[i]) * delta_x * delta_y
            b_vector[i] += Q[i] * delta_x * delta_y

            A_matrix[i+phi_2_offset,i] += (-2/5) * (sigma_t[i] - nu_sigma_f[i] - sigma_s0[i]) * delta_x * delta_y
            A_matrix[i+phi_2_offset,i + phi_2_offset] += ((sigma_t[i] - nu_sigma_f[i] - sigma_s2[i]) + (4/5) * (sigma_t[i] - nu_sigma_f[i] - sigma_s0[i])) * delta_x * delta_y
            b_vector[i+phi_2_offset] += (-2/5) * Q[i] * delta_x * delta_y
    return A_matrix, b_vector


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
def get_fourier_unitaries(J, K, y_max, z_max, matrix, doFullSolution, A_mat_size):
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
