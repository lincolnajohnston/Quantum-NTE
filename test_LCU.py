import numpy as np
from linear_solvers.hhl import HHL
from qiskit.quantum_info import Statevector
from qiskit import transpile
from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, expm

# number of variable points in each direction
n_x = 2
n_y = 2

# number of points (including boundary values) in each direction
n_pts_x = n_x + 2
n_pts_y = n_y + 2

delta_x = 0.5
delta_y = 0.5

top_flux = 0.5
bottom_flux = 1.5
right_flux = 2.5
left_flux = 3.5
top_x_BCs = np.ones(n_pts_x) * top_flux
bottom_x_BCs = np.zeros(n_pts_x) * bottom_flux
right_y_BCs = np.zeros(n_pts_y) * right_flux
left_y_BCs = np.zeros(n_pts_y) * left_flux


#sigma_a = np.array([1, 2, 3, 4])
#nu_sigma_f = np.array([2,4,3,1])
#Q = np.array([5,4,6,7])
sigma_a = np.zeros(n_x*n_y)
nu_sigma_f = np.zeros(n_x*n_y)
D = np.zeros(n_x*n_y)
Q = np.zeros(n_x*n_y)


A_mat_size = (n_x) * (n_y)

def get_solution_vector(solution, n):
    """Extracts and normalizes simulated state vector
    from LinearSolverResult."""
    solution_vector = Statevector(solution.state).data.real
    solution_length = len(solution_vector)
    solution_vector = solution_vector[int(solution_length/2):int(solution_length/2) + n]
    norm = solution.euclidean_norm
    return norm * solution_vector / np.linalg.norm(solution_vector)

def unroll_index(index_vec):
    return index_vec[0]*n_y + index_vec[1]

def roll_index(index):
    return np.array([math.floor(index/n_y), index % n_y])

# define BC values in this function
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

def initialize_XSs():
    x_range = (n_pts_x - 1) * delta_x
    y_range = (n_pts_y - 1) * delta_y

    fuel_radius = min(x_range,y_range)/3

    for i in range(n_x):
        for j in range(n_y):
            x_val = (i + 1) * delta_x - x_range
            y_val = (j + 1) * delta_y - y_range
            if (math.sqrt(x_val * x_val + y_val * y_val) < fuel_radius):
                # use fuel XSs
                sigma_a[i * n_y + j] = 2
                nu_sigma_f[i * n_y + j] = 2
                D[i * n_y + j] = 1
                Q[i * n_y + j] = 5
            else:
                # use moderator XSs
                sigma_a[i * n_y + j] = 1
                D[i * n_y + j] = 0.1


# make b vector
b_vector = Q
initialize_XSs()

# make A matrix
A_matrix = np.zeros((A_mat_size, A_mat_size))
for i in range(A_mat_size):
    A_matrix[i,i] = (2*D[i]/(delta_x*delta_x) + 2*D[i]/(delta_y*delta_y) + sigma_a[i] - nu_sigma_f[i])
    current_index = roll_index(i)
    if(current_index[0] > 0):
        A_matrix[i,unroll_index(current_index + np.array([-1, 0]))] = -D[i] / (delta_x*delta_x) # (i-1,j) term
    else:
        b_vector[i] += D[i] / (delta_x*delta_x) * get_BC_flux(current_index + np.array([-1, 0]))
    if(current_index[0] < n_x - 1):
        A_matrix[i,unroll_index(current_index + np.array([1, 0]))] = -D[i] / (delta_x*delta_x) # (i+1,j) term
    else:
        b_vector[i] += D[i] / (delta_x*delta_x) * get_BC_flux(current_index + np.array([1, 0]))
    if(current_index[1] > 0):
        A_matrix[i,unroll_index(current_index + np.array([0, -1]))] = -D[i] / (delta_y*delta_y) # (i,j-1) term
    else:
        b_vector[i] += D[i] / (delta_y*delta_y) * get_BC_flux(current_index + np.array([0, -1]))
    if(current_index[1] < n_y - 1):
        A_matrix[i,unroll_index(current_index + np.array([0, 1]))] = -D[i] / (delta_y*delta_y) # (i,j+1) term
    else:
        b_vector[i] += D[i] / (delta_y*delta_y) * get_BC_flux(current_index + np.array([0, 1]))

print("A matrix:")
print(A_matrix)
print("\n b vector: ")
print(b_vector)
eigenvalues, eigenvectors = np.linalg.eig(A_matrix)
print("A eigenvalues: ", eigenvalues)
print("A condition number: ", max(eigenvalues) / min(eigenvalues))

#print(np.linalg.matrix_power(A_matrix, 4))
print(expm(-(1j) * 4 * A_matrix))


# Do LCU routine (https://arxiv.org/pdf/1511.02306.pdf), equation 18
J = 100
K = 64
y_max = 10
z_max = 4
delta_y = y_max / J
delta_z = 2 * z_max / K

M = np.zeros((A_mat_size, A_mat_size))

for j in range(J):
    y = j * delta_y
    for k in range(-K,K+1):
        z = k * delta_z
        
        M = M + delta_y * delta_z * z * math.exp(-z*z/2) * expm(-(1j) * A_matrix * y * z)
        print(M)

M = M * (1j) / math.sqrt(2 * math.pi)
print("LCU 1/A: ", M)
print("Actual 1/A: ", np.linalg.inv(A_matrix))


