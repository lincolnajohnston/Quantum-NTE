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


from scipy.linalg import ishermitian, expm
from qiskit.quantum_info.operators import Operator
from linear_solvers.matrices.numpy_matrix import NumPyMatrix
np.set_printoptions(threshold=np.inf)

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
                sigma_a[i * n_y + j] = 10
                nu_sigma_f[i * n_y + j] = 1
                D[i * n_y + j] = 1
                Q[i * n_y + j] = 5
            else:
                # use moderator XSs
                sigma_a[i * n_y + j] = 1
                D[i * n_y + j] = 5

# Perform Gram-Schmidt orthogonalization to have basis of vectors including the alpha vector
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

def is_unitary(matrix):
    I = matrix.dot(np.conj(matrix).T)
    return I.shape[0] == I.shape[1] and np.allclose(I, np.eye(I.shape[0]))


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

matrix = A_matrix
vector = b_vector

# State preparation circuit - default is qiskit
if isinstance(vector, QuantumCircuit):
    nb = vector.num_qubits
    vector_circuit = vector
elif isinstance(vector, (list, np.ndarray)):
    if isinstance(vector, list):
        vector = np.array(vector)
    nb = int(np.log2(len(vector)))
    vector_circuit = QuantumCircuit(nb)
    # pylint: disable=no-member
    vector_circuit.isometry(
        vector / np.linalg.norm(vector), list(range(nb)), None
    )


# Hamiltonian simulation circuit - default is Trotterization
if isinstance(matrix, QuantumCircuit):
    matrix_circuit = matrix
elif isinstance(matrix, (list, np.ndarray)):
    if isinstance(matrix, list):
        matrix = np.array(matrix)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square!")
    if np.log2(matrix.shape[0]) % 1 != 0:
        raise ValueError("Input matrix dimension must be 2^n!")
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("Input matrix must be hermitian!")
    if matrix.shape[0] != 2**vector_circuit.num_qubits:
        raise ValueError(
            "Input vector dimension does not match input "
            "matrix dimension! Vector dimension: "
            + str(vector_circuit.num_qubits)
            + ". Matrix dimension: "
            + str(matrix.shape[0])
        )
    matrix_circuit = NumPyMatrix(matrix, evolution_time=2 * np.pi)
else:
    raise ValueError(f"Invalid type for matrix: {type(matrix)}.")


# check if the matrix can calculate the condition number and store the upper bound
if (
    hasattr(matrix_circuit, "condition_bounds")
    and matrix_circuit.condition_bounds() is not None
):
    kappa = matrix_circuit.condition_bounds()[1]
else:
    kappa = 1


# Do LCU routine (https://arxiv.org/pdf/1511.02306.pdf), equation 18
A_mat_size = len(matrix)
J = 64
K = 16
num_unitaries = 2 * J * K
num_LCU_bits = math.ceil(np.log2(num_unitaries))
ql = QuantumRegister(num_LCU_bits)  # LCU ancilla zero bits
cl = ClassicalRegister(num_LCU_bits)  # right hand side and solution
y_max = 3
z_max = 4
delta_y = y_max / J
delta_z = z_max / K

# Initialise the quantum registers
qb = QuantumRegister(nb)  # right hand side and solution

qc = QuantumCircuit(qb, ql)

# State preparation
qc.append(vector_circuit, qb[:])

M = np.zeros((A_mat_size, A_mat_size))
U = np.zeros((A_mat_size * 2 * J * K, A_mat_size * 2 * J * K), dtype=complex)
alphas = np.zeros(2 * J * K)
alpha = 0

for j in range(J):
    print("J: ", j)
    y = j * delta_y
    for k in range(-K,K):
        condition_mat = np.zeros((2*J*K, 2*J*K))
        condition_mat[j * 2 * K + (k + K),j * 2 * K + (k + K)] = 1.0
        z = k * delta_z
        alpha_temp = (1) / math.sqrt(2 * math.pi) * delta_y * delta_z * z * math.exp(-z*z/2)
        uni_mat = (1j) * expm(-(1j) * matrix * y * z)
        assert(is_unitary(uni_mat))
        M_temp = alpha_temp * uni_mat
        #U_temp = np.kron(condition_mat, uni_mat)
        #print(U_temp[A_mat_size*(j * 2 * K + (k + K)):A_mat_size*(j * 2 * K + (k + K) + 1),A_mat_size*(j * 2 * K + (k + K)):A_mat_size*(j * 2 * K + (k + K) + 1)])
        if(alpha_temp < 0): # if alpha is negative, incorporate negative phase into U unitary
            alpha_temp *= -1
            U[A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1),A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1)] = -1 * uni_mat
        else:
            U[A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1),A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1)] = uni_mat
        '''if(alpha_temp < 0): # if alpha is negative, incorporate negative phase into U unitary
            alpha_temp *= -1
            U = U - U_temp
        else:
            U = U + U_temp'''
        alpha += alpha_temp
        alphas[2 * j * K + (k + K)] = alpha_temp
        M = M + M_temp
print(U[len(U)-2*A_mat_size:len(U),len(U)-2*A_mat_size:len(U)])
matrix_invert = np.linalg.inv(matrix)
print(matrix_invert)
#print("prob: ", math.pow(np.linalg.norm(np.matmul(matrix, vector/ np.linalg.norm(vector))),2))
print(M)
print("Matrix inverse error: ", M - matrix_invert)
print(np.linalg.norm(M - matrix_invert))
V = gram_schmidt_ortho(alphas)
zero_vec = np.zeros(2*J*K).T
zero_vec[0] = 1
print("V * zero_vec: ", np.matmul(V,zero_vec))
print("normalized alphas: ", alphas / np.linalg.norm(alphas))
print("V is unitary: ", is_unitary(V))
V = np.kron(V, np.eye(A_mat_size))

# check if W is unitary
#print("V is unitary: ", is_unitary(V))
#print("U is unitary: ", is_unitary(U))
#print("W is unitary: ", is_unitary(W))

#W = np.matmul(np.conj(V).T, np.matmul(U , V)) # U and V are sparse matrices so this can be made more efficient
#LCU = Operator(W)
#qc.unitary(LCU, ql[:] + qb[:], label='lcu')

V_op = Operator(V)
U_op = Operator(U)
V_inv_op = Operator(np.conj(V).T)

qc.unitary(V_op, qb[:] + ql[:], label='V') # further to the right registers are the more major ones (ql has most major qubits here)
qc.unitary(U_op, qb[:] + ql[:], label='U')
qc.unitary(V_inv_op, qb[:] + ql[:], label='V_inv')

qc.save_statevector()

backend = QasmSimulator(method="statevector")
#backend_options = {'method': 'statevector'}
job = execute(qc, backend)
job_result = job.result()
state_vec = job_result.get_statevector(qc).data
print(state_vec)
print(state_vec[0:A_mat_size])
state_vec = np.absolute(state_vec[0:A_mat_size])
state_vec = state_vec / np.linalg.norm(state_vec)
print("quantum solution vector: ", state_vec)

print("expected quantum solution: ", np.matmul(M, vector / np.linalg.norm(vector)))

classical_sol_vec = np.linalg.solve(A_matrix, b_vector)
print('classical solution vector:          ', classical_sol_vec)





