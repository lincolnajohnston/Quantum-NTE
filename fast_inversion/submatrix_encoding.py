from qiskit import QuantumCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation, CXGate, XGate, QFT, HGate, RYGate
from qiskit.quantum_info import Statevector
import numpy as np
from scipy.linalg import qr
import math

# Given the ability to apply the NxN matrix, A, in which somewhere along the diagonal is the 2x2 matrix, B
# apply the B matrix to the quantum circuit. The B matrix is located
# at position [m:m+2,m:m+2] in the A matrix, where m must be a multiple of 2 (if it is not a multiple of two
# in the current discretization, N can just be doubled to make it divisible by 2)

# google AI created this function
def create_tridiagonal(n, a, b, c):
    """
    Creates an n x n tridiagonal matrix.

    Args:
        n: The size of the matrix (number of rows and columns).
        a: List of subdiagonal elements (length n-1).
        b: List of main diagonal elements (length n).
        c: List of superdiagonal elements (length n-1).

    Returns:
        A NumPy array representing the tridiagonal matrix.
    """
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = b[i]
        if i > 0:
            matrix[i, i - 1] = a[i - 1]
        if i < n - 1:
            matrix[i, i + 1] = c[i]
    return matrix

#return the edge-averaged diffusion coefficient (at the edge of a cell)
# which is the harmonic mean of the diffusion coefficient divided by the cell width, dx
def get_edge_av_diff_coef(D1, D2, dx1, dx2):
    return 2 * (D1/dx1) * (D2/dx2) / (D1/dx1 + D2/dx2)


# get the inverse of the L matrix, if approx=true, assume the VA^(-1)U term is zero in the Woodbury matrix approximation
# k is the constant to add to the diagonals of the C matrix (taken from the absorption cross section matrix, T) to avoid a singular matrix
def get_A_inv(D0, D1, m, n, h, k, approx=False):
    D_01 = get_edge_av_diff_coef(D0, D1, h, h)
    tridiag = 1/(h*h) * create_tridiagonal(n,-1*np.ones(n-1), 2*np.ones(n), -1*np.ones(n-1)) # O(log(n))
    tridiag_inv = np.linalg.inv(tridiag) # O(log(n)) in quantum computer using disgonalization, DCT
    D_mat_inv = np.diag(np.concatenate((1/D0*np.ones(m),1/D1*np.ones(n-m)))) # O(M), M is number of materials, O(1) in term of n
    A_inv = tridiag_inv @ D_mat_inv
    return A_inv

def get_A(D0, D1, m, n, h):
    D_01 = get_edge_av_diff_coef(D0, D1, h, h)
    tridiag = 1/(h*h) * create_tridiagonal(n,-1*np.ones(n-1), 2*np.ones(n), -1*np.ones(n-1)) # O(log(n))
    D_mat = np.diag(np.concatenate((D0*np.ones(m),D1*np.ones(n-m)))) # O(M), M is number of materials, O(1) in term of n
    A = D_mat @ tridiag
    return A

# used Google AI to make a random matrix that is also unitary
def random_unitary_matrix(n):
    """
    Generates a random unitary matrix of size n x n.

    Args:
        n (int): The dimension of the unitary matrix.

    Returns:
        numpy.ndarray: A random unitary matrix of shape (n, n).
    """
    # Generate a random complex matrix
    x = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)
    
    # Compute the QR decomposition
    q, r = qr(x)
    
    # Normalize the diagonal elements of R
    r_diag_phase = np.diag(r) / np.abs(np.diag(r))
    
    # Ensure positive diagonal elements
    u = q * r_diag_phase

    return u

# Create circuit: 1 ancilla + n data qubits
n = 3
N = int(math.pow(2,n))
domain_size = 1
h = 1/n

# define block_position, then calculate m
#block_position = [1,1] # state of all qubits (from most to least significant) except the least significant one to extract the submatrix, B
#m = int(sum([math.pow(2,n-i-1) * block_position[i] for i in range(n-1)])) # position of start of B_matrix in larger, padded A_matrix

# define m, then calculate block_position
m = 2
block_position = [int(i) for i in bin(int(m/2))[2:].zfill(n-1)]

D0 = 2
sigma_a_0 = 7
nu_sigma_f_0 = 9

D1 = 4
sigma_a_1 = 5
nu_sigma_f_1 = 1

#A_mat = get_A(D0, D1, m, N, h)
#A_inv_mat = get_A_inv(D0, D1, m, N, h, 0.5*sigma_a_1) # TODO: A_inv_mat will not be unitary, so I need to block encode it, just do a random unitary A_inv_mat for now
A_inv_mat = random_unitary_matrix(N)
B_mat = A_inv_mat[m:m+2,m:m+2]
qc = QuantumCircuit(n+1)

# q[0] = ancilla, q[1] = q2, q[2] = q1, q[3] = q0

# Initialize one qubit state
#initial_state = np.array([1,2,3,4,5,6,7,8])
initial_state = np.random.rand(2)
initial_state = initial_state / np.linalg.norm(initial_state)
eigvec_input_state = StatePreparation(initial_state)
qc.append(eigvec_input_state, [0]) # initial state in the 0th qubit

# U gate
# Apply U to q0, requires n-1 ancillas initilized to |0>
for i,b in enumerate(block_position):
    if b == 1:
        qc.x(n-i-1)

#apply the A_inv gate to the first n qubits
A_gate = UnitaryGate(A_inv_mat)
qc.append(A_gate, list(range(n)))  # Apply the inverse of the diffusion operator


# V gate. 1 ancilla qubit at the end of the circuit initialized to |0>

# apply X gates to match the bit representation of the location of the 2x2 submatrix in the larger NxN matrix
# used to make the controlled X gate in the next step perform correctly
for i,b in enumerate(block_position):
    if b == 0:
        qc.x(n-i-1)

# apply an x gate to the ancilla, controlled on gates [1,n] of the circuit
controlled_X = XGate().control(n-1)
qc.append(controlled_X, list(range(1,n+1)))  # controlled-X from ancilla to q0

# unapply the X gates previously applied
for i,b in enumerate(block_position):
    if b == 0:
        qc.x(n-i-1)

# Simulate and extract statevector
state = Statevector.from_instruction(qc)

# Extract amplitudes where ancilla = |1⟩ (i.e., successful projection)
post_selected_state = state.data[:]  # indices 8–15: ancilla = 1
expected_state = B_mat @ initial_state
print("actual state: ", post_selected_state)
print("expected state: ", expected_state)

# Show amplitudes for successful post-selection
#print("Post-selected (A|ψ⟩, unnormalized):")
#for i, amp in enumerate(post_selected):
#    if abs(amp) > 1e-6:
#        print(f"|{i:03b}>: {amp}")

qc.draw('mpl', filename="padded-matrix-pic.png")
