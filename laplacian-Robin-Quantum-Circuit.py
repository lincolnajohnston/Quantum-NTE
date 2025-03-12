import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import math
import cmath

from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation, CXGate, XGate, QFT, HGate
from qiskit.quantum_info import Statevector
from QPE import PhaseEstimation
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Operator
import fable

# return the A matrix and the b vector for the equation del^2(x) = 0. 1-D, Dirichlet BC where a = u_0, b = u_N
def get_laplacian_dirichlet_bc(N, x_range, a, b):
    h = x_range / N
    return_mat = 1/(h*h) * (2 * np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),1) - np.diag(np.ones(N-2),-1))

    return_vec = np.zeros(N-1)
    return_vec[0] = a / (h*h)
    return_vec[N-2] = b / (h*h)

    return return_mat, return_vec

# return the A matrix and the b vector for the equation del^2(x) = 0. 1-D, Neumann BC where a = u_0, b = u_N
def get_laplacian_neumann_bc(N, x_range, a, b):
    h = x_range / N
    return_mat = 1/(h*h) * (2 * np.diag(np.ones(N+1)) - np.diag(np.ones(N),1) - np.diag(np.ones(N),-1))
    return_mat[0,0] = 1/(h*h)
    return_mat[N,N] = 1/(h*h)

    return_vec = np.zeros(N+1)
    return_vec[0] = -2 * a / h
    return_vec[N] = 2 * b / h

    return return_mat, return_vec

# return the A matrix and the b vector for the equation del^2(x) = 0. 1-D, Dirichlet BC where a = u_0, b = u_N
def get_laplacian_robin_bc(N, x_range, a, b, c, d):
    h = x_range / N
    return_mat = 1/(h*h) * (2 * np.diag(np.ones(N+1)) - np.diag(np.ones(N),1) - np.diag(np.ones(N),-1))
    return_mat[0,0] = 1/(h*h) * (1-a/h)
    return_mat[0,1] = 1/(h*h) * a/h
    return_mat[N,N] = 1/(h*h) * (1+c/h)
    return_mat[N,N-1] = 1/(h*h) * (-c/h)

    return_vec = np.zeros(N+1)
    return_vec[0] = b / (h*h)
    return_vec[N] = d / (h*h)

    return return_mat, return_vec

def get_discrete_cosine_transform(N, K_min, K_max):
    return_mat = np.zeros((K_max - K_min + 1,K_max - K_min + 1))
    #for j in range(1,len(return_mat)+1):
    #    for k in range(1, len(return_mat)+1):
    for j in range(K_min, K_max + 1):
        for k in range(K_min, K_max + 1):
            return_mat[j-K_min,k-K_min] =  math.sqrt(2/N) * math.cos(math.pi * j * k / (N)) / (math.sqrt(2) if k % (N) == 0 else 1) / (2 if j % (N) == 0 else 1) # actual DCT
    return return_mat

def get_discrete_sine_transform(N, K_min, K_max):
    return_mat = np.zeros((K_max - K_min + 1,K_max - K_min + 1))
    #for j in range(1,len(return_mat)+1):
    #    for k in range(1, len(return_mat)+1):
    for j in range(K_min, K_max + 1):
        for k in range(K_min, K_max + 1):
            return_mat[j-K_min,k-K_min] = math.sqrt(2/N) * math.sin(math.pi * j * k / N) # correct values for the DST
            #return_mat[j-K_min,k-K_min] = math.sqrt(2/N) * math.sin(math.pi * j * k / N) / (math.sqrt(2) if k == K_min else 1)# / (math.sqrt(2) if j == K_min else 1)# testing random stuff
    return return_mat

def get_eigenvalues(N, K_min, K_max, x_range):
    h = x_range / N
    return np.diag([1/(h*h) * (2 - 2 * math.cos(math.pi * k / N)) for k in range(K_min, K_max + 1)])

# create the QFT unitary matrix then invert it (conjugate transpose it)
def get_QFT_matrix(n_bits, inverse = False):
    mat_size = int(math.pow(2,n_bits))
    omega = cmath.exp(2j*math.pi/mat_size)
    final_mat = np.ones((mat_size,mat_size), dtype=np.complex_)
    for i in range(mat_size):
        final_mat[:,i] *= omega ** i
    for i in range(mat_size):
        final_mat[i,:]  = np.power(final_mat[i,:],i)
    return (1/math.sqrt(mat_size)) * final_mat.conj() if inverse else (1/math.sqrt(mat_size)) * final_mat

# returns the matrix corresponding to the control of the input matrix, "unitary_mat".
# If "below" is True, the control is in the least significant bit position, else it is the most significant bit position
def getControl(unitary_mat, below=False):
    mat_size = len(unitary_mat)
    controlled_matrix = np.eye(2*mat_size)
    if(below):
        controlled_matrix[1:2*mat_size:2, 1:2*mat_size:2] = unitary_mat
    else:
        controlled_matrix[mat_size:2*mat_size, mat_size:2*mat_size] = unitary_mat
    return controlled_matrix

# returns the gate that does the transformation |x> -> |x+1 mod 2^n>, 
# where n is the number of qubits in the operation, x is a computational basis state
def getP_matrix(n):
    N = int(math.pow(2,n))
    P_mat = np.eye(N)
    not_gate = np.array([[0,1],[1,0]])
    I_N = N
    for i in range(n):
        I_N = int(I_N/2)
        P_mat =  P_mat @ np.kron(np.eye(I_N), not_gate)
        not_gate = getControl(not_gate,below=True)
    return P_mat

# return True if matrix is unitary, False otherwise, O(len(matrix)^2)
def is_unitary(matrix):
    I = matrix.dot(np.conj(matrix).T)
    return I.shape[0] == I.shape[1] and np.allclose(I, np.eye(I.shape[0]))

# does this even make sense
def get_O_D_matrix(n, inverse=False):
    N = int(math.pow(2,n))
    return_mat = np.zeros((N,N))
    for i in range(N):
        if inverse:
            return_mat[i, min(N-1,round((math.sin(i * math.pi / N - math.pi/2) + 1) * N/2))] = 1
        else:
            return_mat[min(N-1,round((math.sin(i * math.pi / N - math.pi/2) + 1) * N/2)), i] = 1
    return return_mat

n_dim = 1
input_folder = 'simulations/Pu239_1G_1D_diffusion_fine/'
n_x = 2
N_x = int(math.pow(2,n_x))
input_file = 'input.txt'
x_range = 4

# create and modify input file
data = ProblemData.ProblemData(input_folder + input_file)
data.n = np.array([N_x] * n_dim)
data.h = x_range / data.n
data.initialize_BC()
data.initialize_geometry()
A_mat_size = math.prod(data.n) * data.G

##### Robin B.C. Laplacian #####
a = 2 # ratio of the derivative term to the field term in the left Robin BC
b = 3 # assume incoming partial current at left boundary is 0
c = 4 # ratio of the derivative term to the field term in the right Robin BC
d = 5 # assume incoming partial current at right boundary is 0
alpha = -(a/data.h + 1)
beta = 1

robin_laplacian, robin_b = get_laplacian_robin_bc(N_x, x_range, a, b, c, d)
robin_eigvals, robin_eigvecs = eigh(robin_laplacian, eigvals_only=False)

# with the 0 and N nodes
sin_trans = get_discrete_sine_transform(N_x+2, 1, N_x+1)
sin_trans_inv = np.linalg.inv(sin_trans)
sin_trans_transposed = np.transpose(sin_trans)
cos_trans = get_discrete_cosine_transform(N_x, 0, N_x)
cos_trans_inv = np.linalg.inv(cos_trans)
cos_trans_transposed = np.transpose(cos_trans)
eigvals_1 = get_eigenvalues(N_x, 0, N_x, x_range)
eigvals_2 = get_eigenvalues(N_x+2, 1, N_x+1, x_range)


# alpha times the matrix made for the Neumann BC + beta time the matrix made for the Dirichlet BC
# TODO: extend the Dirichlet matrix to be on the first and last rows and columns, this is currently broken because it is not
neu_term = cos_trans @ eigvals_1 @ cos_trans_transposed
dir_term = sin_trans @ eigvals_2 @ sin_trans_transposed
approx_lap_rob_matrix = alpha / (alpha + beta) * neu_term + beta / (alpha + beta) * (N_x*N_x) / ((N_x+2) * (N_x+2)) * dir_term
print("Desired Robin Laplacian: ", robin_laplacian)
print("Estimated Robin Laplacian: ", np.round(approx_lap_rob_matrix,2))
print("Error in Robin Laplacian: ", approx_lap_rob_matrix - robin_laplacian)
print("Robin b vector: ", robin_b)


# make the quantum circuit using numpy
B = (1/math.sqrt(2)) * np.array([[1,1j],[1,-1j]])
B_inv = (1/math.sqrt(2)) * np.array([[1,1],[-1j,1j]])
P_n = getP_matrix(n_x)


qc = QuantumCircuit(3*n_x+5, 3*n_x+5)

# put the quantum circuit in the coarse phi_0 state and do interpolation onto the fine grid
b_vector = np.zeros(N_x)
# robin BC setup
#b_vector[0] = b
#b_vector[N_x-1] = d

# set to 0 state
b_vector[0] = 1

b_vector /= np.linalg.norm(b_vector)


# b vector state preparation
eigvec_input_state = StatePreparation(b_vector)
qc.append(eigvec_input_state, list(range(n_x)))

#create B gates
B_gate = UnitaryGate(B, label="B")
B_inv_gate = UnitaryGate(B_inv, label="B_inv")
B_gate_controlled = B_gate.control(n_x, ctrl_state='0'*n_x)
B_inv_gate_controlled = B_inv_gate.control(n_x, ctrl_state='0'*n_x)

q_gate_shift = 2*n_x+4
###### T_N ######
#apply first B gates
qc.append(B_gate, [n_x + q_gate_shift])
qc.append(B_inv_gate_controlled, range(q_gate_shift, n_x + 1 + q_gate_shift))

# apply a bunch of CNOT gates
for i in range(n_x - 1 + q_gate_shift, q_gate_shift-1, -1):
    qc.cnot(n_x + q_gate_shift,i)

# Apply P_n gate
for i in range(n_x-1 + q_gate_shift, -1 + q_gate_shift, -1):
    P_n_cnot_gate = XGate().control(i+1 - q_gate_shift)
    qc.append(P_n_cnot_gate, [3*n_x+4] + list(range(q_gate_shift, i)) + [i])

###### QFT ######
qft = QFT(n_x+1, do_swaps=True)
qc.append(qft,range(q_gate_shift, n_x+1+q_gate_shift))

###### T_N^-1 ######
# Apply (P_n)^-1 gate
for i in range(q_gate_shift, n_x + q_gate_shift):
    P_n_cnot_gate = XGate().control(i+1 - q_gate_shift)
    qc.append(P_n_cnot_gate, [n_x + q_gate_shift] + list(range(q_gate_shift, i)) + [i])

# apply a bunch of CNOT gates
for i in range(n_x-1 + q_gate_shift,-1 + q_gate_shift,-1):
    qc.cnot(n_x + q_gate_shift,i)

#apply final B gates
qc.append(B_gate_controlled, range(q_gate_shift, n_x+1+q_gate_shift))
qc.append(B_inv_gate, [n_x + q_gate_shift])

###### SCALING OF END ROWS ######
H_gate = HGate()
H_gate_controlled_1 = H_gate.control(n_x+1, ctrl_state='0'*(n_x+1))
H_gate_controlled_2 = H_gate.control(n_x+1, ctrl_state='1' + '0'*(n_x))
qc.append(H_gate_controlled_1, list(range(q_gate_shift, n_x+1+q_gate_shift)) + [n_x + 1])
qc.append(H_gate_controlled_2, list(range(q_gate_shift, n_x+1+q_gate_shift)) + [n_x + 1])

### O_D ###
O_D = get_O_D_matrix(n_x)
O_D_inv = get_O_D_matrix(n_x, inverse=True)
qc, alpha = fable.fable(O_D, qc, epsilon=0, max_i = 3*n_x+4)


qc, alpha = fable.fable(O_D_inv, qc, epsilon=0, max_i = 3*n_x+4)

###### INVERSE SCALING OF END ROWS ######
qc.append(H_gate_controlled_2, list(range(q_gate_shift, n_x+1+q_gate_shift)) + [n_x + 1])
qc.append(H_gate_controlled_1, list(range(q_gate_shift, n_x+1+q_gate_shift)) + [n_x + 1])

###### T_N ######
#apply first B gates
qc.append(B_gate, [n_x + q_gate_shift])
qc.append(B_inv_gate_controlled, range(q_gate_shift, n_x + 1 + q_gate_shift))

# apply a bunch of CNOT gates
for i in range(n_x - 1 + q_gate_shift, q_gate_shift-1, -1):
    qc.cnot(n_x + q_gate_shift,i)

# Apply P_n gate
for i in range(n_x-1 + q_gate_shift, -1 + q_gate_shift, -1):
    P_n_cnot_gate = XGate().control(i+1 - q_gate_shift)
    qc.append(P_n_cnot_gate, [3*n_x+4] + list(range(q_gate_shift, i)) + [i])

###### QFT ######
qft = QFT(n_x+1, do_swaps=True, inverse=True)
qc.append(qft,range(q_gate_shift, n_x+1+q_gate_shift))

###### T_N^-1 ######
# Apply (P_n)^-1 gate
for i in range(q_gate_shift, n_x + q_gate_shift):
    P_n_cnot_gate = XGate().control(i+1 - q_gate_shift)
    qc.append(P_n_cnot_gate, [n_x + q_gate_shift] + list(range(q_gate_shift, i)) + [i])

# apply a bunch of CNOT gates
for i in range(n_x-1 + q_gate_shift,-1 + q_gate_shift,-1):
    qc.cnot(n_x + q_gate_shift,i)

#apply final B gates
qc.append(B_gate_controlled, range(q_gate_shift, n_x+1+q_gate_shift))
qc.append(B_inv_gate, [n_x + q_gate_shift])

# plot O_D by itself
#ax = sns.heatmap(np.abs(O_D), linewidth=0.5)
#plt.show()

# print operator unitary matrix
circOp = Operator.from_circuit(qc)
circuit_unitary = circOp.data
circuit_unitary *= np.conj(circuit_unitary[0,0]) / abs(circuit_unitary[0,0]) # make the phase of the first term 0 so it's easier to compare to the correct cosine transform
cos_trans_error = circuit_unitary[0:N_x,0:N_x] - cos_trans[0:N_x,0:N_x]
print("circuit unitary: ", np.round(circuit_unitary,3))
print("desired cosine transform: ", cos_trans[0:N_x,0:N_x])
print("circuit unitary is unitary: ", is_unitary(circuit_unitary[0:N_x,0:N_x]))
print("desired cosine transform is unitary: ", is_unitary(cos_trans[0:N_x,0:N_x]))
#test_input_state = np.zeros(2 * N_x)
#test_input_state[0] = 1
#print("expected output: ", np.round(circOp.data @ test_input_state,3))

# test the quantum circuit cosine transformation to see if I can make the laplacian from it
circuit_cos_trans = circuit_unitary[0:N_x+1,0:N_x+1]
circuit_cos_trans_inv = np.transpose(circuit_cos_trans)
neu_term_circuit = circuit_cos_trans @ eigvals_1 @ circuit_cos_trans_inv
neu_term_correct = cos_trans @ eigvals_1 @ cos_trans_transposed
print("neu_term_circuit is unitary: ", is_unitary(neu_term_circuit))
print("neu_term_correct is unitary: ", is_unitary(neu_term_correct))
ax = sns.heatmap(np.abs(circuit_unitary[:N_x,:N_x]), linewidth=0.5)
plt.show()

qc.save_statevector()

# Run emulator
backend = QasmSimulator(method="statevector")
new_circuit = transpile(qc, backend)
print(dict(new_circuit.count_ops())) # print the counts of each type of gate
job = backend.run(new_circuit)
job_result = job.result()

# print statevector of non-junk qubits
state_vec = job_result.get_statevector(qc).data
print("Desired Cosine transform: ", cos_trans)
print("quantum state vector: ", state_vec)

qc.draw('mpl', filename="laplacian-robin-circuit-pic.png")
