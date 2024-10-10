import numpy as np
from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian
import time
import ProblemData
import LcuFunctions

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
np.set_printoptions(threshold=np.inf)


start = time.perf_counter()

data = ProblemData.ProblemData("input.txt")

# make A matrix and b vector
if data.sim_method == "sp3":
    A_mat_size = 2 * (data.n_x) * (data.n_y) * data.G
    A_matrix, b_vector = data.sp3_construct_A_matrix(A_mat_size) 
elif data.sim_method == "diffusion":
    A_mat_size = (data.n_x) * (data.n_y) * data.G
    A_matrix, b_vector = data.diffusion_construct_A_matrix(A_mat_size)
    


# use the material data (like XSs) to make the A matrix for the equation being solved
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

classical_sol_vec = np.linalg.solve(A_matrix, b_vector)

# Initialise the quantum registers
qc = QuantumCircuit(qb, ql)

circuit_setup_time = time.perf_counter()
print("Circuit Setup Time: ", circuit_setup_time - unitary_construction_time)

gate_time = time.perf_counter()
print("Gate U and V Application Time: ", gate_time - op_time)

qc.save_statevector()

# Run quantum algorithm
backend = QasmSimulator(method="statevector")
new_circuit = transpile(qc, backend)
job = backend.run(new_circuit)
job_result = job.result()
state_vec = job_result.get_statevector(qc).data
state_vec = np.real(state_vec[len(quantum_b_vector) - len(b_vector):len(quantum_b_vector)])

if data.sim_method == "sp3":
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
state_vec.resize((data.G, data.n_x,data.n_y))
for g in range(data.G):
    ax = sns.heatmap(state_vec[g,:,:], linewidth=0.5)
    plt.title("Quantum Solution, Group " + str(g))
    #plt.savefig('quantum_sol_g' + str(g) + '.png')
    plt.figure()

classical_sol_vec = classical_sol_vec[:int(data.G * data.n_x * data.n_y)]
classical_sol_vec.resize((data.G, data.n_x,data.n_y))
for g in range(data.G):
    ax = sns.heatmap(classical_sol_vec[g,:,:], linewidth=0.5)
    plt.title("Real Solution, Group " + str(g))
    #plt.savefig('real_sol_g' + str(g) + '.png')
    plt.figure()

plt.show()
