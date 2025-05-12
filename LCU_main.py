import numpy as np
import os
from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian
import time
import ProblemData
import LcuFunctions
import math

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
np.set_printoptions(threshold=np.inf)


start = time.perf_counter()

sim_path = 'simulations/Pu239_1G_2D_diffusion_coarse/'
input_file = 'input.txt'
data = ProblemData.ProblemData(sim_path + input_file)
save_results = False

# make A matrix and b vector
if data.sim_method == "sp3":
    A_mat_size = 2 * math.prod(data.n) * data.G
    A_matrix, b_vector = data.sp3_construct_A_matrix(A_mat_size) 
elif data.sim_method == "diffusion":
    A_mat_size = math.prod(data.n) * data.G
    A_matrix, b_vector = data.diffusion_construct_A_matrix(A_mat_size)
    


# use the material data (like XSs) to make the A matrix for the equation being solved
print("A matrix:")
print(A_matrix)
print("\n b vector: ")
print(b_vector)
eigenvalues, eigenvectors = np.linalg.eig(A_matrix)
print("A eigenvalues: ", eigenvalues)
print("A condition number: ", max(eigenvalues) / min(eigenvalues))

material_initialization_time = time.perf_counter()
print("Initialization Time: ", material_initialization_time - start)

# Do LCU routine (https://arxiv.org/pdf/1511.02306.pdf), equation 18
num_LCU_bits = 5
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

'''classical_sol_vec.resize((data.G, data.n_x,data.n_y), refcheck=False)
for g in range(data.G):
    ax = sns.heatmap(classical_sol_vec[g,:,:], linewidth=0.5)
    plt.title("Real Solution, Group " + str(g))
    plt.figure()
plt.show()'''

# select optimal J, K, y_max, and z_max in just about the least efficient way possible
'''best_j = 0
best_y_max = 0
best_z_max = 0
best_error_norm = np.inf
for j in range(int(num_LCU_bits/4),num_LCU_bits - int(num_LCU_bits/4)):
    J = pow(2,j)
    K = pow(2,num_LCU_bits-j-1)
    for y_max in np.linspace(3,25,10):
        last_error_norm = 9999999
        for z_max in np.linspace(0.5,5,10):
            U, alphas, error_norm = LcuFunctions.get_fourier_unitaries(J, K, y_max, z_max, quantum_mat, False, A_mat_size)
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
for y_max in np.linspace(5,25,10):
    U, alphas, error_norm = LcuFunctions.get_fourier_unitaries(J, K, y_max, best_z_max, quantum_mat, False, A_mat_size)
    print("y_max: ", y_max)
    print("Error: ", error_norm)
    if error_norm < best_error_norm:
        best_y_max = y_max
        best_error_norm = error_norm
    else:
        break
best_error_norm = np.inf
for z_max in np.linspace(0.5,5,10):
    U, alphas, error_norm = LcuFunctions.get_fourier_unitaries(J, K, best_y_max, z_max, quantum_mat, False, A_mat_size)
    print("z_max: ", z_max)
    print("Error: ", error_norm)
    if(last_error_norm < error_norm):
        break
    if error_norm < best_error_norm:
        best_z_max = z_max
        best_error_norm = error_norm
    else:
        break'''


# manually input parameters for LCU (16x16 diffusion, dx=0.5, dy=0.5, 5 LCU bits)
'''best_j = 3
best_y_max = 5.0
best_z_max = 3.0'''

# manually input parameters for LCU (16x16 diffusion, dx=0.15, dy=0.15, 5 LCU bits)
'''best_j = 2
best_y_max = 15.0
best_z_max = 2.5'''

# manually input parameters for LCU (8x8 diffusion, G=8, dx=0.5, dy=0.5, 4 LCU bits)
'''best_j = 3
best_y_max = 15.0
best_z_max = 2.5'''

# manually input parameters for LCU (8x8 diffusion, G=8, dx=0.2, dy=0.2, 5 LCU bits)
'''best_j = 2
best_y_max = 10.0
best_z_max = 2.0'''

# manually input parameters for LCU (32x32 diffusion, dx=0.5, dy=0.5, 3 LCU bits) (does not work well)
'''best_j = 1
best_y_max = 1.0
best_z_max = 2.0'''

# manually input parameters for LCU (16x16 sp3, dx=0.5, dy=0.5, 4 LCU bits)
'''best_j = 3
best_y_max = 1.5
best_z_max = 1.5'''

# manually input parameters for LCU (16x16 sp3, dx=0.15, dy=0.15, 4 LCU bits)
'''best_j = 2
best_y_max = 20.0
best_z_max = 3.0'''

# manually input parameters for LCU (16x16 sp3, dx=0.2, dy=0.2, 4 LCU bits)
'''best_j = 2
best_y_max = 25.0
best_z_max = 1.5'''

# manually input parameters for LCU (16x16 sp3, dx=0.2, dy=0.2, 5 LCU bits)
best_j = 2
best_y_max = 21.0
best_z_max = 2.5

print("Best j: ", best_j)
print("Best y_max: ", best_y_max)
print("Best z_max: ", best_z_max)

U, alphas, error_norm = LcuFunctions.get_fourier_unitaries(pow(2,best_j), pow(2,num_LCU_bits-best_j-1), best_y_max, best_z_max, quantum_mat, True, A_mat_size)
print("Error Norm: ", error_norm)

unitary_construction_time = time.perf_counter()
print("Unitary Construction Time: ", unitary_construction_time - material_initialization_time)

# Initialise the quantum registers
nb = int(np.log2(len(quantum_b_vector)))
qb = QuantumRegister(nb)  # right hand side and solution
ql = QuantumRegister(num_LCU_bits)  # LCU ancilla zero bits
cl = ClassicalRegister(num_LCU_bits)  # right hand side and solution

qc = QuantumCircuit(qb, ql)

circuit_setup_time = time.perf_counter()
print("Circuit Setup Time: ", circuit_setup_time - unitary_construction_time)

alpha = np.sum(alphas)

V = LcuFunctions.gram_schmidt_ortho(np.sqrt(alphas))
v_mat_time = time.perf_counter()
print("Construction of V matrix time: ", v_mat_time - circuit_setup_time)

op_time = time.perf_counter()
print("Operator Construction Time: ", op_time - v_mat_time)

# Run circuit using Numpy
'''b_state = quantum_b_vector
l_state = np.zeros(2**num_LCU_bits)
l_state[0] = 1
l_state = V @ l_state
state_vec = np.kron(l_state, b_state)
state_vec = U @ state_vec
state_vec = (np.kron(np.conj(V).T,np.eye(2**nb))) @ state_vec'''

# Run circuit sing Qiskit
# b vector State preparation
qc.append(LcuFunctions.get_b_setup_gate(quantum_b_vector, nb), qb[:])
V_gate = UnitaryGate(V, 'V', False)
U_gate = UnitaryGate(U, 'U', False)
V_inv_gate = UnitaryGate(np.conj(V).T, 'V_inv', False)
qc.append(V_gate, ql[:])
qc.append(U_gate, qb[:] + ql[:])
qc.append(V_inv_gate, ql[:])
qc.save_statevector()
# Run quantum algorithm
backend = QasmSimulator(method="statevector")
new_circuit = transpile(qc, backend)
job = backend.run(new_circuit)
job_result = job.result()
state_vec = job_result.get_statevector(qc).data
#print(state_vec[0:A_mat_size])
#qc.draw('mpl', filename="test_circuit.png")

state_vec = np.real(state_vec[len(quantum_b_vector) - len(b_vector):len(quantum_b_vector)])
classical_sol_vec = np.linalg.solve(A_matrix, b_vector)
#print(classical_sol_vec)

if data.sim_method == "sp3":
    # find scalar flux value from 0th and 2nd moments in SP3 equations
    for i in range(int(len(state_vec)/2)):
        state_vec[i] -= 2 * state_vec[int(i + len(state_vec)/2)]
        classical_sol_vec[i] -= 2 * classical_sol_vec[int(i + len(classical_sol_vec)/2)]
    state_vec = state_vec[:int(len(state_vec)/2)]
    classical_sol_vec = classical_sol_vec[:int(len(classical_sol_vec)/2)]

state_vec = (np.sum(state_vec) / np.abs(np.sum(state_vec))) * state_vec / np.linalg.norm(state_vec) # scale quantum result to match true answer
classical_sol_vec = classical_sol_vec / np.linalg.norm(classical_sol_vec) # scale result to match true answer and ensure positivity

precision = np.linalg.norm(np.abs(state_vec)-classical_sol_vec)
print("precision: ", precision)
# save data vectors to files
if save_results:
    i = 0
    while os.path.exists(sim_path + 'saved_data/stats' + str(i) + '.txt'):
        i += 1
    f = open(sim_path + 'saved_data/stats' + str(i) + '.txt', "w")
    f.write("precision: " +  str(precision))
    f.close()
    np.savetxt(sim_path + 'saved_data/psi' + str(i) + '.npy', state_vec)
    np.savetxt(sim_path + 'saved_data/real_psi_solution' + str(i) + '.npy', classical_sol_vec)


# Print results
print("quantum solution estimate: ", state_vec)
#print("expected quantum solution: ", np.matmul(M, vector))

print('classical solution vector:          ', classical_sol_vec)

sol_rel_error = (state_vec - classical_sol_vec) / classical_sol_vec
#print("Relative solution error: ", sol_rel_error)

sol_error = state_vec - classical_sol_vec
#print("Solution error: ", sol_error)

solve_time = time.perf_counter()
print("Total time: ", solve_time - start)

# Make graphs of results
state_vec.resize(tuple([data.G] + list(data.n)))
classical_sol_vec = classical_sol_vec[:int(A_mat_size)]
classical_sol_vec.resize(tuple([data.G] + list(data.n)))

if (data.dim == 2):

    xticks = np.round(np.array(range(data.n[0]))*data.h[0] - (data.n[0] - 1)*data.h[0]/2,3)
    yticks = np.round(np.array(range(data.n[1]))*data.h[1] - (data.n[1] - 1)*data.h[1]/2,3)

    flux_mins = np.zeros((data.G,1))
    flux_maxes = np.zeros((data.G,1))
    for g in range(data.G):
        flux_maxes[g] = max(np.max(state_vec[g,:,:]), np.max(classical_sol_vec[g,:,:]))
        flux_mins[g] = min(np.min(state_vec[g,:,:]), np.min(classical_sol_vec[g,:,:]))

    for g in range(data.G):
        ax = sns.heatmap(state_vec[g,:,:], linewidth=0.5, xticklabels=xticks, yticklabels=yticks, vmin=flux_mins[g], vmax=flux_maxes[g])
        ax.invert_yaxis()
        plt.title("Quantum Solution, Group " + str(g))
        #plt.savefig('quantum_sol_g' + str(g) + '.png')
        plt.figure()

    for g in range(data.G):
        ax = sns.heatmap(classical_sol_vec[g,:,:], linewidth=0.5, xticklabels=xticks, yticklabels=yticks, vmin=flux_mins[g], vmax=flux_maxes[g])
        ax.invert_yaxis()
        plt.title("Real Solution, Group " + str(g))
        #plt.savefig('real_sol_g' + str(g) + '.png')
        plt.figure()
    
    # interpolate classical solution to finer grid
    mult_factor = 8
    #interp_clas_sol_vec = np.kron(classical_sol_vec[0,:,:],np.ones((mult_factor,mult_factor))) * 0
    interp_clas_sol_vec = classical_sol_vec[0,:,:] * 0
    for g in range(data.G):
        ax = sns.heatmap(interp_clas_sol_vec, linewidth=0.5, xticklabels=xticks, yticklabels=yticks, vmin=flux_mins[g], vmax=flux_maxes[g])
        ax.invert_yaxis()
        plt.title("Real Solution, Group " + str(g))
        #plt.savefig('real_sol_g' + str(g) + '.png')
        plt.figure()

    '''sol_rel_error.resize((data.G, data.n_x,data.n_y))
    for g in range(data.G):
        ax = sns.heatmap(sol_rel_error[g,:,:], linewidth=0.5, xticklabels=xticks, yticklabels=yticks)
        ax.invert_yaxis()
        plt.title("Relative error between quantum and real solution, Group " + str(g))
        plt.figure()'''

    '''for g in range(data.G):
        ax = sns.heatmap(sol_error[g,:,:], linewidth=0.5)
        plt.title("Actual error between quantum and real solution, Group " + str(g))
        plt.figure()'''
    plt.show()
