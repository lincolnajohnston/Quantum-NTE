import math
import numpy as np
import matplotlib.pyplot as plt
import random

from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Operator
from qiskit.quantum_info import DensityMatrix, partial_trace

# returns true if the density matrix represents a pure quantum state
def is_pure(dm):
    if isinstance(dm, DensityMatrix):
        dm = dm.data
    pure_state = dm[0,:]
    pure_state = pure_state / np.linalg.norm(pure_state)
    dm_test = np.outer(pure_state, pure_state)
    diff = abs(dm_test - dm).sum()
    return diff < 1E-10

def get_state_from_dm(dm):
    if isinstance(dm, DensityMatrix):
        dm = dm.data
    if is_pure(dm) == False:
        raise Exception("not a pure matrix")
    return np.array([math.sqrt(dm[i,i]) for i in range(len(dm.data))])

def Identity(num_qubits):
    return np.eye(int(math.pow(2,num_qubits)))

# Measure one qubit in the M basis, but indirectly by transforming it so measurement in the computational 
# basis will give the same result, then performing another transformation to collapse to the same state 
def doMeasurementOfM(state, rho_bits, eigenvalues):     
    total_prob = 0
    rand = random.random()
    i = -1

    # measure the state in the computational basis and collapse the state to either 0 or 1
    traced_state = getTraceOfState(state, 2*rho_bits + 1, [i for i in range(2*rho_bits+1) if i != rho_bits])
    while (total_prob < rand):
        i+=1
        marginal_prob = traced_state[i] ** 2
        total_prob += marginal_prob
    M_bit_state = [1-i, i]
    collapse_operator = np.kron(np.kron(Identity(rho_bits), np.outer(M_bit_state,M_bit_state)), Identity(rho_bits))
    state = collapse_operator @ state / math.sqrt(marginal_prob)

    # return the state to the eigenvectors of the measurement operator, M, corresponding to the measurement
    # I don't think this step is strictly necessary to get the correct result for tau
    #state = T_inv @ state

    # return measurement and new state the circuit is in
    return eigenvalues[i], state

# Like doing a trace of density matrix but on a quantum state, basically collapsing a large state into a smaller state
# this is still very untested, not sure it works basically at all, it is also very inefficient and limiting the numpy implementation
def getTraceOfState(state, n_bits, trace_bits):
    n_trace_bits = len(trace_bits)
    bit_list = list(range(n_bits))
    untraced_bits = [bit_i for bit_i in bit_list if bit_i not in trace_bits]
    n_untraced_bits = len(untraced_bits)
    out_state = np.zeros(int(math.pow(2,n_untraced_bits)))

    for i in range(int(math.pow(2,n_untraced_bits))):
        untraced_binary_str = bin(i)[2:].zfill(n_untraced_bits)
        untraced_binary_list = [int(bit) for bit in untraced_binary_str]
        #print("i: ", i)
        for j in range(int(math.pow(2,n_trace_bits))):
            traced_binary_str = bin(j)[2:].zfill(n_trace_bits)
            traced_binary_list = [int(bit) for bit in traced_binary_str]

            large_index = int(np.sum([traced_binary_list[k] * math.pow(2,n_bits - trace_bits[k] - 1) for k in range(n_trace_bits)]) + np.sum([untraced_binary_list[k] * math.pow(2,n_bits - untraced_bits[k] - 1) for k in range(n_untraced_bits)]))
            #print("j: ", j)
            #print("large_index", large_index)
            out_state[i] += state[large_index] * state[large_index]
        out_state[i] = math.sqrt(out_state[i])
    
    return out_state


######### Input ###########
sim_type = "numpy"

alpha_0 = 1
alpha_1 = 2

rho_bits = 2

# Make QSP Circuit
#phi_0 = np.array([1/2,math.sqrt(3)/2])
#phi_0 = np.array([math.sqrt(3)/2,1/2])
#rho_0 = np.outer(phi_0, phi_0.conj())
#rho_0 = np.array([[1/4, math.sqrt(3)/4], [math.sqrt(3)/4, 3/4]])
phi_0 = np.array([1/2, 1/2, 1/2, 1/2])
rho_0 = np.outer(phi_0, phi_0.conj())

#phi_1 = np.array([1/math.sqrt(5), math.sqrt(4)/math.sqrt(5)])
#rho_1 = np.outer(phi_1, phi_1.conj())
phi_1 = np.array([1/math.sqrt(7), math.sqrt(2)/math.sqrt(7), math.sqrt(3)/math.sqrt(7), 1/math.sqrt(7)])
rho_1 = np.outer(phi_1, phi_1.conj())

beta_0 = 1/math.sqrt(3)
beta_1 = math.sqrt(2)/math.sqrt(3)
beta = np.array([beta_0, beta_1])
sigma_matrix = np.outer(beta, beta.conj())
M_matrix = np.array([[(alpha_0 * alpha_0) / (beta_0 * beta_0 * np.dot(phi_1,phi_1)), (alpha_1 * alpha_0) / (beta_1 * beta_0 * np.dot(phi_0,phi_1))],
              [(alpha_0 * alpha_1) / (beta_0 * beta_1 * np.dot(phi_1,phi_0)), (alpha_1 * alpha_1) / (beta_1 * beta_1 * np.dot(phi_0,phi_0))]])

eigenvalues, eigenvectors = np.linalg.eig(M_matrix)

phi_goal = alpha_0 * phi_0 + alpha_1 * phi_1

M_eigenvalues, M_eigenvectors = np.linalg.eig(M_matrix)

# T transforms the state so that it can be measured in the computational basis and retain the same probabilities of each eigenvalue being measured
T = np.outer([1,0],eigenvectors[:,0]) + np.outer([0,1],eigenvectors[:,1])

qc1 = QuantumCircuit(2*rho_bits+1,2*rho_bits+1)

# initialize system
if sim_type == "qiskit":
    sigma_stateprep = StatePreparation(beta)
    rho_0_stateprep = StatePreparation(phi_0)
    rho_1_stateprep = StatePreparation(phi_1)
    qc1.append(sigma_stateprep, [rho_bits])
    qc1.append(rho_0_stateprep, list(range(rho_bits)))
    qc1.append(rho_1_stateprep, list(range(rho_bits + 1, 2 * rho_bits + 1)))
elif sim_type == "numpy":
    sigma_dm = DensityMatrix(sigma_matrix)
    rho_0_dm = DensityMatrix(rho_0)
    rho_1_dm = DensityMatrix(rho_1)
    total_dm = DensityMatrix(np.kron(np.kron(rho_1, sigma_matrix), rho_0))

# put gates in quantum circuit
for i in range(rho_bits):
    qc1.cswap(rho_bits,i,rho_bits+1+i)

num_iter = 10000
# create/run quantum circuit
if sim_type == "qiskit":
    T_gate = UnitaryGate(T)
    qc1.append(T_gate, [rho_bits])
    qc1.measure([rho_bits],[rho_bits])
    qc1.measure(list(range(rho_bits)), list(range(rho_bits)))
    backend = QasmSimulator(method="statevector")
    new_circuit = transpile(qc1, backend)
    job = backend.run(new_circuit, shots=num_iter)
    job_result = job.result()
    counts = job_result.get_counts()

    # post-processing: weight the counts by the eigenvalue associated with the measurement on the sigma qubit
    weighted_counts = {}
    for i, (bitkey, n) in enumerate(counts.items()):
        eig = eigenvalues[int(bitkey[rho_bits])]
        if bitkey[rho_bits+1:2*rho_bits+1] in weighted_counts:
            weighted_counts[bitkey[rho_bits+1:2*rho_bits+1]] += eig * n
        else:
            weighted_counts[bitkey[rho_bits+1:2*rho_bits+1]] = eig * n
    predicted_state = [math.sqrt(weighted_counts['{:b}'.format(i).zfill(rho_bits)]/num_iter) for i in range(len(weighted_counts))]
elif sim_type == "numpy":
    total_dm_final = total_dm.evolve(qc1)

    # Use numpy arrays to manually do measurements
    rho_out_state = get_state_from_dm(total_dm_final)
    rho_out_state = np.kron(np.kron(Identity(rho_bits), T), Identity(rho_bits)) @ rho_out_state

    counts = np.zeros(int(math.pow(2,rho_bits)))
    for iter in range(num_iter):
        if(iter % 1000 == 0):
            print(iter)
        m_val, new_state = doMeasurementOfM(rho_out_state, rho_bits, M_eigenvalues)
        state_out = getTraceOfState(new_state, 2*rho_bits+1, list(range(rho_bits+1)))

        # sample the remaining state in the computational basis
        threshold = random.random()
        cum_prob = 0
        for sampled_basis in range(len(state_out)):
            cum_prob += state_out[sampled_basis] * state_out[sampled_basis]
            if cum_prob > threshold:
                break
        counts[sampled_basis] += m_val

    predicted_state = [math.sqrt(counts[i]/num_iter) for i in range(len(counts))]

# output results of quantum circuit
print("predicted_state: ", predicted_state)    
print("desired_state: ", phi_goal)  
error = np.linalg.norm(phi_goal - predicted_state)
print("L2 error: ", error)

qc1.draw('mpl', filename="test_circuit.png")