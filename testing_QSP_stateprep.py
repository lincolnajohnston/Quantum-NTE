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

def getProjectorProbability(M, state):
    return np.dot(M @ state, (M @ state).conj())

# Measure one qubit in the M basis, but indirectly by transforming it so measurement in the computational 
# basis will give the same result, then performing another transformation to collapse to the same state 
def doMeasurementOfM(M, state, rho_bits):
    eigenvalues, eigenvectors = np.linalg.eig(M)

    # T transforms the state so that it can be measured in the computational basis and retain the same probabilities of each eigenvalue being measured
    T = np.kron(np.kron(Identity(rho_bits), np.outer([1,0],eigenvectors[:,0]) + np.outer([0,1],eigenvectors[:,1])), Identity(rho_bits))
    T_inv = np.kron(np.kron(Identity(rho_bits), np.outer(eigenvectors[:,0], [1,0]) + np.outer(eigenvectors[:,1], [0,1])), Identity(rho_bits))
                    
    # transform state                
    state = T @ state
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
    state = T_inv @ state
    return eigenvalues[i], state

# measure using the M operator but do the measurement directly in the basis of M
def doMeasurementOfM_old(M, state, rho_bits):
    eigenvalues, eigenvectors = np.linalg.eig(M)
    total_prob = 0
    rand = random.random()
    i = -1
    while (total_prob < rand):
        i+=1
        Mi = np.kron(np.kron(Identity(rho_bits), np.outer(eigenvectors[:,i], eigenvectors[:,i].conj())), Identity(rho_bits))
        marginal_prob = getProjectorProbability(Mi, state)
        total_prob += marginal_prob
    new_state = Mi @ state / math.sqrt(marginal_prob)
    return eigenvalues[i], new_state

# Like doing a trace of density matrix but on a quantum state, basically collapsing a large state into a smaller state
# this is still very untested, not sure it works basically at all
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
sim_type = "density_matrix"

alpha_0 = 1
alpha_1 = 2

#alpha_norm = math.sqrt(alpha_0 * alpha_0 + alpha_1 * alpha_1)
#alpha_0 = alpha_0 / alpha_norm
#alpha_1 = alpha_1 / alpha_norm

rho_bits = 1

# Make QSP Circuit
#sigma_matrix = np.array([[1/2, 1/2], [1/2, 1/2]])
#M_matrix = np.array([[1/3, math.sqrt(2)/3], [math.sqrt(2)/3, 2/3]])

#phi_0 = np.array([1/2,math.sqrt(3)/2])
phi_0 = np.array([math.sqrt(3)/2,1/2])
rho_0 = np.outer(phi_0, phi_0.conj())
#rho_0 = np.array([[1/4, math.sqrt(3)/4], [math.sqrt(3)/4, 3/4]])

phi_1 = np.array([1/math.sqrt(5), math.sqrt(4)/math.sqrt(5)])
rho_1 = np.outer(phi_1, phi_1.conj())

beta_0 = 1/math.sqrt(3)
beta_1 = math.sqrt(2)/math.sqrt(3)
beta = np.array([beta_0, beta_1])
sigma_matrix = np.outer(beta, beta.conj())
M_matrix = np.array([[(alpha_0 * alpha_0) / (beta_0 * beta_0 * np.dot(phi_1,phi_1)), (alpha_1 * alpha_0) / (beta_1 * beta_0 * np.dot(phi_0,phi_1))],
              [(alpha_0 * alpha_1) / (beta_0 * beta_1 * np.dot(phi_1,phi_0)), (alpha_1 * alpha_1) / (beta_1 * beta_1 * np.dot(phi_0,phi_0))]])

eigenvalues, eigenvectors = np.linalg.eig(M_matrix)

O_matrix = np.array([[1,3],[3,1]])
O_operator = Operator(O_matrix)
phi_goal = alpha_0 * phi_0 + alpha_1 * phi_1
tau_goal = np.outer(phi_goal, phi_goal.conj())

qc1 = QuantumCircuit(2*rho_bits+1,2*rho_bits+1)

# initialize system
if sim_type == "quantum_state":
    sigma_stateprep = StatePreparation(sigma_matrix)
    rho_0_stateprep = StatePreparation(phi_0)
    rho_1_stateprep = StatePreparation(phi_1)
    qc1.append(sigma_stateprep, [0])
    qc1.append(rho_0_stateprep, [1])
    qc1.append(rho_1_stateprep, [2])
elif sim_type == "density_matrix":
    sigma_dm = DensityMatrix(sigma_matrix)
    rho_0_dm = DensityMatrix(rho_0)
    rho_1_dm = DensityMatrix(rho_1)
    total_dm = DensityMatrix(np.kron(np.kron(rho_1, rho_0), sigma_matrix))

# put gates in quantum circuit
qc1.cswap(0,1,2)
qc1.swap(0,1)
#qc1.x(0)

# swap order of qubits so the top is the most significant qubit
#for i in range(rho_bits):
#    qc1.swap(i,2*rho_bits-i)
#qc1.save_statevector()

# create/run quantum circuit
if sim_type == "quantum_state":
    backend = QasmSimulator(method="statevector")
    new_circuit = transpile(qc1, backend)
    job = backend.run(new_circuit, shots=1000)
    job_result = job.result()
    counts = job_result.get_counts()
elif sim_type == "density_matrix":
    total_dm_final = total_dm.evolve(qc1)

    #measure_operator = Operator(np.kron(np.kron(O_matrix, M_matrix), Identity(rho_bits)))
    #quantum_expectation = total_dm_final.expectation_value(measure_operator)
    #classical_expectation = np.trace(tau_goal @ O_matrix)

    # try to get counts from qiskit DensityMatrix operations
    '''count_dict = {'{:b}'.format(i).zfill(rho_bits):0 for i in range(int(math.pow(2,rho_bits)))}
    rho_a = partial_trace(state=total_dm_final, qargs=[1, 2])
    rho_b = partial_trace(state=total_dm_final, qargs=[0, 2])
    measure_operator = Operator(np.kron(np.kron(Identity(rho_bits), M_matrix), Identity(rho_bits)))
    #weight = total_dm_final.expectation_value(measure_operator)
    weight = rho_b.measure()
    result = rho_a.measure()[0]
    count_dict[result] += weight'''

    # Use numpy arrays to manually do measurements
    rho_out_state = get_state_from_dm(total_dm_final)

    counts = np.zeros(int(math.pow(2,rho_bits)))
    num_iter = 10000
    for iter in range(num_iter):
        if(iter % 1000 == 0):
            print(iter)
        #M_diag = np.multiply(M_matrix, [[1,0],[0,1]])
        #M_offdiag = np.multiply(M_matrix, [[0,1],[1,0]])
        m_val, new_state = doMeasurementOfM(M_matrix, rho_out_state, rho_bits)
        new_dm = DensityMatrix(np.outer(new_state, new_state.conj()))
        #tao_out = partial_trace(state=new_dm, qargs=list(range(rho_bits,2*rho_bits+1)))
        state_out = getTraceOfState(new_state, 2*rho_bits+1, list(range(rho_bits+1)))
        #state_out = getTraceOfState(new_state, 2*rho_bits+1, [0,2])
        tao_out = np.outer(state_out.conj(), state_out)
        #print(is_pure(new_dm))
        #print(is_pure(tao_out))

        # sample the remaining state in the computational basis
        threshold = random.random()
        cum_prob = 0
        for sampled_basis in range(len(state_out)):
            cum_prob += state_out[sampled_basis] * state_out[sampled_basis]
            if cum_prob > threshold:
                break
        counts[sampled_basis] += m_val

    print(counts) 
    predicted_state = [math.sqrt(counts[i]/num_iter) for i in range(len(counts))]
    print("predicted_state: ", predicted_state)    
    print("desired_state: ", phi_goal)    


print("total_dm: ", get_state_from_dm(np.real(total_dm.data)))
print("total_dm_final: ", get_state_from_dm(np.real(total_dm_final.data)))
CSWAP = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,1,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
X = np.array([[0,1],[1,0]])

#manual_total_dm_final = CSWAP @ np.kron(np.kron(rho_1, rho_0), sigma_matrix) @ CSWAP.conj().T
#manual_total_dm_final = np.kron(np.eye(2), SWAP) @ CSWAP @ np.kron(np.kron(rho_1, rho_0), sigma_matrix) @ CSWAP.conj().T @ np.kron(np.eye(2), SWAP).conj().T
#manual_total_dm_final = np.kron(np.kron(np.eye(2),np.eye(2)), X) @ np.kron(np.kron(rho_1, rho_0), sigma_matrix)
#print("manual total_dm_final: ", get_state_from_dm(manual_total_dm_final))
#print("difference between manual and qiskit total_dm_final: ", manual_total_dm_final - total_dm_final.data)

#sqrt_counts = {key: math.sqrt(value) for key, value in counts.items()}

#print("state: ", state_vec)

qc1.draw('mpl', filename="test_circuit.png")