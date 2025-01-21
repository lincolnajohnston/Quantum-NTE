import math
import numpy as np
import matplotlib.pyplot as plt
import random

from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation

# returns true if the density matrix represents a pure quantum state
def is_pure(dm):
    pure_state = dm[0,:]
    pure_state = pure_state / np.linalg.norm(pure_state)
    dm_test = np.outer(pure_state, pure_state)
    diff = abs(dm_test - dm).sum()
    return diff < 1E-10

def get_state_from_dm(dm):
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

# Creates a quantum state where each term is approxmiately a sequential integer (and then scaled)
def get_interpolater_state(N):
    perfect_interp = [i for i in range(int(math.pow(2,N)))] # the state we are aiming to approximate
    return perfect_interp / np.linalg.norm(perfect_interp)
    '''equal_state = [1/math.sqrt(2), 1/math.sqrt(2)]
    large_equal_state = [1]
    total_norm = 1
    out_state = [1]
    for i in range(N):
        q_n = [1,math.pow(N,1/(N))-math.pow(i,1/(N))+1]
        q_n_norm = np.linalg.norm(q_n)
        q_n = q_n / q_n_norm
        total_norm = total_norm * q_n_norm

        out_state = np.kron(out_state,q_n)
        large_equal_state = np.kron(large_equal_state, equal_state)
    large_equal_state = large_equal_state * (math.pow(2,N/2) / total_norm)
    int_state = out_state - large_equal_state
    return int_state / np.linalg.norm(int_state)'''

nc = 3
Nc = int(math.pow(2,nc))
nf = 6
Nf = int(math.pow(2,nf))
dn = nf-nc

'''coarse_sol_1 = np.ones(Nc)
coarse_sol_1 = coarse_sol_1 / np.linalg.norm(coarse_sol_1)

coarse_sol_2 = np.zeros(Nc)
coarse_sol_2[0] = 1'''
#coarse_sol = np.array([random.uniform(0, 1) for _ in range(Nc)])
coarse_sol = np.array([0.1,0.2,0.3,0.4,0.45,0.5,0.8,0.9])
#coarse_sol_mins = [min(coarse_sol[i], coarse_sol[i+1]) for i in range(len(coarse_sol)-1)] + [coarse_sol[-1]]
print("coarse solution: ", coarse_sol)
coarse_sol_diff = np.array(coarse_sol)
coarse_sol_diff[:Nc-1] -= coarse_sol_diff[1:Nc]
coarse_sol_diff = np.abs(coarse_sol_diff)
coarse_sol_diff[Nc-1] = 0 

coarse_sol_norm = np.linalg.norm(coarse_sol)
coarse_sol_diff_norm = np.linalg.norm(coarse_sol_diff)
coarse_sol = coarse_sol / coarse_sol_norm
coarse_sol_diff = coarse_sol_diff / coarse_sol_diff_norm

interp_state = get_interpolater_state(dn)
interpolater_norm = (math.pow(2,dn)-1) / interp_state[-1] # norm that assures that the last value of interp_state is math.pow(2,dn)-1, which is directly on the linear interpolation line

alpha_0 = math.pow(2,dn/2) * coarse_sol_norm
alpha_1 = interpolater_norm * coarse_sol_diff_norm / math.pow(2,dn)

print("alpha 1 probability: ", math.pow(2,dn/2) * coarse_sol_norm / (math.pow(2,dn/2) * coarse_sol_norm + interpolater_norm * coarse_sol_diff_norm / math.pow(2,dn)))

'''alpha_norm = math.sqrt(alpha_0 * alpha_0 + alpha_1 * alpha_1)
alpha_0 = alpha_0 / alpha_norm
alpha_1 = alpha_1 / alpha_norm'''

beta_0 = 1/math.sqrt(3)
beta_1 = math.sqrt(2)/math.sqrt(3)
beta = np.array([beta_0, beta_1])
sigma_matrix = np.outer(beta, beta.conj())
phi_dot_product = np.dot(coarse_sol,coarse_sol_diff) * np.dot([1/math.pow(2,(dn)/2) for i in range(int(Nf/Nc))], interp_state)

M_matrix = np.array([[(alpha_0 * alpha_0) / (beta_0 * beta_0 * 1), (alpha_1 * alpha_0) / (beta_1 * beta_0 * phi_dot_product)],
              [(alpha_0 * alpha_1) / (beta_0 * beta_1 * phi_dot_product), (alpha_1 * alpha_1) / (beta_1 * beta_1 * 1)]])

phi_goal = alpha_0 * np.kron(coarse_sol, 1/math.pow(2,(dn)/2) * np.ones(int(Nf/Nc))) + alpha_1 * np.kron(coarse_sol_diff, interp_state)

M_eigenvalues, M_eigenvectors = np.linalg.eig(M_matrix)

# T transforms the state so that it can be measured in the computational basis and retain the same probabilities of each eigenvalue being measured
T = np.outer([1,0],M_eigenvectors[:,0]) + np.outer([0,1],M_eigenvectors[:,1])

qc1 = QuantumCircuit(2*nf+1,2*nf+1)
# ancilla qubit
sigma_stateprep = StatePreparation(beta)
# term 1
stateprep_phi_c = StatePreparation(coarse_sol)
#term 2
stateprep_phi_0 = StatePreparation(coarse_sol_diff)
interpolater_stateprep = StatePreparation(interp_state)

qc1.append(sigma_stateprep, [nf])

qc1.append(stateprep_phi_c, list(range(nc, nf)))
for i in range(nc):
    qc1.h(i)

qc1.append(stateprep_phi_0, list(range(nf + 1 + nc, 2 * nf + 1)))
qc1.append(interpolater_stateprep, list(range(nf + 1, nf + 1 + nc)))

# put cswap gates in quantum circuit
for i in range(nf):
    qc1.cswap(nf,i,nf+1+i)

num_iter = 1000000
T_gate = UnitaryGate(T)
qc1.append(T_gate, [nf])
qc1.measure([nf],[nf])
qc1.measure(list(range(nf)), list(range(nf)))
backend = QasmSimulator(method="statevector")
new_circuit = transpile(qc1, backend)
job = backend.run(new_circuit, shots=num_iter)
job_result = job.result()
counts = job_result.get_counts()

# post-processing: weight the counts by the eigenvalue associated with the measurement on the sigma qubit
weighted_counts = {}
for i, (bitkey, n) in enumerate(counts.items()):
    eig = M_eigenvalues[int(bitkey[nf])]
    if bitkey[nf+1:2*nf+1] in weighted_counts:
        weighted_counts[bitkey[nf+1:2*nf+1]] += eig * n
    else:
        weighted_counts[bitkey[nf+1:2*nf+1]] = eig * n
predicted_state = [math.sqrt(weighted_counts['{:b}'.format(i).zfill(nf)]/num_iter) if '{:b}'.format(i).zfill(nf) in weighted_counts else 0 for i in range(Nf)]


print("predicted_state: ", predicted_state)    
print("desired_state: ", phi_goal)  
error = np.linalg.norm(phi_goal - predicted_state)
print("L2 error: ", error)



#counts = {key: value + (0 if counts2.get(key)==None else counts2.get(key)) for key, value in counts1.items()}

#plt.bar(counts.keys(), counts.values(), color='g')
plt.plot(predicted_state)
plt.title("Measured State")
plt.figure()
plt.plot(phi_goal)
#plt.bar(counts1.keys(), counts1.values(), color='g')
plt.title("Desired State")
plt.show()
#print("counts: ", counts)
#print("state: ", state_vec)

qc1.draw('mpl', filename="circuit1.png")