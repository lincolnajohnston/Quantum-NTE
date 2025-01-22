import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation

# Linear Combination of states reference: https://arxiv.org/pdf/2112.12307

############## Functions ##############

# Creates a quantum state where each term is approxmiately a sequential integer [1,2,3,...] (and then scaled)
def get_interpolater_state(N, last_qubits, qc1):
    total_norm = 1
    last_val = 1
    dot_prod = 1
    for i in range(N):
        q_n = [1,math.pow(N,1/(N))-math.pow(i,1/(N))+1]
        q_n_norm = np.linalg.norm(q_n)
        q_n = q_n / q_n_norm
        last_val *= q_n[1]
        state = StatePreparation(q_n)
        for last_qubit in last_qubits:
            qc1.append(state,[last_qubit-i-1])
        total_norm = total_norm * q_n_norm
        dot_prod *= sum(q_n) / math.sqrt(2)

    return last_val, dot_prod

# Creates a quantum state where each term is a sequential integer (and then scaled), used to find error of quantum solution
def get_perfect_interpolater_state(N):
    perfect_interp = [i+1 for i in range(int(math.pow(2,N)))] # the state we are aiming to approximate
    return perfect_interp / np.linalg.norm(perfect_interp)

############## Inputs ##############
ncx = 2
ncy = 2
nc = ncx + ncy
Ncx = int(math.pow(2,ncx))
Ncy = int(math.pow(2,ncy))
nfx = 2
nfy = 3
nf = nfx + nfy
Nfx = int(math.pow(2,nfx))
Nfy = int(math.pow(2,nfy))
Nf = Nfx * Nfy
dnx = nfx - ncx
dny = nfy - ncy
num_iter = 100

#coarse_sol = np.array([random.uniform(0, 1) for _ in range(Nc)])
#coarse_sol = np.array([0.1,0.2,0.3,0.4,0.45,0.5,0.8,0.9])
#coarse_sol = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.43, 0.45, 0.47, 0.5, 0.6, 0.8, 0.82, 0.9, 0.99])
coarse_sol = np.kron(np.sin(0.3 * np.array(list(range(4)))) + 1, np.sin(0.3 * np.array(list(range(4)))) + 1)
print("coarse solution: ", coarse_sol)
coarse_sol_shift_x = [coarse_sol[i + 1]-coarse_sol[i] if (i+1)%Ncx!=0 else 0 for i in range(Ncx * Ncy)]
coarse_sol_shift_xy = [coarse_sol[i + Ncx + 1] - coarse_sol[i + Ncx] - coarse_sol[i + 1] + coarse_sol[i] if (i+1)%Ncx!=0 and i < Ncx * (Ncy-1) else 0 for i in range(Ncx * Ncy)]

coarse_sol_norm = np.linalg.norm(coarse_sol)
coarse_sol_shift_x_norm = np.linalg.norm(coarse_sol_shift_x)
coarse_sol_shift_xy_norm = np.linalg.norm(coarse_sol_shift_xy)
coarse_sol = coarse_sol / coarse_sol_norm
coarse_sol_shift_x = coarse_sol_shift_x / coarse_sol_shift_x_norm
coarse_sol_shift_xy = coarse_sol_shift_xy / coarse_sol_shift_xy_norm

############## Quantum Circuit Setup ##############

qc1 = QuantumCircuit(4*(nf)+3,4*(nf)+3)

last_val_x, interp_dot_prod_x = get_interpolater_state(nfx - ncx, [2*nf-nc-(nfy-ncy), 4*nf-nc-(nfy-ncy)], qc1)
last_val_y, interp_dot_prod_y = get_interpolater_state(nfy - ncy, [3*nf-nc, 4*nf-nc], qc1)
interpolater_norm_x = (math.pow(2,dnx)-1) / last_val_x # norm that assures that the last value of interp_state is math.pow(2,dn)-1, which is directly on the linear interpolation line
interpolater_norm_y = (math.pow(2,dny)-1) / last_val_y # norm that assures that the last value of interp_state is math.pow(2,dn)-1, which is directly on the linear interpolation line
alpha_0 = coarse_sol_norm * math.pow(2,(dnx + dny)/2)
alpha_1 = coarse_sol_shift_x_norm * interpolater_norm_x * math.pow(2,(dny)/2) / math.pow(2,dnx)
alpha_2 = coarse_sol_shift_x_norm * interpolater_norm_y * math.pow(2,(dnx)/2) / math.pow(2,dny)
alpha_3 = coarse_sol_shift_xy_norm * interpolater_norm_x * interpolater_norm_y / math.pow(2,dnx) / math.pow(2,dny)

perf_interp_state_x = get_perfect_interpolater_state(dnx)
perf_interp_state_y = get_perfect_interpolater_state(dny)
perf_interpolater_norm_x = (math.pow(2,dnx)-1) / perf_interp_state_x[-1] # norm that assures that the last value of interp_state is math.pow(2,dn)-1, which is directly on the linear interpolation line
perf_interpolater_norm_y = (math.pow(2,dny)-1) / perf_interp_state_y[-1]
perf_alpha_1 = coarse_sol_shift_x_norm * perf_interpolater_norm_x * math.pow(2,(dny)/2) / math.pow(2,dnx)
perf_alpha_2 = coarse_sol_shift_x_norm * perf_interpolater_norm_y * math.pow(2,(dnx)/2) / math.pow(2,dny)
perf_alpha_3 = coarse_sol_shift_xy_norm * perf_interpolater_norm_x * perf_interpolater_norm_y / math.pow(2,dnx) / math.pow(2,dny)

beta_00 = 1/math.sqrt(3)
beta_01 = math.sqrt(2)/math.sqrt(3)
beta_0 = np.array([beta_00, beta_01])
sigma_0_matrix = np.outer(beta_0, beta_0.conj())
beta_10 = 1/math.sqrt(3)
beta_11 = math.sqrt(2)/math.sqrt(3)
beta_1 = np.array([beta_00, beta_01])
sigma_1_matrix = np.outer(beta_0, beta_0.conj())
beta_20 = 1/math.sqrt(3)
beta_21 = math.sqrt(2)/math.sqrt(3)
beta_2 = np.array([beta_00, beta_01])
sigma_2_matrix = np.outer(beta_0, beta_0.conj())

phi_dot_product_01 = np.dot(coarse_sol,coarse_sol_shift_x) * interp_dot_prod_x # the closer the dot product is to 0, the more variance there will be in the result
phi_dot_product_02 = np.dot(coarse_sol,coarse_sol_shift_x) * interp_dot_prod_y
phi_dot_product_03 = np.dot(coarse_sol,coarse_sol_shift_xy) * interp_dot_prod_x * interp_dot_prod_y
phi_dot_product_12 = interp_dot_prod_x * interp_dot_prod_y
phi_dot_product_13 = np.dot(coarse_sol_shift_x,coarse_sol_shift_xy) * interp_dot_prod_y
phi_dot_product_23 = np.dot(coarse_sol,coarse_sol_shift_xy) * interp_dot_prod_x

phi_dot_product_a2 = alpha_0 * phi_dot_product_02 + alpha_1 * phi_dot_product_12
phi_dot_product_b3 = alpha_0 * phi_dot_product_03 + alpha_1 * phi_dot_product_13 + alpha_2 * phi_dot_product_23
phi_dot_product_aa = alpha_0**2 + alpha_1**2 + 2*alpha_0*alpha_1*phi_dot_product_01
phi_dot_product_bb = alpha_0**2 + alpha_1**2 + alpha_2**2 + 2*alpha_0*alpha_1*phi_dot_product_01 + 2*alpha_0*alpha_2*phi_dot_product_02 + 2*alpha_1*alpha_2*phi_dot_product_12

# matrix that allows the two states to be linearly combined
M0_matrix = np.array([[(alpha_0 * alpha_0) / (beta_00 * beta_00 * 1), (alpha_1 * alpha_0) / (beta_01 * beta_00 * phi_dot_product_01)],
              [(alpha_0 * alpha_1) / (beta_00 * beta_01 * phi_dot_product_01), (alpha_1 * alpha_1) / (beta_01 * beta_01 * 1)]])
M1_matrix = np.array([[(1 * 1) / (beta_10 * beta_10 * 1), (alpha_2 * 1) / (beta_11 * beta_10 * phi_dot_product_a2)],
              [(1 * alpha_2) / (beta_10 * beta_11 * phi_dot_product_a2), (alpha_2 * alpha_2) / (beta_11 * beta_11 * phi_dot_product_aa)]])
M2_matrix = np.array([[(1 * 1) / (beta_20 * beta_20 * 1), (alpha_3 * 1) / (beta_21 * beta_20 * phi_dot_product_b3)],
              [(1 * alpha_3) / (beta_20 * beta_21 * phi_dot_product_b3), (alpha_3 * alpha_3) / (beta_21 * beta_21 * phi_dot_product_bb)]])

# desired state: used for testing and finding error of quantum state
phi_goal = (alpha_0 * np.kron(np.kron(coarse_sol, 1/math.pow(2,(dny)/2) * np.ones(int(Nfy/Ncy))), 1/math.pow(2,(dnx)/2) * np.ones(int(Nfx/Ncx))) +
           perf_alpha_1 * np.kron(np.kron(coarse_sol_shift_x, 1/math.pow(2,(dny)/2) * np.ones(int(Nfy/Ncy))), perf_interp_state_x) + 
            perf_alpha_2 * np.kron(np.kron(coarse_sol_shift_x, perf_interp_state_y), 1/math.pow(2,(dnx)/2) * np.ones(int(Nfx/Ncx))) + 
            perf_alpha_3 * np.kron(np.kron(coarse_sol_shift_x, perf_interp_state_y), perf_interp_state_x))

#fine_mesh_test = coarse_sol_norm * math.pow(2,(dnx + dny)/2) * np.kron(np.kron(coarse_sol, 1/math.pow(2,(dny)/2) * np.ones(int(Nfy/Ncy))), 1/math.pow(2,(dnx)/2) * np.ones(int(Nfx/Ncx)))
#                + interpolater_norm_x * coarse_sol_shift_x_norm * math.pow(2,(dny)/2) / math.pow(2,dnx) * np.kron(np.kron(coarse_sol_shift_x, 1/math.pow(2,(dny)/2) * np.ones(int(Nfy/Ncy))), )

M0_eigenvalues, M0_eigenvectors = np.linalg.eig(M0_matrix)
M1_eigenvalues, M1_eigenvectors = np.linalg.eig(M1_matrix)
M2_eigenvalues, M2_eigenvectors = np.linalg.eig(M2_matrix)

# ancilla qubit
sigma_0_stateprep = StatePreparation(beta_0)
sigma_1_stateprep = StatePreparation(beta_1)
sigma_2_stateprep = StatePreparation(beta_2)
qc1.append(sigma_0_stateprep, [4*nf])
qc1.append(sigma_1_stateprep, [4*nf+1])
qc1.append(sigma_2_stateprep, [4*nf+2])

# term 1
stateprep_phi_0 = StatePreparation(coarse_sol)
qc1.append(stateprep_phi_0, list(range(nf-nc, nf)))
for i in range(nfx-ncx):
    qc1.h(i)
for i in range(nfx-ncx, nf-nc):
    qc1.h(i)

#term 2
stateprep_phi_1 = StatePreparation(coarse_sol_shift_x)
qc1.append(stateprep_phi_1, list(range(2 * nf - nc, 2 * nf)))
for i in range(nf+nfx-ncx, 2*nf-nc):
    qc1.h(i)

# term 3
qc1.append(stateprep_phi_1, list(range(3 * nf - nc, 3 * nf)))
for i in range(2*nf, 2*nf + nfx - ncx):
    qc1.h(i)

# term 4
stateprep_phi_3 = StatePreparation(coarse_sol_shift_xy)
qc1.append(stateprep_phi_3, list(range(4 * nf - nc, 4 * nf)))

# put cswap gates in quantum circuit
for i in range(nf):
    qc1.cswap(4*nf,i,nf+i) # first combination
for i in range(nf):
    qc1.cswap(4*nf+1,i,2*nf+i) # second combination
for i in range(nf):
    qc1.cswap(4*nf+2,i,3*nf+i) # third combination

############## Perform Measurements, Record Counts ##############

# T transforms the state so that it can be measured in the computational basis and retain the same probabilities of each eigenvalue being measured
T0 = np.outer([1,0],M0_eigenvectors[:,0]) + np.outer([0,1],M0_eigenvectors[:,1])
T1 = np.outer([1,0],M1_eigenvectors[:,0]) + np.outer([0,1],M1_eigenvectors[:,1])
T2 = np.outer([1,0],M2_eigenvectors[:,0]) + np.outer([0,1],M2_eigenvectors[:,1])
T0_gate = UnitaryGate(T0)
T1_gate = UnitaryGate(T1)
T2_gate = UnitaryGate(T2)
qc1.append(T0_gate, [4*nf])
qc1.append(T1_gate, [4*nf+1])
qc1.append(T2_gate, [4*nf+2])
qc1.measure([4*nf],[4*nf])
qc1.measure([4*nf+1],[4*nf+1])
qc1.measure([4*nf+2],[4*nf+2])


qc1.measure(list(range(nf)), list(range(nf)))
backend = QasmSimulator(method="statevector")
new_circuit = transpile(qc1, backend)
job = backend.run(new_circuit, shots=num_iter)
job_result = job.result()
counts = job_result.get_counts()

############## Post-Processing ##############

# weight the counts by the eigenvalue associated with the measurement on the sigma qubit
weighted_counts = {}
for i, (bitkey, n) in enumerate(counts.items()):
    eig_0 = M0_eigenvalues[int(bitkey[4*nf])]
    eig_1 = M1_eigenvalues[int(bitkey[4*nf+1])]
    eig_2 = M2_eigenvalues[int(bitkey[4*nf+2])]
    if bitkey[3*nf+3:4*nf+3] in weighted_counts:
        weighted_counts[bitkey[3*nf+3:4*nf+3]] += eig_0 * eig_1 * eig_2 * n
    else:
        weighted_counts[bitkey[3*nf+3:4*nf+3]] = eig_0 * eig_1 * eig_2 * n
predicted_state = [math.sqrt(abs(weighted_counts['{:b}'.format(i).zfill(nf)]/num_iter)) if '{:b}'.format(i).zfill(nf) in weighted_counts else 0 for i in range(Nf)]
predicted_state = np.array(predicted_state).reshape((Ncy, Ncx, int(Nfy/Ncy), int(Nfx/Ncx)))
predicted_state = np.transpose(predicted_state, (0, 2, 1, 3))
predicted_state = predicted_state.reshape(Nfy,Nfx)
phi_goal = np.array(phi_goal).reshape((Ncy, Ncx, int(Nfy/Ncy), int(Nfx/Ncx)))
phi_goal = np.transpose(phi_goal, (0, 2, 1, 3))
phi_goal = phi_goal.reshape(Nfy,Nfx)
#for i in range(Ncy):
#    phi_goal[i] = np.transpose(phi_goal[i])
#phi_goal = np.array(phi_goal).reshape((Ncx * Nfy, int(Nfx/Ncx)))
#phi_goal = np.array(phi_goal).reshape((Ncy,Ncx, int(Nfx/Ncx)))

############## Show Results ##############

# print difference between measured and desired state and L2
print("predicted_state: ", predicted_state)    
print("desired_state: ", phi_goal)  
error = np.linalg.norm(phi_goal - predicted_state)
print("L2 error: ", error)

# plot results
'''plt.plot(predicted_state)
plt.plot(phi_goal)
plt.plot(predicted_state - phi_goal)'''
heatmap = sns.heatmap(predicted_state)
heatmap.invert_yaxis()
plt.figure()
heatmap = sns.heatmap(phi_goal)
heatmap.invert_yaxis()

plt.title("Desired vs Measured States")
plt.legend(['Measured State', 'Desired State', 'error'])
plt.show()

qc1.draw('mpl', filename="2d-interpolator-circuit.png")