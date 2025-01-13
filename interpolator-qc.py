import math
import numpy as np
import matplotlib.pyplot as plt
import random

from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation

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

'''phi_bits = 1
qc = QuantumCircuit(phi_bits + 1,phi_bits)

alpha_1 = 1
alpha_2 = 2
coef_initializer = 1/math.sqrt(alpha_1*alpha_1 + alpha_2*alpha_2) * np.array([[alpha_1, alpha_2],[alpha_2, -alpha_1]])

phi_1_mat = 1/math.sqrt(2) * np.array([[1, 1],[1,-1]])
#phi_1_mat = np.array([[1,0],[0,1]])
phi_2_mat = np.array([[0,1],[1,0]])

coef_gate = UnitaryGate(coef_initializer)
phi_1_gate = UnitaryGate(phi_1_mat).control(1)
phi_2_gate = UnitaryGate(phi_2_mat).control(1)

qc.append(coef_gate,[phi_bits])
qc.x([phi_bits])
qc.append(phi_1_gate,[i%(phi_bits+1) for i in range(-1,phi_bits)])
qc.x(1)
qc.append(phi_2_gate,[phi_bits,range(phi_bits)])

qc.measure(range(phi_bits), range(phi_bits))'''

nc = 3
Nc = int(math.pow(2,nc))
nf = 5
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

alpha_1 = math.sqrt(math.pow(2,dn/2) * coarse_sol_norm)
alpha_2 = math.sqrt(interpolater_norm * coarse_sol_diff_norm / math.pow(2,dn))

print("alpha 1 probability: ", math.pow(2,dn/2) * coarse_sol_norm / (math.pow(2,dn/2) * coarse_sol_norm + interpolater_norm * coarse_sol_diff_norm / math.pow(2,dn)))

alpha_norm = math.sqrt(alpha_1 * alpha_1 + alpha_2 * alpha_2)
alpha_1 = alpha_1 / alpha_norm
alpha_2 = alpha_2 / alpha_norm

total_shots = 100000
alpha_1_shots = alpha_1 * alpha_1 * total_shots
alpha_2_shots = alpha_2 * alpha_2 * total_shots

# term 1
stateprep_phi_c = StatePreparation(coarse_sol).control(1)

#term 2
stateprep_phi_0 = StatePreparation(coarse_sol_diff).control(1)
interpolater_stateprep = StatePreparation(interp_state).control(1)

# Make Circuit 1
qc1 = QuantumCircuit(nf+1,nf)
qc1.x(0)
qc1.append(stateprep_phi_c, [0] + list(range(dn+1,nf+1)))
for i in range(dn):
    qc1.ch(0,i+1)

qc1.measure(range(1,nf+1), range(nf))

qc1.save_statevector()
# Run quantum algorithm
backend = QasmSimulator(method="statevector")
new_circuit = transpile(qc1, backend)
job = backend.run(new_circuit, shots=alpha_1_shots)
job_result = job.result()
counts1 = job_result.get_counts()
#state_vec = job_result.get_statevector(qc).data

#from qiskit.visualization import plot_histogram
#plot_histogram(counts)

counts1 = dict(sorted(counts1.items()))
sqrt_counts1 = {key: math.sqrt(value) for key, value in counts1.items()}

# Make Circuit 2
qc2 = QuantumCircuit(nf+1,nf)
qc2.x(0)
qc2.append(stateprep_phi_0, [0] + list(range(dn+1,nf+1)))
qc2.append(interpolater_stateprep, list(range(dn+1)))

qc2.measure(range(1,nf+1), range(nf))

qc2.save_statevector()
# Run quantum algorithm
backend = QasmSimulator(method="statevector")
new_circuit = transpile(qc2, backend)
job = backend.run(new_circuit, shots=alpha_2_shots)
job_result = job.result()
counts2 = job_result.get_counts()
#state_vec = job_result.get_statevector(qc).data

#from qiskit.visualization import plot_histogram
#plot_histogram(counts)

counts2 = dict(sorted(counts2.items()))
sqrt_counts2 = {key: math.sqrt(value) for key, value in counts2.items()}

counts = {key: value + (0 if counts2.get(key)==None else counts2.get(key)) for key, value in counts1.items()}

plt.bar(counts.keys(), counts.values(), color='g')
plt.title("Fine grid counts")
plt.figure()
plt.bar(counts1.keys(), counts1.values(), color='g')
plt.title("Circuit 1 (Coarse Grid) counts")
plt.show()
print("counts: ", counts)
#print("state: ", state_vec)

qc1.draw('mpl', filename="circuit1.png")
qc2.draw('mpl', filename="circuit2.png")