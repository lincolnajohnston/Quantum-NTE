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

alpha_0 = 1
alpha_1 = 1

alpha_norm = math.sqrt(alpha_0 * alpha_0 + alpha_1 * alpha_1)
alpha_0 = alpha_0 / alpha_norm
alpha_1 = alpha_1 / alpha_norm

beta_0 = 1/math.sqrt(2)
beta_1 = 1/math.sqrt(2)

rho_bits = 1

# Make Circuit
sigma_stateprep = StatePreparation([beta_0, beta_1])
phi_0 = [1/math.sqrt(3), math.sqrt(2)/math.sqrt(3)]
phi_1 = [1/math.sqrt(2), 1/math.sqrt(2)]
rho_0_stateprep = StatePreparation(phi_0)
rho_1_stateprep = StatePreparation(phi_1)

M_matrix = np.array(
    [[alpha_0 * alpha_0 / (beta_0 * beta_0), alpha_0 * alpha_1 / (beta_0 * beta_1 * np.dot(phi_0, phi_1))], [alpha_0 * alpha_1 / (beta_0 * beta_1 * np.dot(phi_0, phi_1)), alpha_1 * alpha_1 / (beta_1 * beta_1)]]
)
M_matrix = np.array([[1,1],[1,1]]) * math.sqrt(2)
eigenvalues, eigenvectors = np.linalg.eig(M_matrix)
print("eigenvalues: ", eigenvalues)
print("eigenvectors: ", eigenvectors)
observable = SparsePauliOp.from_operator(M_matrix)
observable_op = Operator(M_matrix)
total_counts = {bin(i)[2:].zfill(2*rho_bits+1):0 for i in range(int(math.pow(2,2*rho_bits+1)))}

for i in range(len(observable.coeffs)):
    qc1 = QuantumCircuit(2*rho_bits+1,2*rho_bits+1)
    qc1.append(sigma_stateprep, [1])
    '''qc1.append(rho_0_stateprep, [1])
    qc1.append(rho_1_stateprep, [2])
    qc1.cswap(0,1,2)
    qc1.swap(0,1)'''

    if observable.paulis.settings['data'][i] == 'X':
        qc1.h(1)
    elif observable.paulis.settings['data'][i] == 'Y':
        qc1.sdg(1)
        qc1.h(1)
    qc1.measure(range(0,2*rho_bits+1), range(2*rho_bits+1))

    qc1.save_statevector()
    # Run quantum algorithm
    backend = QasmSimulator(method="statevector")
    new_circuit = transpile(qc1, backend)
    job = backend.run(new_circuit, shots=1000)
    job_result = job.result()
    counts = job_result.get_counts()

    coef = observable.coeffs[i]
    counts = dict(sorted(counts.items()))
    for bin,count in counts.items():
        total_counts[bin] += coef * count
    #sqrt_counts = {key: math.sqrt(value) for key, value in counts.items()}

plt.bar(total_counts.keys(), total_counts.values(), color='g')
plt.title("Fine grid counts")
plt.show()
print("counts: ", total_counts)
#print("state: ", state_vec)

qc1.draw('mpl', filename="test_circuit.png")