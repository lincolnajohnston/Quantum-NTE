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
from qiskit.circuit.library import StatePreparation, CXGate, XGate, QFT, HGate, RYGate, U1Gate, MCXGate
from qiskit.quantum_info import Statevector
from QPE import PhaseEstimation
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Operator
import fable

# Apply a Quantum Full Adder on the indices indicated, with an optional extra control qubit
def applyQFA(qc, ind, control_index = -1):
    if control_index == -1:
        qc.ccx(ind[1], ind[2], ind[3])
        qc.cx(ind[1], ind[2])
        qc.ccx(ind[0], ind[2], ind[3])
        qc.cx(ind[0], ind[2])
    else:
        cccx = MCXGate(3)
        qc.append(cccx, [control_index, ind[1], ind[2], ind[3]])
        qc.ccx(control_index, ind[1], ind[2])
        qc.append(cccx, [control_index, ind[0], ind[2], ind[3]])
        qc.ccx(control_index, ind[0], ind[2])

# Apply a Quantum Majority Gate on the indices indicated, with an optional extra control qubit
def applyQMG(qc, ind, control_index = -1):
    qc.x(ind[2])
    if control_index == -1:
        qc.ccx(ind[1], ind[2], ind[3])
        qc.cx(ind[1], ind[2])
        qc.ccx(ind[0], ind[2], ind[3])
        qc.cx(ind[1], ind[2])
    else:
        cccx = MCXGate(3)
        qc.append(cccx, [control_index, ind[1], ind[2], ind[3]])
        qc.ccx(control_index, ind[1], ind[2])
        qc.append(cccx, [control_index, ind[0], ind[2], ind[3]])
        qc.ccx(control_index, ind[1], ind[2])
    qc.x(ind[2])

# Take the sum of two states, with the indices for each register input
def sum_of_states(qc, A_ind, B_ind, anc_ind, control_index):
    assert(len(A_ind) == len(B_ind))
    assert(len(anc_ind) == len(B_ind) + 1)
    n_bits = len(A_ind)

    # apply QFA (Quantum Full Adder) gates to find sum of numbers
    '''for i in range(n_bits):
        applyQFA(qc, [anc_ind[-1 - i], A_ind[i], B_ind[i], anc_ind[-2-i]], control_index)

    # apply QMG (Quantum Majority Gate) gates to undo the carries
    for i in range(n_bits-2, -1, -1):
        applyQMG(qc, [anc_ind[-1 - i], A_ind[i], B_ind[i], anc_ind[-2-i]], control_index)'''
    
    for i in range(n_bits):
        applyQFA(qc, [anc_ind[-1 - i], A_ind[i], B_ind[i], anc_ind[-2-i]], control_index)

    # apply QMG (Quantum Majority Gate) gates to undo the carries
    for i in range(n_bits-2, -1, -1):
        applyQMG(qc, [anc_ind[-1 - i], A_ind[i], B_ind[i], anc_ind[-2-i]], control_index)
    
    return B_ind + [anc_ind[0]]



a = 5
x = 6
m = math.floor(math.log2(a)) + 1
n = math.floor(math.log2(x)) + 1
a_binary = np.array([int(bit) for bit in bin(a)[2:].zfill(m)])
x_binary = np.array([int(bit) for bit in bin(x)[2:].zfill(n)])


qc = QuantumCircuit(3*m + 4*n)

# encode the numbers to be added
for i in range(m):
    # encode the a number
    if(a_binary[m - i - 1] == 1):
        qc.x(m + i)
for i in range(n):
    # encode the x number
    if(x_binary[n - i - 1] == 1):
        qc.x(4*n + i)

for i in range(n):
    sum_ind = sum_of_states(qc, list(range(n-i, m+n)), list(range(m+n, 2*m+n+i)), [2*m+n+i] + list(range(2*m+4*n-i, 3*m+4*n)), 2*m+2*n+i)
    sum_ind = np.flip(3*m+4*n - 1 - np.array(sum_ind)) # reverse the indices


# find the circuit unitary
#circOp = Operator.from_circuit(qc)
#circuit_unitary = circOp.data

qc.save_statevector()

# Run emulator
backend = QasmSimulator(method="statevector")
new_circuit = transpile(qc, backend)
print(dict(new_circuit.count_ops())) # print the counts of each type of gate
job = backend.run(new_circuit)
job_result = job.result()

# print statevector of non-junk qubits
state_vec = job_result.get_statevector(qc).data
max_state = np.argmax(state_vec)
final_binary = np.array([int(bit) for bit in bin(max_state)[2:].zfill(3*m+4*n)])
print("binary in order of sig digs:", final_binary)
print("binary from top to bottom: ", np.flip(final_binary))
sum_in_binary = [final_binary[sum_ind[i]] for i in range(m+n)]
print("Sum in binary: ", sum_in_binary)
summation_answer = sum([final_binary[sum_ind[i]] * int(math.pow(2,m+n - 1 - i)) for i in range(m+n)])
print("summation: ", summation_answer)

qc.draw('mpl', filename="quantum_adder.png")
