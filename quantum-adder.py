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
from qiskit.circuit.library import StatePreparation, CXGate, XGate, QFT, HGate, RYGate, U1Gate
from qiskit.quantum_info import Statevector
from QPE import PhaseEstimation
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Operator
import fable

def applyQFA(qc, ind):
    qc.ccx(ind[1], ind[2], ind[3])
    qc.cx(ind[1], ind[2])
    qc.ccx(ind[0], ind[2], ind[3])
    qc.cx(ind[0], ind[2])

def applyQMG(qc, ind):
    qc.ccx(ind[1], ind[2], ind[3])
    qc.cx(ind[1], ind[2])
    qc.ccx(ind[0], ind[2], ind[3])
    qc.cx(ind[1], ind[2])

def sum_of_states(qc, A_ind, B_ind, anc_ind):
    assert(len(A_ind) == len(B_ind))
    assert(len(anc_ind) == len(B_ind) + 1)
    n_bits = len(A_ind)

    # apply QFA (Quantum Full Adder) gates to find sum of numbers
    for i in range(n_bits):
        applyQFA(qc, [anc_ind[-1 - i], A_ind[i], B_ind[i], anc_ind[-2-i]])

    # apply QMG (Quantum Majority Gate) gates to undo the carries
    for i in range(n_bits-2, -1, -1):
        applyQMG(qc, [anc_ind[-1 - i], A_ind[i], B_ind[i], anc_ind[-2-i]])
    
    return B_ind + [anc_ind[0]]



A = 6
B = 3
n_bits = math.floor(math.log2(max(A,B))) + 1
A_binary = np.array([int(bit) for bit in bin(A)[2:].zfill(n_bits)])
B_binary = np.array([int(bit) for bit in bin(B)[2:].zfill(n_bits)])


qc = QuantumCircuit(3*n_bits+1)

# encode the numbers to be added
for i in range(n_bits):
    if(A_binary[n_bits - i - 1] == 1):
        qc.x(i)
    if(B_binary[n_bits - i - 1] == 1):
        qc.x(n_bits + i)

sum_ind = sum_of_states(qc, list(range(n_bits)), list(range(n_bits, 2*n_bits)), list(range(2*n_bits, 3*n_bits + 1)))


# find the circuit unitary
circOp = Operator.from_circuit(qc)
circuit_unitary = circOp.data

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
final_binary = np.array([int(bit) for bit in bin(max_state)[2:].zfill(3*n_bits+1)])
print(final_binary)
summation_answer = sum([final_binary[sum_ind[i]] * int(math.pow(2,n_bits - i)) for i in range(n_bits + 1)])
print("summation: ", summation_answer)

qc.draw('mpl', filename="quantum_adder.png")
