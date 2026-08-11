import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import math
import cmath

from qiskit import transpile
from qiskit_aer import Aer, AerSimulator, QasmSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation, CXGate, XGate, QFT, HGate, RYGate, IntegerComparator, DraperQFTAdder
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
from helpers import fable

n = 3
N = math.pow(2,n)
M = 6

qc = QuantumCircuit(3*(n+1)+1)

# b vector state preparation
x_state = np.zeros(int(2*N)) # system is size 2N just because that was the convention for the basis change matrix for the Discrete Cosine Transform operator that this was going to be used for
x_val = 15
x_state[x_val] = 1
x_state_prep = StatePreparation(x_state)
qc.append(x_state_prep, list(range(n+1)))

# M integer bit representation state preparation
int_state_1 = np.zeros(int(2*N))
int_state_1[M] = 1
int_state_1_prep = StatePreparation(int_state_1)
qc.append(int_state_1_prep, list(range(n+1, 2*(n+1))))

# compare the b vector state to pre-set integers, put result in flag qubits
int_comp_1 = IntegerComparator(num_state_qubits=n+1, value=M, geq=True)
int_comp_2 = IntegerComparator(num_state_qubits=n+1, value=2*N-1, geq=True)
qc.append(int_comp_1, list(range(n+1)) + list(range(2*(n+1), 3*(n+1))))
qc.append(int_comp_2, list(range(n+1)) + list(range(2*(n+1)+1, 3*(n+1)+1)))

adder_1 = DraperQFTAdder(n+1, kind='fixed')
qc.append(adder_1, [int(2*(n+1))] + list(range(int(2*(n+1)+2), int(2*(n+1)+2+n))) + list(range(n+1)))

adder_M = DraperQFTAdder(n+1, kind='fixed').control(1)
qc.append(adder_M,[int(2*(n+1)+1)] + list(range((n+1), 2*(n+1))) + list(range(n+1)))

qc.save_statevector()

# Run emulator
backend = QasmSimulator(method="statevector")
new_circuit = transpile(qc, backend)
print(dict(new_circuit.count_ops())) # print the counts of each type of gate
job = backend.run(new_circuit)
job_result = job.result()

# print statevector of non-junk qubits
state_vec = job_result.get_statevector(qc).data
print(state_vec)
print("index of max value of statevector: ", np.argmax(state_vec))
print("binary of that index: ", bin(np.argmax(state_vec))[2:])

qc.draw('mpl', filename="comparator-test.png")
