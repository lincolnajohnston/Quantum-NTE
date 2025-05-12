import sys
import os
sys.path.append(os.getcwd())
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
# ind: indices of QuantumCircuit, qc, to apply QFA, adds values in bits 1,2,and 3. 
# Put the sum bit (%2) in bit 3, but the carry result in bit 4
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

# Apply a Quantum Full Subtractor on the indices indicated, with an optional extra control qubit
# ind: indices of QuantumCircuit, qc, to apply QFS, subtracts values in bits 1 and 2 from bit 3 
# Put the dif bit in bit 3, but the borrow result in bit 4
def applyQFS(qc, ind, control_index = -1):
    if control_index == -1:
        qc.cx(ind[0], ind[2])
        qc.ccx(ind[0], ind[2], ind[3])
        qc.cx(ind[1], ind[2])
        qc.ccx(ind[1], ind[2], ind[3])
    else:
        cccx = MCXGate(3)
        qc.ccx(control_index, ind[0], ind[2])
        qc.append(cccx, [control_index, ind[0], ind[2], ind[3]])
        qc.ccx(control_index, ind[1], ind[2])
        qc.append(cccx, [control_index, ind[1], ind[2], ind[3]])

# Apply a Quantum Majority Gate on the indices indicated, with an optional extra control qubit
# ind: indices of QuantumCircuit, qc, to apply QMG, reverses the carry done in QFA. 
#bits 1,2,and 3 unchanged, maybe reverse bit 4 to undo carry
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

# Apply a Quantum Majority Gate on the indices indicated, with an optional extra control qubit
# ind: indices of QuantumCircuit, qc, to apply QMG, reverses the carry done in QFA. 
#bits 1,2,and 3 unchanged, maybe reverse bit 4 to undo carry
def applySubQMG(qc, ind, control_index = -1):
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

# Take the sum of two states
# n_bits: number of qubits in each register representing the numbers being summed
# A_ind: indices of qubits for first number to sum, vector of length n_bits ->  state unchanged
# B_ind: indices of qubits for second number to sum, vector of length n_bits  -> last n_bits digits of the sum result
# anc_ind: indices of ancilla qubits needed for the summation, vector of length n_bits + 1, will return to |0> except the first qubit which has the most significant bit of the sum result
# expended ancillas: 1
# reusable ancillas: n_bits
# return vector of indices where the sum result is represented
# taken from https://arxiv.org/pdf/quant-ph/9808061v2
def sum_of_states(qc, A_ind, B_ind, anc_ind, control_index, n_bits):
    assert(len(A_ind) == len(B_ind))
    assert(len(anc_ind) == len(B_ind) + 1)

    # apply QFA (Quantum Full Adder) gates to find sum of numbers
    for i in range(n_bits):
        applyQFA(qc, [anc_ind[-1 - i], A_ind[i], B_ind[i], anc_ind[-2-i]], control_index)

    # apply QMG (Quantum Majority Gate) gates to undo the carries
    for i in range(n_bits-2, -1, -1):
        applyQMG(qc, [anc_ind[-1 - i], A_ind[i], B_ind[i], anc_ind[-2-i]], control_index)
    
    return B_ind + [anc_ind[0]]

# Take the difference of two states, where B > A
# n_bits: number of qubits in each register representing the numbers being summed
# A_ind: indices of qubits for first number to sum, vector of length n_bits ->  state unchanged
# B_ind: indices of qubits for second number to sum, vector of length n_bits  -> difference result
# anc_ind: indices of ancilla qubits needed for the summation, vector of length n_bits + 1, will return to |0>
# expended ancillas: 0
# reusable ancillas: n_bits + 1
# return vector of indices where the sum result is represented
# taken from https://arxiv.org/pdf/quant-ph/9808061v2
def diff_of_states(qc, A_ind, B_ind, anc_ind, control_index, n_bits):
    assert(len(A_ind) == len(B_ind))
    assert(len(anc_ind) == len(B_ind) + 1)

    # apply QFA (Quantum Full Adder) gates to find sum of numbers
    for i in range(n_bits):
        applyQFS(qc, [anc_ind[-1 - i], A_ind[i], B_ind[i], anc_ind[-2-i]], control_index)

    # apply QMG (Quantum Majority Gate) gates to undo the carries
    for i in range(n_bits-2, -1, -1):
        applySubQMG(qc, [anc_ind[-1 - i], A_ind[i], B_ind[i], anc_ind[-2-i]], control_index)
    
    return B_ind

# multiply two states by each other (fact1 and fact2) and put the result in "product"
# fact1_anc_reg: ancillas used for effectively multiplying fact1 (a) by 2^i 0 <= i <= n, length n   -> state unchanged (stay in |0> state)
# fact1: first term being multiplied (length m) -> state unchanged
# product: input as an ancilla register of length (m+n) -> outputs product from operation
# fact2: second term being multiplied (length n) -> state unchanged
# carry_reg: m+n ancillas used for carries in the summation operations -> state unchanged (stay in |0> state)
# expended ancillas: m+n
# reusable ancillas: m+2n
# return vector of indices where the product is represented
# general method taken from https://arxiv.org/pdf/quant-ph/9511018
def multiply_states_explicit(qc, fact1_anc_reg, fact1, product, fact2, carry_reg):
    n_bits = len(fact2)
    m_bits = len(fact1)
    for i in range(n_bits):
        sum_ind = sum_of_states(qc, fact1_anc_reg[n_bits-i:] + fact1, product[0:m_bits+i], [product[m_bits+i]] + carry_reg[n_bits-i:], fact2[i], m_bits+i)
        #sum_ind = np.flip(qc.num_qubits - 1 - np.array(sum_ind)) # reverse the indices
    return sum_ind

# same as the other multiply states, but don't have to specify each of the ancilla registers
# return the vector holding the results and the vector holding the indices of the reusable ancillas
def multiply_states(qc, fact1, fact2, ancillas):
    n_bits = len(fact2)
    m_bits = len(fact1)
    #assert(len(ancillas) == 2*m_bits + 3*n_bits)
    product_ind = multiply_states_explicit(qc, ancillas[n_bits + m_bits:2*n_bits + m_bits], fact1, ancillas[0:n_bits + m_bits], fact2, ancillas[2*n_bits+m_bits:3*n_bits + 2*m_bits])
    return product_ind, ancillas[n_bits + m_bits:]

# apply gates to get the twos complement of the bit string for the indices passed in
def apply_twos_complement(qc, indices):
    # reverse every qubit
    for i in indices:
        qc.x(i)
    
    # add one to the result (P_n gate)
    for i in range(len(indices)-1,0,-1):
        P_n_cnot_gate = XGate().control(i)
        qc.append(P_n_cnot_gate, list(np.flip(indices[len(indices) - i - 1:])))
    qc.x(indices[-1])

# from a long binary string in vector format, binary_vector, get the value represented by the indices 
# in "ind", with the earlier indices being the most significant if bigEndian=True, and the later indices
# beign more significant if bigEndian=False
def getValueFromBinary(binary_vector, ind, bigEndian=False):
    if bigEndian:
        return sum([binary_vector[ind[i]] * int(math.pow(2,len(ind) - 1 - i)) for i in range(len(ind))])
    else:
        return sum([binary_vector[ind[i]] * int(math.pow(2,i)) for i in range(len(ind))])
    
p_vals = [1,7]
# return the number of qubits that need to be set aside to apply the O_D oracle when the
# number of qubits representing the input is "n" and the number of term used for the Taylor series used for
# the approximation is "n_taylor_terms"
def get_qubits_needed_for_OD(n, n_taylor_terms):
    if n == 1:
        return 8*n + 4*p_vals[0]
    
    qs = (20 + 2 * (n_taylor_terms-2) + 4 * (n_taylor_terms / 2 * (n_taylor_terms + 1) - 3)) * n
    for i in range(n_taylor_terms):
        qs += 2 * p_vals[i]
    qs += 2 * p_vals[n_taylor_terms-1]
    return qs

#def apply_OD(qc, )
    
'''####### test Subtractor #######
A = 5
B = 7
n = math.floor(math.log2(max(A,B))) + 1
A_binary = np.array([int(bit) for bit in bin(A)[2:].zfill(n)])
B_binary = np.array([int(bit) for bit in bin(B)[2:].zfill(n)])

qc = QuantumCircuit(3*n+1)

# encode the numbers to be added
for i in range(n):
    if(A_binary[n - i - 1] == 1):
        qc.x(i)
    if(B_binary[n - i - 1] == 1):
        qc.x(n + i)

sum_ind = diff_of_states(qc, list(range(n)), list(range(n, 2*n)), list(range(2*n, 3*n + 1)), -1, n)'''

####### test cosine Taylor expansion with only one term #######
'''x = 0
N = 4
n = math.ceil(math.log2(N))
x_binary = np.array([int(bit) for bit in bin(x)[2:].zfill(n)])

qc = QuantumCircuit(9*n+4)

# encode the numbers to be multiplied
for i in range(n):
    # encode x
    if(x_binary[n - i - 1] == 1):
        qc.x(n + 1 + i)
        qc.x(2*n + 1 + i)

# encode the fraction in the Taylor series
qc.x(3*n+1)

sum_ind = multiply_states_explicit(qc, list(range(1,n+1)), list(range(n+1,2*n+1)), list(range(3*n+2, 5*n+2)), list(range(2*n+1, 3*n+1)), list(range(7*n+3, 9*n+3)))
sum_ind = multiply_states_explicit(qc, [0], list(range(3*n+2, 5*n+2)), list(range(5*n + 2, 7*n + 3)), [3*n+1], list(range(7*n+3, 9*n+4)))

# find the circuit unitary
#circOp = Operator.from_circuit(qc)
#circuit_unitary = circOp.data'''

####### test cosine Taylor expansion with one term #######
x = 3
N = 4
n = math.ceil(math.log2(N))
x_binary = np.array([int(bit) for bit in bin(x)[2:].zfill(n)])
p1 = 1

n_qubits = int(get_qubits_needed_for_OD(n, 2))
qc = QuantumCircuit(n_qubits)

# encode the fraction coefficients in the Taylor series
first_coef_ind = [0]
qc.x(0) # first coefficient


# encode x twice
x1_start_index = p1
x2_start_index = p1 + n
x1_ind = list(range(p1,p1+n))
x2_ind = list(range(p1+n,p1+2*n))
for i in range(n):
    if(x_binary[n - i - 1] == 1):
        qc.x(x1_start_index + i)
        qc.x(x2_start_index + i)

ancilla_ind = list(range(2*n+p1, 8*n+4*p1))

x_sq_ind, ancilla_ind = multiply_states(qc, x1_ind, x2_ind, ancilla_ind)
term1_ind, ancilla_ind = multiply_states(qc, x_sq_ind, first_coef_ind, ancilla_ind)

sum_ind = term1_ind


####### test cosine Taylor expansion with two terms #######
'''x = 3
N = 4
n = math.ceil(math.log2(N))
x_binary = np.array([int(bit) for bit in bin(x)[2:].zfill(n)])
p1 = 1
p2 = 7 # represent 1/4! = 1/24 as 5/128

qc = QuantumCircuit(16*n + 2*p1 + 3*p2)

# encode the fraction coefficients in the Taylor series
first_coef_ind = [0]
qc.x(0) # first coefficient

# second coefficient (5/128)
c2_start_index = p1
second_coef_ind = list(range(p1,p1+p2))
qc.x(c2_start_index + 0)
qc.x(c2_start_index + 2)

# encode x twice
x1_start_index = p1 + p2
x2_start_index = p1 + p2 + n
x1_ind = list(range(p1+p2,p1+p2+n))
x2_ind = list(range(p1+p2+n,p1+p2+2*n))
for i in range(n):
    if(x_binary[n - i - 1] == 1):
        qc.x(x1_start_index + i)
        qc.x(x2_start_index + i)

ancilla_ind = list(range(p1+p2+2*n, 16*n+2*p1+3*p2))

x_sq_ind, ancilla_ind = multiply_states(qc, x1_ind, x2_ind, ancilla_ind)
term1_ind, ancilla_ind = multiply_states(qc, x_sq_ind, first_coef_ind, ancilla_ind)

sum_ind = term1_ind'''


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
final_binary = np.flip(np.array([int(bit) for bit in bin(max_state)[2:].zfill(qc.num_qubits)])) # gets the binary for the maximum state in the computational basis, corresponds to the bit representation of the circuit from bottom to top
print("binary from top to bottom: ", final_binary)
sum_in_binary = [final_binary[sum_ind[i]] for i in range(len(sum_ind))]
print("Sum in binary: ", sum_in_binary)
summation_answer = getValueFromBinary(final_binary, sum_ind, bigEndian=False)
print("result: ", summation_answer)

qc.draw('mpl', filename="quantum_cosine_approximator.png")
