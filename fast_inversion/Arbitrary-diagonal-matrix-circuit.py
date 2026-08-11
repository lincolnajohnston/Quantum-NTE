from math import pi, acos, sqrt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit.library import StatePreparation, CXGate, XGate, QFT, HGate, RYGate, IntegerComparator, DraperQFTAdder
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
import numpy as np

def controlled_increment(qc, control, T, anc, start=0):
    """
    Controlled increment of the integer stored in T by (1 << start) when `control==1`.
    This implements a ripple increment beginning at bit index `start`.
    - control: single qubit (QuantumRegister index or Qubit)
    - T: list-like of qubits (T[0] is LSB)
    - anc: list-like ancilla qubits used for carries; length must be >= (len(T)-start)
    - start: integer >=0 the bit index where the increment adds 1<<start
    This implements a forward chain of CCX/CX then the reverse uncompute,
    leaving ancillas returned to zero.
    """
    n = len(T)
    L = n - start
    assert L >= 1
    assert len(anc) >= L

    # Forward chain
    for j in range(start, n):
        # carry qubit is control for j==start, else anc[j-start-1]
        if j == start:
            carry = control
        else:
            carry = anc[j - start - 1]
        # anc[j-start] ^= carry & T[j]
        qc.ccx(carry, T[j], anc[j - start])
        # T[j] ^= carry
        qc.cx(carry, T[j])

    # Reverse chain to uncompute ancillas
    for j in reversed(range(start, n)):
        if j == start:
            carry = control
        else:
            carry = anc[j - start - 1]
        # anc[j-start] ^= carry & T[j]  (CCX)
        qc.ccx(carry, T[j], anc[j - start])
        # anc[j-start] ^= carry  (CNOT)
        qc.cx(carry, anc[j - start])

# flip the bits in the "anc" register if the "x" register is in a state between min_state(inclusive) and max_state(exclusive)
# I think this takes O(n) Clifford + T gates where n is the number of qubits being compared
def apply_comparators(qc, x, x_anc, anc, min_state=0, max_state=0, backwards=False):
    integer_comp_low = IntegerComparator(num_state_qubits=len(x), value=min_state, geq=True)
    integer_comp_high = IntegerComparator(num_state_qubits=len(x), value=max_state, geq=False)

    if backwards: # apply the comparators in the opposite order for uncomputing the flag ancillas
        qc.append(integer_comp_high, x[:] + anc[2:3] + x_anc[:len(x)-1])
        qc.append(integer_comp_low, x[:] + anc[1:2] + x_anc[:len(x)-1])
    else:
        qc.append(integer_comp_low, x[:] + anc[1:2] + x_anc[:len(x)-1])
        qc.append(integer_comp_high, x[:] + anc[2:3] + x_anc[:len(x)-1])

# controlled upon the comparator flag qubits, apply a rotation that sets the diagonal value to sigma/sigma_max
# We can apply these 2 qubit rotation gates to precision epsilon using log(1/epsilon) Clifford + T gates
def apply_rotation_to_xs(qc, anc, xs_val):
    rotation_gate = RYGate(2*np.arccos(xs_val)).control(2)
    qc.append(rotation_gate, anc[::-1])


L_mat = 2
L = 4
N_mat = int(2**L_mat) # total number of distinct materials (assuming 1 dimension), more generally, the minimum number of blocks of identical diagonal values needed to represent the diagonal matrix 
N = int(2**L)

xs_list = np.array(range(N_mat))
xs_max = max(xs_list)

x = QuantumRegister(L, 'x') # main qubits being acted on
anc = QuantumRegister(3, 'anc')   # ancillas
x_anc = QuantumRegister(L, 'x_anc') # ancilla qubits used for comparator circuit

qc = QuantumCircuit(x, anc, x_anc)
dN = int(N/N_mat)
for i in range(N_mat): # O(N_mat * n * log(1/epsilon)) elementary gates
    apply_comparators(qc, x, x_anc, anc, min_state=i*dN, max_state=(i+1)*dN, backwards=False) # O(n) elementary (Clifford + T) gates
    apply_rotation_to_xs(qc, anc, xs_list[i]/xs_max) # O(log(1/epsilon)) elementary (Clifford + T) gates
    apply_comparators(qc, x, x_anc, anc, min_state=i*dN, max_state=(i+1)*dN, backwards=True) # uncompute comparator flag qubits, O(n) elementary (Clifford + T) gates

# get the unitary of the circuit
U = Operator(qc).data
print(U)

# run the circuit, produce statevector results
print(qc.draw(fold=120)) # print the circuit to the command line
x_val = 0
bits = [(x_val >> i) & 1 for i in range(L)]
qc_in = QuantumCircuit(2*L+3)
# set x bits
for i, b in enumerate(bits):
    if b:
        qc_in.x(i)  # our x register is first in qc definition
sv_in = Statevector.from_instruction(qc_in)
# apply the constructed unitary qc to sv_in
sv_out = sv_in.evolve(qc)
print(sv_out)

qc.draw('mpl', filename="arbitrary_diagonal_matrix_circuit.png")
