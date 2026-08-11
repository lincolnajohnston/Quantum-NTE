from math import pi, acos, sqrt
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit.library import StatePreparation, CXGate, XGate, QFT, HGate, RYGate, IntegerComparator, DraperQFTAdder
from qiskit.circuit import ControlledGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
import numpy as np

# contruct a quantum circuit for block-encoding one of the diagonals of the 1D absorption or fission matrix (C or A)
# given an input of |k> and |Delta_1>

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


# ChatGPT functions for applying division
def append_left_shift_and_bring_down(qc, r, x_bit):
    """
    r <- 2r + x_bit

    Assumes r[-1] is initially 0, which holds if len(r) is large enough
    to store values up to 2*N_f - 1.
    """
    # Shift r left by one bit.
    # Little-endian: r[0] is least significant bit.
    for j in reversed(range(len(r) - 1)):
        qc.swap(r[j], r[j + 1])

    # Bring down the next dividend bit into r[0].
    qc.cx(x_bit, r[0])

def append_controlled_subtract_const(qc, c, r, control, const):
    """
    Reversibly computes r <- r - const mod 2^m, controlled on control.

    Uses a QFT adder with a classical constant encoded into an ancilla
    register as the two's-complement value 2^m - const.
    """
    m = len(r)
    mod = 2**m
    twos_complement = (mod - const) % mod


    # Load the two's-complement constant.
    for j in range(m):
        if (twos_complement >> j) & 1:
            qc.x(c[j])

    adder = DraperQFTAdder(m, kind="fixed").to_gate()
    controlled_adder = adder.control(1)

    # |c>|r> -> |c>|r + c>, controlled by control.
    qc.append(controlled_adder, [control] + list(c) + list(r))

    # Unload the constant.
    for j in range(m):
        if (twos_complement >> j) & 1:
            qc.x(c[j])

def append_divide_by_const(qc, x, q, r, N_f):
    """
    Appends an exact reversible division circuit.

    Input:
        x: dividend register, little-endian
        q: quotient register, little-endian, initially |0>
        r: remainder register, little-endian, initially |0>
        N_f: positive classical integer divisor

    Output:
        |x>|0>|0> -> |x>|floor(x/N_f)>|x mod N_f>
    """
    if N_f <= 0:
        raise ValueError("N_f must be a positive integer.")

    n = len(x)

    if len(q) < n:
        raise ValueError("q should have at least len(x) qubits.")

    # Need one extra bit so that after r <- 2r + bit, no overflow occurs.
    min_r_bits = math.ceil(math.log2(N_f)) + 1
    if len(r) < min_r_bits:
        raise ValueError(f"r needs at least {min_r_bits} qubits.")

    flag = QuantumRegister(1, "cmp_flag")
    qc.add_register(flag)

    # Comparator ancillas
    comparator = IntegerComparator(
        num_state_qubits=len(r),
        value=N_f,
        geq=True
    )
    cmp_anc = QuantumRegister(comparator.num_qubits - len(r) - 1, "cmp_anc")
    qc.add_register(cmp_anc)

    cmp_gate = comparator.to_gate()
    inv_cmp_gate = cmp_gate.inverse()

    m = len(r)
    c = QuantumRegister(m, f"const_{N_f}")
    qc.add_register(c)

    # Long division, from most significant input bit to least significant.
    for i in reversed(range(n)):
        # r <- 2r + x_i
        append_left_shift_and_bring_down(qc, r, x[i])

        # flag <- [r >= N_f]
        qc.append(cmp_gate, list(r) + [flag[0]] + list(cmp_anc))

        # q_i <- flag
        qc.cx(flag[0], q[i])

        # uncompute comparator flag
        qc.append(inv_cmp_gate, list(r) + [flag[0]] + list(cmp_anc))

        # if q_i == 1: r <- r - N_f
        append_controlled_subtract_const(qc, c, r, q[i], N_f)


D = 2
L_mat = 2
L = 4
N_mat = int(2**L_mat) # total number of distinct materials (assuming 1 dimension), more generally, the minimum number of blocks of identical diagonal values needed to represent the diagonal matrix 
N = int(2**L)

xs_list = np.array(range(N_mat))
xs_max = max(xs_list)

# in these registers, "first" means most significant
x = QuantumRegister(L*D, 'x') # main qubits being acted on
anc_BE = QuantumRegister(1, 'anc_BE')   # block-encoding ancillas, only one for any D
anc_k = QuantumRegister(D, 'anc_k')   # k ancillas, D ancillas representing the values from 0 to 2^D - 1, also each bit represents the starting index of the cross sections being accessed in each dimension
anc_Delta = QuantumRegister(2*D, 'anc_Delta') # 2*D ancilla qubits used to represent the offsets representing values from -1 to 1 for each dimension. first qubit is sign (0 is positive, 1 is negative)
anc_nx = QuantumRegister(2*D, 'anc_nx') # 2*D ancilla qubits used to represent the index of the dimension whose cross sections are set to 0, if first qubit is 1, there is no index where cross sections are set to 0, if second qubit is 0, that index is 0, if second qubit is 1, that index is N - 1

# division register
anc_q = QuantumRegister(L*D, "q")
N_f = N-1 # N-1 for Dirichlet BCs, N+1 for Robin BCs
r_bits = math.ceil(math.log2(N_f)) + 1
anc_r = QuantumRegister(r_bits, "r")

qc = QuantumCircuit(x, anc_BE, anc_k, anc_Delta, anc_nx, anc_q, anc_r)
dN = int(N/N_mat)

######### set x input ########
qc.x(x[0])

######### set the input for k and the Deltas #########
# list of bits in strings go from most significant to least significant (bottom to top on the circuit usually)
#k_stateprep = StatePreparation('1') # 1D k state prep
k_stateprep = StatePreparation('00') # 2D k state prep, 0 offset for both dimensions
qc.append(k_stateprep, anc_k)

#Delta_stateprep = StatePreparation('01') # 1D Delta stateprep
Delta_stateprep = StatePreparation('1100') # 2D Delta stateprep, -1 offset for more significant dimension (y) and 0 offset for less significant dimension (x)
qc.append(Delta_stateprep, anc_Delta)


######### compute the anc_is and anc_nx values #########
# Delta_d = -1
# anc_is and anc_nx are set to 0 in this case so do nothing

# Delta_d = 0
for d in range(D): # d=0 will be the least significant dimension
    # nx gate
    nx_gate = XGate().control(num_ctrl_qubits=2, ctrl_state='00') # set the nx ancilla to -1 if Delta is '00' (Delta=0)
    qc.append(nx_gate, [anc_Delta[2*d], anc_Delta[2*d+1], anc_nx[2*d+1]]) # set the more significant nx bit to 1 controlled on '00'

# Delta_d = 1
for d in range(D): # d=0 will be the least significant dimension
    # nx gate
    nx_gate = XGate().control(num_ctrl_qubits=2, ctrl_state='01') # set the nx ancilla to 1 if Delta is '01' (Delta = +1)
    qc.append(nx_gate, [anc_Delta[2*d], anc_Delta[2*d+1], anc_nx[2*d]]) # set the less significant nx bit to 1 controlled on '00'

######### compute the indices for each dimension from the raveled index ########


append_divide_by_const(qc, x, anc_q, anc_r, N_f)


# run the circuit, produce statevector results
'''print(qc.draw(fold=120)) # print the circuit to the command line
x_val = 0
bits = [(x_val >> i) & 1 for i in range(L)]
qc_in = QuantumCircuit(qc.num_qubits)
# set x bits
for i, b in enumerate(bits):
    if b:
        qc_in.x(i)  # our x register is first in qc definition
sv_in = Statevector.from_instruction(qc_in)
# apply the constructed unitary qc to sv_in
sv_out = sv_in.evolve(qc)
print(sv_out)'''

# Add classical registers
c_q = ClassicalRegister(len(anc_q), "c_q")
c_r = ClassicalRegister(len(anc_r), "c_r")
qc.add_register(c_q)
qc.add_register(c_r)

# Measure quotient and remainder
qc.measure(anc_q, c_q)
qc.measure(anc_r, c_r)

sim = AerSimulator(method="matrix_product_state")

'''compiled = transpile(qc, sim, optimization_level=1)
result = sim.run(compiled, shots=1024).result()

counts = result.get_counts()
print(counts)'''

qc.draw('mpl', filename="one_diagonal_in_mass_matrix_1D.png")
