# Efficient gate-level implementation (Qiskit)
# Requires: qiskit
# Tested conceptually; please run locally in your environment.

from math import pi, acos, sqrt
from qiskit import QuantumCircuit, QuantumRegister, Aer, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import XGate

def prepare_offset_O(qc, O):
    """
    Applies Hadamard gate
    """
    qc.h(O[0])

def compute_x_into_T(qc, x, T):
    """
    Compute T := x (i.e., copy x into T).
    Assumes x has length a+1, T has length a+1 and T initially |0>.
    Uses CNOT(x[i] -> T[i]) for i=0..a-1.
    LSB convention: x[0] is least significant.
    """
    for i in range(len(x)):
        qc.cx(x[i], T[i])

def uncompute_offset_xor(qc, x, T, O, R):
    """
    The small 8-CNOT uncompute routine described earlier.
    - x: system register (x[0] is LSB)
    - T: target register (T[0] is LSB, T[1] next)
    - O: 2-qubit offset register (O[0] LSB, O[1] MSB) to be cleared
    - R: 2-qubit scratch register (init |00>), R[0] LSB, R[1] MSB
    After this, O returns to |00> and R returns to |00>, T and x unchanged.
    """

    # controlled-Hadamard and cx to fix up the ancillas for odd x
    qc.ch(x[0], O[0]) # x[0] is |1> if x is odd,
    qc.cx(x[0], T[-1]) # remainder from T=x/2 is always 1 if x is odd, reset to |0> state

    # fix up the even |x> inputs:

    # fix the |x> inputs divisible by 4
    # if x is divisible by 2 (even) and 4 (double-even) and T is odd, switch O register and MSB of T
    even_fixup = XGate().control(3)
    qc.x(x[0])
    qc.x(x[1])
    qc.append(even_fixup, x[:2] + T[:1] + O[:1])
    qc.append(even_fixup, x[:2] + T[:1] + T[-1:])
    qc.x(x[0])
    qc.x(x[1])

    # fix the |x> inputs divisible by 2 but not 4
    # if x is divisible by 2 (even) but not 4 (double-even) and T is even, switch O register and MSB of T
    qc.x(x[0])
    qc.x(T[0])
    qc.append(even_fixup, x[:2] + T[:1] + O[:1])
    qc.append(even_fixup, x[:2] + T[:1] + T[-1:])
    qc.x(T[0])
    qc.x(x[0])


def single_controlled_increment(qc: QuantumCircuit, control, T, anc):
    """
    Controlled increment-by-1 of register T (LSB T[0]) when control==1.
    Leaves anc restored to |0>.
    Requirements:
      - T: list-like of n qubits (LSB at index 0)
      - anc: list-like of length >= max(0, n-1)
    """
    n = len(T)
    if n == 0:
        return

    # Forward pass: compute carries into anc and flip bits
    for j in range(0, n):
        if j == 0:
            # anc[0] := control & T[0]  (only if n>=2)
            if n >= 2:
                qc.ccx(control, T[0], anc[0])
            # flip T[0] by control
            qc.cx(control, T[0])
        elif j < n - 1:
            # anc[j] := anc[j-1] & T[j]
            qc.ccx(anc[j-1], T[j], anc[j])
            # flip T[j] by previous carry
            qc.cx(anc[j-1], T[j])
        else:
            # j == n-1: flip top bit by previous carry
            qc.cx(anc[n-2], T[n-1])

    # Reverse pass: uncompute ancillas (mirror of forward)
    for j in range(n-1, -1, -1):
        if j == n - 1:
            if n >= 2:
                qc.cx(anc[n-2], T[n-1])
            else:
                # n == 1 case handled naturally by the single CX above; nothing extra here
                pass
        elif j > 0:
            # undo flip on T[j] and then undo anc[j]
            qc.cx(anc[j-1], T[j])
            qc.ccx(anc[j-1], T[j], anc[j])
        else:  # j == 0
            qc.cx(control, T[0])
            if n >= 2:
                qc.ccx(control, T[0], anc[0])


def controlled_decrement_by_one_two_controls(qc: QuantumCircuit, ctrl1, ctrl2, T, anc, ctrl_anc):
    """
    Two-control controlled decrement-by-1 of register T (mod 2^n).
    If (ctrl1 == 1 and ctrl2 == 1) then T := T - 1 (mod 2^n). Otherwise T unchanged.
    All ancillas (anc and ctrl_anc) are returned to |0>.
    - qc: QuantumCircuit
    - ctrl1, ctrl2: control qubits (Qubit or index)
    - T: list-like of n qubits (LSB at index 0)
    - anc: carry ancillas list, length >= max(0, n-1)
    - ctrl_anc: single ancilla qubit used to store ctrl1 & ctrl2 (must start |0>)
    """
    n = len(T)
    if n == 0:
        return

    # 1) compute combined control ctrl_anc := ctrl1 & ctrl2
    qc.ccx(ctrl1, ctrl2, ctrl_anc)

    # 2) conjugate trick: NOT(T); then do single-controlled increment by ctrl_anc; then NOT(T)
    for i in range(n):
        qc.x(T[i])

    #single_controlled_increment(qc, ctrl_anc, T, anc)
    controlled_increment(qc, ctrl_anc, T, anc)

    for i in range(n):
       qc.x(T[i])

    # 3) uncompute ctrl_anc
    qc.ccx(ctrl1, ctrl2, ctrl_anc)

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

def add_offset_into_T(qc, O, T, anc_for_inc):
    """
    Add the 2-qubit register O (bits O[0] LSB, O[1] MSB) into T in-place,
    i.e. T := T + O (interpreting O as an integer 0..3 but O only holds 0..2 here).
    We do:
      - controlled_increment(control = O[0], add 1 starting at bit 0)
      - controlled_increment(control = O[1], add 1 starting at bit 1)  (i.e. add 2)
    anc_for_inc must be a list of ancilla qubits >= len(T) (we reuse slices).
    """
    # add O[0] * 1 (start=0) using anc_for_inc[0:len(T)-0]
    controlled_increment(qc, O[0], T, anc_for_inc, start=0)

def divide_T_by_2(qc, T):
    m = len(T)
    for i in range(m-1):
        qc.swap(T[i], T[(i+1)%m])



# --- Full assembly function -------------------------------------------------

def build_shift_isometry_circuit(a):
    """
    Build the full QuantumCircuit that maps:
      |x> |0^{m}> -> |x> (1/2|2x> + 1/sqrt(2)|2x+1> + 1/2|2x+2>)
    where m = a+1.
    Returns the QuantumCircuit and the registers in a dict for convenience.
    """
    m = a + 1
    # Registers
    x = QuantumRegister(m, 'x')       # system
    T = QuantumRegister(m, 'T')       # target ancilla (holds 2x + offset)
    O = QuantumRegister(1, 'O')       # offset preparer
    R = QuantumRegister(1, 'R')       # small scratch for uncompute_offset_xor
    # ancillas for ripple increments: need length m (for start=0 case)
    ANC = QuantumRegister(m, 'anc')   # these will be returned to zero by design
    ANC_2 = QuantumRegister(1, 'anc_2')   # these will be returned to zero by design

    qc = QuantumCircuit(x, T, O, R, ANC, ANC_2, name=f"shift_iso_a{a}")

    # 1) compute T := x
    compute_x_into_T(qc, x, T)

    # 2) prepare O superposition (offset register)
    prepare_offset_O(qc, O)

    # 3) coherently add O into T: T := T + O
    #add_offset_into_T(qc, O, T, ANC)
    qc.x(x[0]) # NOT x[0] so that the MINUS1 operator is controlled on |x> being even
    controlled_decrement_by_one_two_controls(qc, O[0], x[0], T, ANC, ANC_2)
    qc.x(x[0])

    # 4) divide T by 2 by shifting all qubits by one
    divide_T_by_2(qc, T)

    # 5) uncompute O using the cheap 8-CNOT routine (clears O and R)
    uncompute_offset_xor(qc, x, T, O, R)

    # After this qc does: |x>|0^m>|0^2>|0^2>|0^m>  -> |x>|T=2x+offset>|00>|00>|0^m>
    # with O and R returned to zero and anc returned to zero.
    # T holds the desired superposition over 2x,2x+1,2x+2 depending on O.
    return qc, {'x': x, 'T': T, 'O': O, 'R': R, 'anc': ANC}

# ---------------- Example and statevector test -------------------------------

if __name__ == "__main__":
    # try a small example a = 3
    a = 3
    qc, regs = build_shift_isometry_circuit(a)
    print(qc.draw(fold=120))

    # simulate the circuit on each basis |x>|0^m> and check ancilla outputs
    backend = Aer.get_backend('statevector_simulator')
    sv = Statevector.from_instruction(qc)

    # test mapping on basis states: we will check the reduced state on x+T qubits
    # index order in our circuit: [x0, x1, ..., x_{a-1}, T0, T1, ..., T_{m-1}, O0,O1, R0,R1, anc...]
    # We'll inspect amplitudes for a few x values:
    dim_x = 1 << (a+1)
    m = a + 1
    for x_val in range(dim_x):
        # prepare input basis state |x>|0^rest> as full statevector
        init_sv = Statevector.from_label('0' * (a + m + 2 + 2 + m))  # total qubits
        # flip bits of x to represent x_val
        bits = [(x_val >> i) & 1 for i in range(a+1)]
        qc_in = QuantumCircuit(a + m + 2 + 2 + m)
        # set x bits
        for i, b in enumerate(bits):
            if b:
                qc_in.x(i)  # our x register is first in qc definition
        sv_in = Statevector.from_instruction(qc_in)
        # apply the constructed unitary qc to sv_in
        sv_out = sv_in.evolve(qc)
        # extract reduced state on x+T to view output
        # We can measure the probability mass on basis states |x>|2x>, |x>|2x+1|, |x>|2x+2|
        # Compute amplitude sum
        def idx_of(xi, ti):
            # index ordering as in qc: x bits first (LSB at lower index), then T
            # the Qiskit Statevector index convention for labels uses reversed bit order when using .data directly,
            # so for robust inspection use to_dict or probabilities. For brevity we just compute probabilities by marginalizing.
            pass

        # print all the non-zero amplitudes:
        full_amp_dict = {}
        for i,val in enumerate(sv_out.data):
            if abs(val) > 1E-8:
                full_amp_dict[bin(i)[2:].zfill(2*m+2)] = val
        print("Full Amplitude Dictionary: ", full_amp_dict)

        # print all of the amplitudes for the T register given the x register is in the original postion and the other registers are all |0>
        '''target_amps = {}
        for t_val in range(2**m):
            # build full string label in Qiskit's bitstring order: left-most is highest index qubit
            # Our qc qubits order: [x0,...,x_{a-1}, T0,...,T_{m-1}, O0,O1, R0,R1, anc...]
            # Qiskit labels bitstrings highest->lowest by index, so compose accordingly:
            # We'll construct labels by building bit list for all qubits in same index order then reverse to string
            total_qubits = a + m + 2 + 2 + m
            bits_list = [0] * total_qubits
            # set x bits
            for i in range(a):
                bits_list[i] = (x_val >> i) & 1
            # set T bits
            for i in range(m):
                bits_list[a + i] = (t_val >> i) & 1
            # others already zero
            bitstring = ''.join(str(b) for b in reversed(bits_list))  # reverse for Qiskit ordering label
            target_amps[t_val] = sv_out.data[int(bitstring,2)]
        print(target_amps)'''