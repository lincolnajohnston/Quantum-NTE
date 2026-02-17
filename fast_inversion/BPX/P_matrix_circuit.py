# Efficient gate-level implementation (Qiskit)
# Requires: qiskit
# Tested conceptually; please run locally in your environment.

from math import pi, acos, sqrt
from qiskit import QuantumCircuit, QuantumRegister, Aer, transpile
from qiskit.quantum_info import Statevector

def prepare_offset_O(qc, O):
    """
    Prepare the 2-qubit offset state
      1/2 |00> + 1/sqrt(2) |01> + 1/2 |10>
    on O[0] (LSB), O[1] (MSB).
    Uses Ry(pi/3) on O[0] and a controlled Ry on O[1].
    """

    # switched qubits
    #angle for first qubit: Ry(pi/3)
    qc.ry(pi/3, O[1])

    # controlled prepare second qubit conditioned on O[0] == 0.
    # We implement "if O[0]==0 then Ry(phi) on O[1]" by X, CRY, X.
    phi = 2 * acos(1.0 / sqrt(3.0))  # 2*arccos(1/sqrt(3))
    qc.x(O[1])
    qc.cry(phi, O[1], O[0])  # controlled-Ry
    qc.x(O[1])

    # original offset implementation
    '''# angle for first qubit: Ry(pi/3)
    qc.ry(pi/3, O[0])

    # controlled prepare second qubit conditioned on O[0] == 0.
    # We implement "if O[0]==0 then Ry(phi) on O[1]" by X, CRY, X.
    phi = 2 * acos(1.0 / sqrt(3.0))  # 2*arccos(1/sqrt(3))
    qc.x(O[0])
    qc.cry(phi, O[0], O[1])  # controlled-Ry
    qc.x(O[0])'''

def compute_2x_into_T(qc, x, T):
    """
    Compute T := 2*x (i.e., copy x into T shifted by 1 bit).
    Assumes x has length a, T has length a+1 and T initially |0>.
    Uses CNOT(x[i] -> T[i+1]) for i=0..a-1.
    LSB convention: x[0] is least significant.
    """
    a = len(x)
    for i in range(a):
        qc.cx(x[i], T[i+1])

def uncompute_offset_xor(qc, x, T, O, R):
    """
    The small 8-CNOT uncompute routine described earlier.
    - x: system register (x[0] is LSB)
    - T: target register (T[0] is LSB, T[1] next)
    - O: 2-qubit offset register (O[0] LSB, O[1] MSB) to be cleared
    - R: 2-qubit scratch register (init |00>), R[0] LSB, R[1] MSB
    After this, O returns to |00> and R returns to |00>, T and x unchanged.
    """
    # copy least significant bits of T into the R ancillas
    # if x was even, then |R> will be 0.5|0> + 1/sqrt(2)|1> + 0.5|2>
    # if x was odd, then |R> will be  0.5|2> + 1/sqrt(2)|3> + 0.5|0>
    qc.cx(T[0], R[0])
    qc.cx(T[1], R[1])

    # add 2 to the R ancilla if x is odd so that |R> = 0.5|0> + 1/sqrt(2)|1> + 0.5|2>
    # this is now exactly the same as the offset register, O
    qc.cx(x[0], R[1])

    # subtract each bit of R from each bit of O, which returns O to |00> because |R> = |O>
    qc.cx(R[0], O[0])
    qc.cx(R[1], O[1])

    # reverse all of the operations done on |R> so that it goes back to |00>
    qc.cx(x[0], R[1])
    qc.cx(T[1], R[1])
    qc.cx(T[0], R[0])

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
    # add O[1] * 2  (start=1) using anc_for_inc[1:len(T)-1]
    # We'll pass anc slice anc_for_inc[1:] and start=1
    controlled_increment(qc, O[1], T, anc_for_inc[1:], start=1)

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
    x = QuantumRegister(a, 'x')       # system
    T = QuantumRegister(m, 'T')       # target ancilla (holds 2x + offset)
    O = QuantumRegister(2, 'O')       # offset preparer
    R = QuantumRegister(2, 'R')       # small scratch for uncompute_offset_xor
    # ancillas for ripple increments: need length m (for start=0 case)
    ANC = QuantumRegister(m, 'anc')   # these will be returned to zero by design

    qc = QuantumCircuit(x, T, O, R, ANC, name=f"shift_iso_a{a}")

    # 1) compute T := 2*x
    compute_2x_into_T(qc, x, T)

    # 2) prepare O superposition (offset register)
    prepare_offset_O(qc, O)

    # 3) coherently add O into T: T := T + O
    add_offset_into_T(qc, O, T, ANC)

    # 4) uncompute O using the cheap 8-CNOT routine (clears O and R)
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
    dim_x = 1 << a
    m = a + 1
    for x_val in range(dim_x):
        # prepare input basis state |x>|0^rest> as full statevector
        init_sv = Statevector.from_label('0' * (a + m + 2 + 2 + m))  # total qubits
        # flip bits of x to represent x_val
        bits = [(x_val >> i) & 1 for i in range(a)]
        qc_in = QuantumCircuit(a + m + 2 + 2 + m)
        # set x bits
        for i, b in enumerate(bits):
            if b:
                qc_in.x(i)  # our x register is first in qc definition
        #qc_in.h(0) # alternate initial state
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

        # For brevity in this code sample, we'll compute probabilities by measuring the full statevector
        probs = sv_out.probabilities_dict()

        # print just the probabilities from the expected locations
        # sum probabilities where the x register equals x_val and T register is 2x,2x+1,2x+2 and other ancillas are 0
        '''target_probs = {}
        for t_val in (2*x_val, 2*x_val+1, 2*x_val+2):
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
            target_probs[t_val] = probs.get(bitstring, 0.0)
        print(f"x={x_val:>2}: Prob(2x)={target_probs[2*x_val]:.6f}, Prob(2x+1)={target_probs[2*x_val+1]:.6f}, Prob(2x+2)={target_probs[2*x_val+2]:.6f}")'''

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