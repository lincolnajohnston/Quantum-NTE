from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate, ZGate
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.aerprovider import QasmSimulator
import math
import numpy as np

def random_unitary(n, seed=None):
    """
    Generate a random n×n unitary matrix using the Haar measure.

    Parameters
    ----------
    n : int
        Dimension of the matrix.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    U : np.ndarray (complex)
        Random unitary matrix (U†U = I).
    """
    if seed is not None:
        np.random.seed(seed)

    # Create a random complex matrix (entries ~ N(0,1) + i*N(0,1))
    Z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)

    # QR decomposition
    Q, R = np.linalg.qr(Z)

    # Normalize Q to ensure uniform Haar distribution
    D = np.diag(R) / np.abs(np.diag(R))
    U = Q @ np.diag(D.conj())

    return U

###############################################################################
# 1. User-provided pieces you must define for YOUR problem
###############################################################################

def prepare_state(sys_reg, anc_reg):
    """
    Build the circuit that prepares the *input state* |Psi>
    on ancillas+system BEFORE applying U_A.
    
    For the demo:
    - put ancilla in |0>
    - put system in some superposition cosθ|0> + sinθ|1>
    You will replace this with however you normally load |psi>.
    """
    qc = QuantumCircuit(anc_reg, sys_reg, name="Prep")

    # Example: rotate system qubit so it's not |0>
    theta = 0.7  # arbitrary
    for sys in sys_reg:
        qc.ry(theta, sys)

    # vector of initial_state
    RY = np.array([[math.cos(theta/2), -math.sin(theta/2)],[math.sin(theta/2), math.cos(theta/2)]])
    RY_kron = RY
    for _ in range(len(sys_reg)):
        RY_kron = np.kron(RY_kron, RY)
    initial_state = RY_kron[:,0] # circuit is initialized to the all zeros state so the state prepared is just the first column of RY_kron

    # ancilla left in |0>, which is fine
    return qc

# dilator matrix that does the transformation E|x> = 0.5|(x-offset) mod N> + 1|x> + 0.5|(x+offset) mod N>
# BE=true returns the entire unitary matrix that would block encode E
# gap_ancillas is the number of ancillas placed between the main register and the LCU ancillas when BE=true
def get_E(qc, anc_reg, sys_reg, n, offset = 1, BE=False, prefix_ancillas = 0, gap_ancillas=0):
    N = int(2**n)
    if BE == False:
        matrix = np.zeros((N, N))
        matrix[0:N,0:N] += np.diag(np.ones(N)) # diagonal terms, identity matrix O(1) time to apply

        # |x> -> 0.5|(x-offset) mod N>
        # |x> -> 1|(x-offset) mod N> is unitary and can be implemented in O(polylog(N) time)
        matrix[0:N,0:N] += np.diag(0.5 * np.ones(N-offset), k=offset)  # |x> -> |x-offset> term when x >= offset
        matrix[0:N,0:N] += np.diag(0.5 * np.ones(offset), k=offset-N) # |x> -> |x-offset> term when x < offset

        # |x> -> 0.5|(x+offset) mod N>
        # |x> -> 1|(x+offset) mod N> is unitary and can be implemented in O(polylog(N) time)
        matrix[0:N,0:N] += np.diag(0.5 * np.ones(offset), k=N-offset)  # |x> -> |x+offset> term when x >= N - offset
        matrix[0:N,0:N] += np.diag(0.5 * np.ones(N-offset), k=-offset) # |x> -> |x+offset> term when x < N - offset

        # Each of the three terms can be implemented as separate unitary matrices. Then LCU can be used to sum them, with alpha values of 0.5, 1, and 0.5
        # the time complexity of doing LCU is O(alpha) (which I think comes from the post-selection procedure, which you can do amplitude amplification for 
        # which takes O(alpha) rotations to get a high probability of success).
        # So alpha=2, E/2 can be block-encoded in O(polylog(N) + alpha) = O(polylog(N)) time
        return matrix
    else:
        # Use LCU to block encode
        gap_ancilla_N = int(2**gap_ancillas)
        total_N = gap_ancilla_N*N
        V = np.array([[1/math.sqrt(2), 1/math.sqrt(2), 0, 0],
                      [1/2, -1/2, 0, 1/math.sqrt(2)],
                      [1/2, -1/2, 0, -1/math.sqrt(2)],
                      [0, 0, 1, 0]])
        V_inv = np.transpose(V)
        I_N = np.eye(total_N)
        E_matrix = np.eye(4*total_N)

        # apply the V matrix
        E_matrix = np.kron(V, I_N) @ E_matrix

        # apply the 3 summands of the LCU controlled on the ancillas V was applied to
        # identity matrix
        ident_control = np.eye(4*total_N)
        ident_control[:total_N,:total_N] = np.eye(total_N)
        E_matrix = ident_control @ E_matrix

        # positive integer shift matrix
        int_plus = np.diag(np.ones(N-offset), k=-offset) + np.diag(np.ones(offset), k=N-offset)
        int_plus = np.kron(np.eye(gap_ancilla_N), int_plus)
        int_plus_control = np.eye(4*total_N)
        int_plus_control[total_N:2*total_N,total_N:2*total_N] = int_plus
        E_matrix = int_plus_control @ E_matrix

        # negative integer shift matrix
        int_minus = np.diag(np.ones(N-offset), k=offset) + np.diag(np.ones(offset), k=offset-N)
        int_minus = np.kron(np.eye(gap_ancilla_N), int_minus)
        int_minus_control = np.eye(4*total_N)
        int_minus_control[2*total_N:3*total_N,2*total_N:3*total_N] = int_minus
        E_matrix = int_minus_control @ E_matrix

        # apply the inverse of V
        E_matrix = np.kron(V_inv, I_N) @ E_matrix

        E_matrix = np.kron(np.eye(int(2**prefix_ancillas)), E_matrix) # prefix ancillas not used right now
        
        E_matrix = np.kron(np.eye(int(N/4)), E_matrix)

        U_A_gate = UnitaryGate(E_matrix)
        qc.append(U_A_gate, anc_reg[0:n] + sys_reg[0:n])

    return qc



def U_A_block_encoding(sys_reg, anc_reg, inverse=False):
    """
    Build your block-encoding unitary U_A.
    Assumption: measuring anc_reg==|0...0> after applying this
    corresponds to having applied (A/alpha) to the system.
    
    For demo:
    We'll do a simple controlled rotation from ancilla to system,
    just so ancilla=|0> has some amplitude correlation.
    
    In reality THIS is where you put your LCU / SELECT / PREPARE
    gadget that implements the block-encoding.
    """
    if len(anc_reg) != len(sys_reg):
        raise ValueError("Must be same number of ancillas and system qubits for this definition of U_A")
    n = len(anc_reg)
    qc = QuantumCircuit(anc_reg, sys_reg, name="U_A")

    '''# Demo trick:
    # We'll entangle ancilla and system in a nontrivial way so that
    # postselecting ancilla=|0> does something non-identity on system.
    #
    # (1) Hadamard on ancilla to create success/failure branches
    qc.h(anc_reg[0])

    # (2) Controlled unitary from ancilla->system.
    # Let's do a controlled-Rx as a stand-in for "applying A/alpha".
    angle = 1.2
    qc.crx(angle, anc_reg[0], sys_reg[0])

    # (3) Hadamard again on ancilla (this mimics an LCU-style uncompute)
    qc.h(anc_reg[0])

    # add an X gate to make the probability of success lower
    qc.x(anc_reg[0])

    return qc.inverse() if inverse else qc'''


    U_A = random_unitary(int(2**(2*n)), seed=10811)
    if inverse:
        U_A_inv = np.linalg.inv(U_A)
        U_A_inv_gate = UnitaryGate(U_A_inv)
        qc.append(U_A_inv_gate, anc_reg[0:n] + sys_reg[0:n])
    else:
        U_A_gate = UnitaryGate(U_A)
        qc.append(U_A_gate, anc_reg[0:n] + sys_reg[0:n])

    return qc


###############################################################################
# 2. Oracles and reflections needed for amplitude amplification
###############################################################################

def O_good_phaseflip(sys_reg, anc_reg):
    """
    Phase flip on the 'good' ancilla pattern.
    Here 'good' = ancilla all |0>.
    
    Implementation: apply Z with ancilla as a multi-controlled Z *on nothing*
    which is equivalent to an n-controlled phase on |0...0>.
    For 1 ancilla, that's just a Z on |0>, which is X-Z-X.
    
    If you have multiple ancillas, you'd build a multi-controlled Z
    targeting an extra work qubit or use mct().
    """
    if len(anc_reg) != len(sys_reg):
        raise ValueError("Must be same number of ancillas and system qubits for this definition of U_A")
    n = len(anc_reg)
    qc = QuantumCircuit(anc_reg, sys_reg, name="O_good")

    # For single ancilla:
    # We want: |0>_anc -> phase -1. That's NOT the usual Z (which flips phase on |1>).
    # So do X - Z - X on the ancilla.
    #qc.x(anc_reg[0])
    #qc.z(anc_reg[0])
    #qc.x(anc_reg[0])
    for i in range(n):
        qc.x(anc_reg[i])
    cz = ZGate().control(n-1) if n > 1 else ZGate()
    qc.append(cz,anc_reg[1:n] + [anc_reg[0]])
    for i in range(n):
        qc.x(anc_reg[i])

    # for multipe ancilla, flip only if all of the ancillas are |0>

    return qc

def O_all_phaseflip(sys_reg, anc_reg):
    if len(anc_reg) != len(sys_reg):
        raise ValueError("Must be same number of ancillas and system qubits for this definition of U_A")
    n = len(anc_reg)
    qc = QuantumCircuit(anc_reg, sys_reg, name="O_good")

    for i in range(n):
        qc.x(anc_reg[i])
    for i in range(n):
        qc.x(sys_reg[i])
    cz = ZGate().control(2*n-1)
    qc.append(cz,anc_reg[1:n] + sys_reg[0:n] + [anc_reg[0]])
    for i in range(n):
        qc.x(anc_reg[i])
    for i in range(n):
        qc.x(sys_reg[i])

    # for multipe ancilla, flip only if all of the ancillas are |0>

    return qc

def reflection_about_initial(sys_reg, anc_reg):
    """
    Implements (2|Psi><Psi| - I) where |Psi> is the *initial combined state*
    BEFORE U_A is applied.
    
    Standard trick:
    R_init = Prep * ( 2|0...0><0...0| - I ) * Prep†
    
    We'll build a little wrapper that returns a circuit for R_init,
    given access to 'prepare_state'.
    """
    # Step 1: make Prep
    prep = prepare_state(sys_reg, anc_reg)

    # Step 2: make reflection about |0...0> state on *all* qubits in anc+sys.
    # For N qubits total, reflection about |0...0> is:
    # X on all qubits
    # multi-controlled Z (phase flip on |11...1>)
    # X on all qubits
    #
    # We'll implement this generically for anc+sys together.
    total_qubits = list(anc_reg) + list(sys_reg)
    N = len(total_qubits)

    refl0 = QuantumCircuit(anc_reg, sys_reg, name="R_zero")

    # X on all
    for q in total_qubits:
        refl0.x(q)

    # multi-controlled Z on last qubit, controlled by others
    # If only 1 qubit total, just do Z.
    if N == 1:
        refl0.z(total_qubits[0])
    else:
        controls = total_qubits[:-1]
        target = total_qubits[-1]
        # mcphase π == mct with Z? We can do mct on X basis trick:
        # Easiest in Qiskit: mct for a multi-controlled X, then wrap with H to turn X into Z.
        # We'll do: H(target); mct(controls, target); H(target)
        refl0.h(target)
        refl0.mct(controls, target)  # multi-controlled Toffoli
        refl0.h(target)

    # X on all
    for q in total_qubits:
        refl0.x(q)

    # Now assemble R_init = Prep * R_zero * Prep†
    R_init = QuantumCircuit(anc_reg, sys_reg, name="R_init")
    R_init.compose(prep, list(anc_reg)+list(sys_reg), inplace=True)
    R_init.compose(refl0, list(anc_reg)+list(sys_reg), inplace=True)
    R_init.compose(prep.inverse(), list(anc_reg)+list(sys_reg), inplace=True)

    return R_init


###############################################################################
# 3. Build one Grover-style amplitude amplification iteration
###############################################################################

def amplitude_amplification_iteration(sys_reg, anc_reg, states_tracker, iters=1):
    """
    Construct ONE full amplitude amplification iteration for the block-encoding.
    
    Q = (Prep U_A) O_good (U_A)† (Prep)† R_init (Prep U_A)
    
    After applying this Q once to |0...0>, amplitude on the "good ancilla"
    subspace is boosted.
    
    We'll return a circuit that:
    - Prepares |Psi>
    - Applies U_A
    - Does the Grover iterate
    (So the *output state* of this circuit is the amplified state.)
    """
    qc = QuantumCircuit(anc_reg, sys_reg, name="AA_iter")

    # We'll need subcircuits:
    prep = prepare_state(sys_reg, anc_reg)

    # using E as block encoding
    #g = 1
    #UA = get_E(qc, anc_reg, sys_reg, len(sys_reg), offset=g, BE=True)
    #UA_dag = UA.inverse()

    # arbitrary block encoding
    UA = U_A_block_encoding(sys_reg, anc_reg)
    #UA_dag = UA.inverse()
    UA_dag = U_A_block_encoding(sys_reg, anc_reg, inverse=True)

    R_init = reflection_about_initial(sys_reg, anc_reg)
    Og = O_good_phaseflip(sys_reg, anc_reg)
    Oall = O_all_phaseflip(sys_reg, anc_reg)

    # STEP 1: Prep U_A
    qc.compose(prep, list(anc_reg)+list(sys_reg), inplace=True)
    qc.compose(UA,   list(anc_reg)+list(sys_reg), inplace=True)
    states_tracker.append(Statevector.from_instruction(qc))

    # for oblivious amplitude amplification, we should be able to do this iteration without calling the "prep" gate, as
    # we want to be able to do this on any arbitrary quantum state without knowing the state or being able to prepare it
    for _ in range(iters):
        # STEP 2: O_good
        qc.compose(Og, list(anc_reg)+list(sys_reg), inplace=True)

        # STEP 3: U_A^† Prep^†
        qc.compose(UA_dag, list(anc_reg)+list(sys_reg), inplace=True)
        #qc.compose(prep.inverse(), list(anc_reg)+list(sys_reg), inplace=True)

        # STEP 4: R_init  (this is reflection about |Psi>)
        #qc.compose(R_init, list(anc_reg)+list(sys_reg), inplace=True) # I think R_init and Oall are the same thing
        qc.compose(Oall, list(anc_reg)+list(sys_reg), inplace=True)

        # STEP 5: Prep U_A again
        #qc.compose(prep, list(anc_reg)+list(sys_reg), inplace=True)
        qc.compose(UA,   list(anc_reg)+list(sys_reg), inplace=True)

        states_tracker.append(Statevector.from_instruction(qc))
    return qc


###############################################################################
# 4. Example: build + measure
###############################################################################

def build_full_experiment(measure=True):
    # one ancilla qubit, one system qubit
    anc_bits = 4
    sys_bits = 4
    anc = QuantumRegister(anc_bits, 'anc')
    sys = QuantumRegister(sys_bits, 'sys')
    c_anc = ClassicalRegister(1, 'c_anc')
    c_sys = ClassicalRegister(1, 'c_sys')

    states_tracker = []
    aa = amplitude_amplification_iteration(sys, anc, states_tracker, iters=4)

    for state in states_tracker:
        #print("State: ", np.round(state.data, decimals=4))

        # for the 1 ancilla, 1 system qubit case, get the angle of the current state
        phi_final = np.array([state.data[int(2**anc_bits) * i] for i in range(int(2**sys_bits))])
        phi_final_mag = np.linalg.norm(phi_final)
        phi_final_angle = math.asin(phi_final_mag)

        print("Magnitude: ", phi_final_mag)
        print("Angle: ", phi_final_angle, "\n")

    # now let's measure to see boosted success probability in ancilla=|0>
    
    if measure:
        full = QuantumCircuit(anc, sys, c_anc, c_sys)
        full.compose(aa, [anc[:anc_bits], sys[:sys_bits]], inplace=True)
        full.measure(anc[:anc_bits], c_anc[:anc_bits])
        full.measure(sys[:sys_bits], c_sys[:sys_bits])
        return full
    return aa

###############################################################################
# 5. Demo usage
###############################################################################

if __name__ == "__main__":
    sim_method = "statevector"
    circ = build_full_experiment(measure=(sim_method=="counts"))
    print(circ.draw(fold=120))
    # You can run this on Aer:
    #

    if sim_method == "counts":
    # counts simulation
        simulator = AerSimulator() 
        compiled_circuit = transpile(circ, simulator)
        job = simulator.run(compiled_circuit, shots=1000000)
        result = job.result()
        counts = result.get_counts(circ)
        print(counts)
    
    #statevector simulation
    if sim_method == "statevector":
        circ.save_statevector()

        # Run emulator in statevector mode
        backend = QasmSimulator(method="statevector")
        new_circuit = transpile(circ, backend)
        #print(dict(new_circuit.count_ops())) # print the counts of each type of gate
        job = backend.run(new_circuit)
        job_result = job.result()

        # print statevector of non-junk qubits
        state_vec = job_result.get_statevector(circ).data
        print(state_vec)

        # for the 1 qubit system and ancilla case, get the magnitude of the dsired portion of the final state
        phi_final = np.array([state_vec[0], state_vec[2]])
        phi_final_mag = np.linalg.norm(phi_final)
        phi_final_angle = math.asin(phi_final_mag)
        
    #
    # Look at probability anc='0'. That is your boosted post-selection success.
    circ.draw('mpl', filename="amplitude-amplification-circuit.png")
    print("sim done")