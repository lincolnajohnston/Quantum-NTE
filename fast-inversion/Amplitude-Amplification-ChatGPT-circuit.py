from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.aerprovider import QasmSimulator
import math
import numpy as np

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
    qc.ry(theta, sys_reg[0])

    # ancilla left in |0>, which is fine
    return qc


def U_A_block_encoding(sys_reg, anc_reg):
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
    qc = QuantumCircuit(anc_reg, sys_reg, name="U_A")

    # Demo trick:
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

    # what U_A look like in matrix form
    H = 1/math.sqrt(2) * np.array([[1, 1],[1, -1]])
    X = np.array([[0, 1],[1, 0]])
    HI = np.kron(H, np.eye(2))
    XI = np.kron(X, np.eye(2))
    CRX = np.array([[1,0,0,0],[0,1,0,0],[0,0,math.cos(0.6),-1j*math.sin(0.6)],[0,0,-1j*math.sin(0.6),math.cos(0.6)]])
    U_A = XI @ HI @ CRX @ HI

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
    qc = QuantumCircuit(anc_reg, sys_reg, name="O_good")

    # For single ancilla:
    # We want: |0>_anc -> phase -1. That's NOT the usual Z (which flips phase on |1>).
    # So do X - Z - X on the ancilla.
    qc.x(anc_reg[0])
    qc.z(anc_reg[0])
    qc.x(anc_reg[0])

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

def amplitude_amplification_iteration(sys_reg, anc_reg, iters=1):
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
    UA = U_A_block_encoding(sys_reg, anc_reg)
    UA_dag = UA.inverse()
    R_init = reflection_about_initial(sys_reg, anc_reg)
    Og = O_good_phaseflip(sys_reg, anc_reg)

    # STEP 1: Prep U_A
    qc.compose(prep, list(anc_reg)+list(sys_reg), inplace=True)
    qc.compose(UA,   list(anc_reg)+list(sys_reg), inplace=True)

    for _ in range(iters):
        # STEP 2: O_good
        qc.compose(Og, list(anc_reg)+list(sys_reg), inplace=True)

        # STEP 3: U_A^† Prep^†
        qc.compose(UA_dag, list(anc_reg)+list(sys_reg), inplace=True)
        #qc.compose(prep.inverse(), list(anc_reg)+list(sys_reg), inplace=True)

        # STEP 4: R_init  (this is reflection about |Psi>)
        #qc.compose(R_init, list(anc_reg)+list(sys_reg), inplace=True)
        qc.compose(Og, list(anc_reg)+list(sys_reg), inplace=True)

        # STEP 5: Prep U_A again
        #qc.compose(prep, list(anc_reg)+list(sys_reg), inplace=True)
        qc.compose(UA,   list(anc_reg)+list(sys_reg), inplace=True)

    return qc


###############################################################################
# 4. Example: build + measure
###############################################################################

def build_full_experiment(measure=True):
    # one ancilla qubit, one system qubit
    anc = QuantumRegister(1, 'anc')
    sys = QuantumRegister(1, 'sys')
    c_anc = ClassicalRegister(1, 'c_anc')
    c_sys = ClassicalRegister(1, 'c_sys')

    aa = amplitude_amplification_iteration(sys, anc, iters=2)

    # now let's measure to see boosted success probability in ancilla=|0>
    full = QuantumCircuit(anc, sys, c_anc, c_sys)
    full.compose(aa, [anc[0], sys[0]], inplace=True)
    if measure:
        full.measure(anc[0], c_anc[0])
        full.measure(sys[0], c_sys[0])
    return full

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
    print("sim done")