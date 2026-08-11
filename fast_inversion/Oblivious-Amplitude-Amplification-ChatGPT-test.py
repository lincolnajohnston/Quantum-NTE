from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate, ZGate
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer, AerSimulator, QasmSimulator
import numpy as np
import math
import random

# This script is to test whether we can do oblivious amplitude amplification when given a superposition of states, some 
# of which having a flag set to |0> and others with that flag set to |1>. We want to effectively renormalize the states
# and discard the states with the |1> flag state.

# For our quantum Monte Carlo k-eigenvalue solve, this is useful because then we could apply OAA

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

def phase_flip_on_flag_zero(qc: QuantumCircuit, flag_qubit):
    """
    Implements diag(-1, +1) on the flag qubit: phase flip on |0>.
    Exactly X-Z-X.
    """
    qc.x(flag_qubit)
    qc.z(flag_qubit)
    qc.x(flag_qubit)

def oaa_iterate(qc: QuantumCircuit, U: QuantumCircuit, Udg: QuantumCircuit, data_qubits, flag_qubit):
    """
    Appends one oblivious amplitude amplification iterate:
        Q = - U S0 U† S0
    Global phase (-1) is irrelevant and not implemented.
    """
    # S0 on flag
    phase_flip_on_flag_zero(qc, flag_qubit)

    # U†
    qc.append(Udg.to_gate(), list(data_qubits) + [flag_qubit])

    # S0 on flag
    phase_flip_on_flag_zero(qc, flag_qubit)

    # U
    qc.append(U.to_gate(), list(data_qubits) + [flag_qubit])

def build_oaa_circuit(U: QuantumCircuit, data_qubits, flag_qubit, t: int):
    """
    Returns a circuit that assumes input state is already in the registers,
    then applies U and t OAA iterations to amplify flag=0 outcomes.
    """
    Udg = U.inverse()

    qc = QuantumCircuit(*U.qregs)  # assumes U is built on the same registers
    qc.append(U.to_gate(), list(data_qubits) + [flag_qubit])

    for _ in range(t):
        # Q = - U S0 U† S0, applied to the current state
        # Implemented as: S0, U†, S0, U  (same operator up to global phase)
        phase_flip_on_flag_zero(qc, flag_qubit)
        qc.append(Udg.to_gate(), list(data_qubits) + [flag_qubit])
        phase_flip_on_flag_zero(qc, flag_qubit)
        qc.append(U.to_gate(), list(data_qubits) + [flag_qubit])

    return qc

if __name__ == "__main__":
    anc_bits = 1
    sys_bits = 2
    anc = QuantumRegister(anc_bits, 'anc')
    sys = QuantumRegister(sys_bits, 'sys')
    c_anc = ClassicalRegister(1, 'c_anc')
    c_sys = ClassicalRegister(1, 'c_sys')

    # create the U operator that we are going to use inside of the OAA procedure, should be 
    # able to be any operation on all system qubits and the ancilla qubit (including ones that entangle the two registers)
    U_circuit = QuantumCircuit(sys, anc, name="AA_iter")
    # applying single qubit gates to the circuit
    '''angle_rad = 3 # angle in radians that we are rotating the ancilla qubit
    U_circuit.ry(angle_rad,4) #applying a angle_rad rotation to the ancilla qubit (qubit index 4) 
    for i in range(sys_bits): # to each of the system qubits, apply a Hadamard gate, this should be able to be any arbitrary gate
        U_circuit.h(i)'''
    
    # applying a random entangling unitary to the entire circuit
    '''n = anc_bits + sys_bits
    U_A = random_unitary(int(2**(n)), seed=136804)
    U_A_gate = UnitaryGate(U_A)
    U_circuit.append(U_A_gate, list(range(sys_bits + 1)))'''

    # applying a custom unitary to a 2 qubit system register case
    U_A = [[math.sqrt(1)/math.sqrt(16),0,0,0,-math.sqrt(15)/math.sqrt(16),0,0,0],
           [0,math.sqrt(1)/math.sqrt(15),0,0,0,0,-math.sqrt(14)/math.sqrt(15),0],
           [0,0,0,math.sqrt(1)/math.sqrt(14),0,0,0,-math.sqrt(13)/math.sqrt(14)],
           [0,0,math.sqrt(1)/math.sqrt(13),0,0,-math.sqrt(12)/math.sqrt(13),0,0],
           [math.sqrt(15)/math.sqrt(16),0,0,0,math.sqrt(1)/math.sqrt(16),0,0,0],
           [0,0,math.sqrt(12)/math.sqrt(13),0,0,math.sqrt(1)/math.sqrt(13),0,0],
           [0,0,0,math.sqrt(13)/math.sqrt(14),0,0,0,math.sqrt(1)/math.sqrt(14)],
           [0,math.sqrt(14)/math.sqrt(15),0,0,0,0,math.sqrt(1)/math.sqrt(15),0]]
    '''U_A = [[0,0,0,0,1,0,0,0],
           [0,0,0,0,0,1,0,0],
           [0,0,0,0,0,0,1,0],
           [0,0,0,0,0,0,0,1],
           [1,0,0,0,0,0,0,0],
           [0,1,0,0,0,0,0,0],
           [0,0,1,0,0,0,0,0],
           [0,0,0,1,0,0,0,0]]'''
    U_A_gate = UnitaryGate(U_A)
    U_circuit.append(U_A_gate, list(range(sys_bits + 1)))

    print("qc circuit:")
    print(U_circuit.draw(fold=120))

    oaa = build_oaa_circuit(U_circuit, list(range(sys_bits)), [sys_bits], t=1) # set the t here to the number of Grover iterations you want
    print("OAA circuit: ")
    print(oaa.draw(fold=120))

    circ = QuantumCircuit(sys, anc, name="full_circuit")
    # set the system register to some random state |psi>
    np.random.seed(13962037)
    for i in range(sys_bits):
        circ.ry(np.random.random() * 3, i)
    circ.compose(oaa,  inplace=True)
    print("full circuit:")
    print(circ.draw(fold=120))

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

    # for the 1 qubit ancilla case, get the magnitude of the desired portion of the final state
    phi_final_mag = np.linalg.norm(state_vec[:int(len(state_vec)/2)])
    phi_final_angle = math.asin(phi_final_mag)
    print("good state magnitude: ", phi_final_mag) # If this magnitude approaches 1 then you are approaching a perfect amplitude amplification
    print("good state angle: ", phi_final_angle) # this angle should approach pi/2 as the amplitude amplification gets better

