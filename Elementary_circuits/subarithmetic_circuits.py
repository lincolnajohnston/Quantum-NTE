from qiskit.circuit.library import UnitaryGate
import numpy as np

# |x> -> T†|x>
def T_dagger_gate(qc, x, gate_counts, gate_types="elementary"):
    """Apply the inverse T phase using the Clifford+T gate set.
    Add the Clifford and T gate totals to ``gate_counts`` in place.
    ``gate_types`` selects the Clifford+T decomposition (``"elementary"``),
    one built-in gate (``"single"``), or counting without application (``"none"``).

    Resource Requirements:
    - Gates: 1 S-dagger gate and 1 T gate
    - Ancillas: 0
    - Post-selection qubits: 0
    """
    if gate_types != "elementary" and gate_types != "single" and gate_types != "none":
        gate_types = "elementary" # if an invalid gate type is input, set the gate type to the default "elementary"

    if gate_types=="elementary":
        qc.sdg(x[0])
        qc.t(x[0])
    gate_counts[0] += 1 # add 1 to the Clifford gate count
    gate_counts[1] += 1 # add 1 to the T gate count

    # apply T_dagger as a single gate
    if gate_types == "single":
        qc.tdg(x[0])
    

# |x> -> V|x> (or V†|x>)
def V_gate(qc, x, gate_counts, dagger=False, gate_types="elementary"):
    """Apply the square root of X, or its inverse, to one qubit.
    Add the Clifford and T gate totals to ``gate_counts`` in place.

    Resource Requirements:
    - Gates: 2 H gates and 1 S (or S-dagger) gate
    - Ancillas: 0
    - Post-selection qubits: 0
    """
    if gate_types != "elementary" and gate_types != "single" and gate_types != "none":
            gate_types = "elementary" # if an invalid gate type is input, set the gate type to the default "elementary"
    
    if gate_types=="elementary": 
        qc.h(x[0])
    gate_counts[0] += 1

    if gate_types=="elementary":
        qc.sdg(x[0]) if dagger else qc.s(x[0])
    gate_counts[0] += 1

    if gate_types=="elementary":
        qc.h(x[0])
    gate_counts[0] += 1

    # Apply V as a single gate
    if gate_types=="single":
        qc.sxdg(x[0]) if dagger else qc.sx(x[0])


# |c>|t> -> |c>S^c|t> (or |c>(S†)^c|t>)
def CS_gate(qc, x, gate_counts, dagger=False, gate_types="elementary"):
    """Apply a controlled S gate, or its inverse, with ``x[1]`` controlling ``x[0]``.
    Add the Clifford and T gate totals to ``gate_counts`` in place.

    Resource Requirements:
    - Gates: 2 CNOT, 3 T, and 2 S-dagger gates
    - Ancillas: 0
    - Post-selection qubits: 0
    """
    if gate_types not in ("elementary", "single", "none"):
        gate_types = "elementary"

    if gate_types == "elementary":
        if dagger:
            T_dagger_gate(qc, [x[1]], gate_counts)
            T_dagger_gate(qc, [x[0]], gate_counts)
        else:
            qc.t(x[1])
            qc.t(x[0])
            gate_counts[1] += 2

        qc.cx(x[1], x[0])
        gate_counts[0] += 1

        if dagger:
            qc.t(x[0])
            gate_counts[1] += 1
        else:
            T_dagger_gate(qc, [x[0]], gate_counts)

        qc.cx(x[1], x[0])
        gate_counts[0] += 1

    if gate_types == "single":
        matrix = np.diag([1, 1, 1, -1j if dagger else 1j])
        qc.append(UnitaryGate(matrix, label="CS†" if dagger else "CS"), x)

    if gate_types != "elementary":
        gate_counts[0] += 4 if dagger else 3
        gate_counts[1] += 3

# |c>|t> -> |c>V^c|t> (or |c>(V†)^c|t>)
def CV_gate(qc, x, gate_counts, dagger=False, gate_types="elementary"):
    """Apply a controlled square root of X with ``x[1]`` controlling ``x[0]``.

    Resource Requirements:
    - Gates: 2 H gates plus one controlled-S decomposition
    - Ancillas: 0
    - Post-selection qubits: 0
    """
    if gate_types not in ("elementary", "single", "none"):
        gate_types = "elementary"

    if gate_types == "elementary":
        qc.h(x[0])
        gate_counts[0] += 1
        CS_gate(qc, x, gate_counts, dagger=dagger)
        qc.h(x[0])
        gate_counts[0] += 1

    if gate_types == "single":
        v = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
        if dagger:
            v = v.conj().T
        qc.append(UnitaryGate(v, label="V†" if dagger else "V").control(1), [x[1], x[0]])

    if gate_types != "elementary":
        gate_counts[0] += 6 if dagger else 5
        gate_counts[1] += 3

# |a>|b>|t> -> |a>|b>|t XOR (a AND b)>
def Toffoli_gate(qc, x, gate_counts, gate_types="elementary"):
    """Apply a Toffoli with ``x[1:3]`` controlling target ``x[0]``.

    Resource Requirements:
    - Gates: 3 controlled-V and 2 CNOT gates, or 1 built-in CCX
    - Ancillas: 0
    - Post-selection qubits: 0
    """
    if gate_types not in ("elementary", "single", "none"):
        gate_types = "elementary"

    if gate_types == "elementary":
        CV_gate(qc, [x[0], x[2]], gate_counts)
        qc.cx(x[1], x[2])
        gate_counts[0] += 1
        CV_gate(qc, [x[0], x[1]], gate_counts)
        CV_gate(qc, [x[0], x[2]], gate_counts, dagger=True)
        qc.cx(x[1], x[2])
        gate_counts[0] += 1

    if gate_types == "single":
        qc.ccx(x[1], x[2], x[0])

    if gate_types != "elementary":
        gate_counts[0] += 18
        gate_counts[1] += 9

# |a>|b>|t> -> |a>|b>|t XOR (a AND b)>
def T_count_optimized_Toffoli_gate(qc, control_1, control_2, target, gate_counts, gate_types="elementary"):
    """Apply the seven-T Clifford+T realization of a Toffoli gate.

    Resource Requirements:
    - Gates: 2 H, 6 CNOT, and 7 T/T-dagger gates, or 1 built-in CCX
    - Ancillas: 0
    - Post-selection qubits: 0
    """
    if gate_types not in ("elementary", "single", "none"):
        gate_types = "elementary"

    if gate_types == "single":
        qc.ccx(control_1, control_2, target)

    if gate_types == "elementary":
        qc.h(target)
        qc.cx(control_2, target)
        qc.tdg(target)
        qc.cx(control_1, target)
        qc.t(target)
        qc.cx(control_2, target)
        qc.tdg(target)
        qc.cx(control_1, target)
        qc.t(control_2)
        qc.t(target)
        qc.h(target)
        qc.cx(control_1, control_2)
        qc.t(control_1)
        qc.tdg(control_2)
        qc.cx(control_1, control_2)
    gate_counts[0] += 8
    gate_counts[1] += 7

# |a>|b>|c> -> |a>|a XOR b>|c XOR (a AND b)>
def Peres_gate(qc, x, gate_counts, gate_types="elementary"):
    """Apply a Peres gate with ``x[2]`` as a, ``x[1]`` as b, and ``x[0]`` as c.

    Resource Requirements:
    - Gates: 3 controlled-V and 1 CNOT gate, or 1 three-qubit unitary
    - Ancillas: 0
    - Post-selection qubits: 0
    """
    if gate_types not in ("elementary", "single", "none"):
        gate_types = "elementary"

    if gate_types == "elementary":
        CV_gate(qc, [x[0], x[2]], gate_counts, dagger=True)
        CV_gate(qc, [x[0], x[1]], gate_counts, dagger=True)
        qc.cx(x[2], x[1])
        gate_counts[0] += 1
        CV_gate(qc, [x[0], x[1]], gate_counts)

    if gate_types == "single":
        P = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0]
        ], dtype=complex)

        peres_gate = UnitaryGate(P, label='Peres')

        qc.append(peres_gate, [x[0],x[1],x[2]])

    if gate_types != "elementary":
        gate_counts[0] += 18
        gate_counts[1] += 9
