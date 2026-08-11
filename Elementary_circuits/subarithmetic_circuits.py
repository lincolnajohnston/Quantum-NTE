from qiskit.circuit.library import UnitaryGate
import numpy as np

def T_dagger_gate(qc,x):
    qc.sdg(x[0])
    qc.t(x[0])

# applies a 1 qubit V gate to the x register
def V_gate(qc, x, dagger=False):
    qc.h(x[0])

    if dagger:
        qc.sdg(x[0])
    else:
        qc.s(x[0])

    qc.h(x[0])

# applies the 2 qubit controlled S gate
def CS_gate(qc, x, dagger=False):
    if dagger:
        T_dagger_gate(qc, [x[1]]) # inverse of the T gate
        T_dagger_gate(qc, [x[0]]) # inverse of the T gate
    else:
        qc.t(x[1])
        qc.t(x[0])

    qc.cx(x[1],x[0]) # controlled not gate

    if dagger:
        qc.t(x[0])
    else:
        T_dagger_gate(qc, [x[0]]) # inverse of the T gate

    qc.cx(x[1],x[0]) # controlled not gate

# applies a controlled V gate to the 2 qubit x register, with the control being on the 2nd qubit (more significant)
def CV_gate(qc, x, dagger=False):
    qc.h(x[0])

    CS_gate(qc, x, dagger=dagger)

    qc.h(x[0])

# applies a Toffoli gate on the 3 qubit x register, with the control being on the 2nd and 3rd (more significant) qubits 
def Toffoli_gate(qc, x, useElementaryGates=True):
    if useElementaryGates:
        CV_gate(qc, [x[0], x[2]])
        qc.cx(x[1], x[2])
        CV_gate(qc, [x[0], x[1]])
        CV_gate(qc, [x[0], x[2]], dagger=True)
        qc.cx(x[1], x[2])
    else:
        qc.ccx(x[1], x[2], x[0]) # using the built-in Toffoli gate

def T_count_optimized_Toffoli_gate(qc, control_1, control_2, target, useElementaryGates=True):
    """Apply the seven-T Clifford+T realization of a Toffoli gate from https://arxiv.org/abs/1706.05113 """
    if not useElementaryGates:
        qc.ccx(control_1, control_2, target)
        return

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

# applies a Peres gate on the 3 qubit x register, with the 3rd qubit being the "A" input, the 2nd qubit being the "B" input, and the 1st qubit being the "C" input 
def Peres_gate(qc, x, useElementaryGates=True):
    if useElementaryGates:
        CV_gate(qc, [x[0], x[2]], dagger=True)
        CV_gate(qc, [x[0], x[1]], dagger=True)
        qc.cx(x[2], x[1])
        CV_gate(qc, [x[0], x[1]])
    else:
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
