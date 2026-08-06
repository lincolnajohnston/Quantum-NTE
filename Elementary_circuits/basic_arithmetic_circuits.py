from qiskit import QuantumCircuit

import numpy as np
import Elementary_circuits.subarithmetic_circuits as subarithmetic_circuits

def Ones_complement(qc, a):
    for i in range(len(a)):
        qc.x(a[i])

def Twos_complement(qc, a, b):
    Ones_complement(qc, a)
    qc.x(b[0])  # prepare the constant 1 in the work register
    Addition_gate(qc, b, a, modular=True, useElementaryGates=False)
    qc.x(b[0])  # Addition_gate preserves b, so return it to |0...0>

# From section 4 of https://arxiv.org/pdf/1712.02630
def Addition_gate(qc, a, b, modular=False, useElementaryGates=True):
    n = len(a)
    if n + int(not modular) != len(b):
        raise Exception("a and b registers are not the same length for modular addition")
    
    # Step 1
    for i in range(1, n):
        qc.cx(a[i], b[i])

    qc.barrier(label="Step 2")

    # Step 2
    if not modular:
        qc.cx(a[n-1], b[n])
    for i in range(n-2,0,-1):
        qc.cx(a[i], a[i+1])

    qc.barrier(label="Step 3")

    # Step 3
    for i in range(n-1):
        qc.barrier(label="Toffoli Gate")
        subarithmetic_circuits.Toffoli_gate(qc, [a[i+1], a[i], b[i]], useElementaryGates = useElementaryGates) # using Clifford + T gates

    qc.barrier(label="Step 4")

    # Step 4
    if not modular:
        subarithmetic_circuits.Peres_gate(qc, [b[n], b[n-1], a[n-1]], useElementaryGates = useElementaryGates)
    else:
        qc.cx(a[n-1], b[n-1])
    for i in range(n-2, -1,-1):
        qc.barrier(label="Peres Gate")
        subarithmetic_circuits.Peres_gate(qc, [a[i+1], b[i], a[i]], useElementaryGates = useElementaryGates)

    qc.barrier(label="Step 5")

    # Step 5
    for i in range(1, n-1):
        qc.cx(a[i], a[i+1])

    qc.barrier(label="Step 6")  

    # Step 6
    for i in range(1, n):
        qc.cx(a[i], b[i])


def Conditional_addition_gate(qc, ctrl, a, b, carry, work, useElementaryGates=True):
    """Conditionally compute b <- b + a and write the carry-out to carry.

    This is the seven-step Ctrl-Add circuit from Section III of
    arXiv:1706.05113. The work qubit must start in |0> and is restored.
    """
    n = len(a)
    if n < 2 or len(b) != n:
        raise ValueError("a and b must have the same length of at least two")

    def apply_toffoli(control_1, control_2, target):
        subarithmetic_circuits.Toffoli_gate(qc, [target, control_1, control_2], useElementaryGates=useElementaryGates)

    # Step 1
    for i in range(1, n):
        qc.cx(a[i], b[i])

    # Step 2
    apply_toffoli(ctrl, a[n - 1], carry)
    for i in range(n - 2, 0, -1):
        qc.cx(a[i], a[i + 1])

    # Step 3
    for i in range(n - 1):
        apply_toffoli(b[i], a[i], a[i + 1])

    # Step 4
    apply_toffoli(b[n - 1], a[n - 1], work)
    apply_toffoli(ctrl, work, carry)
    apply_toffoli(b[n - 1], a[n - 1], work)
    apply_toffoli(ctrl, a[n - 1], b[n - 1])

    # Step 5
    for i in range(n - 2, -1, -1):
        apply_toffoli(b[i], a[i], a[i + 1])
        apply_toffoli(ctrl, a[i], b[i])

    # Step 6
    for i in range(1, n - 1):
        qc.cx(a[i], a[i + 1])

    # Step 7
    for i in range(1, n):
        qc.cx(a[i], b[i])


def Integer_multiplication_gate(qc, a, b, p, useElementaryGates=True):
    """Compute the unsigned product of equal-width a and b into zeroed p."""
    n = len(a)
    if n != len(b) or len(p) != 2 * n + 1:
        raise ValueError("a and b must have length n and p must have length 2*n + 1")

    # Paper Step 1: the b[0] partial product needs only a Toffoli array.
    for i in range(n):
        subarithmetic_circuits.Toffoli_gate(qc, [p[i], b[0], a[i]], useElementaryGates=useElementaryGates)

    # Paper Steps 2 and 3: conditionally add each shifted partial product.
    for j in range(1, n):
        Conditional_addition_gate(
            qc,
            b[j],
            a,
            p[j:j + n],
            p[j + n],
            p[j + n + 1],
            useElementaryGates=useElementaryGates,
        )