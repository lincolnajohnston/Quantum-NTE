from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.synthesis import SolovayKitaevDecomposition, qs_decomposition

import numpy as np
from Elementary_circuits import subarithmetic_circuits
from Elementary_circuits import basic_arithmetic_circuits
from Elementary_circuits import midsize_basic_circuits


def M_2_star(qc, L_f, x, ps, ancillas):
    n = L_f + 2
    if len(x) != n or n < 2:
        raise ValueError("x must contain exactly L_f + 2 qubits")
    if len(ps) < 1:
        raise ValueError("ps must contain at least one post-selection qubit")

    # Discard the most-significant input bit into the post-selection register.
    # Post-selecting ps[0] in |0> after the Hadamard makes the resulting block
    # independent of that bit, as required by multiplication by two modulo 2**n.
    qc.swap(x[-1], ps[0])
    qc.h(ps[0])

    # Use the now-zero most-significant qubit as the branch bit.  On branch 0
    # retain r, and on branch 1 replace r by r - 1, where r is the low n-1
    # bits of the input.  Rotating the branch bit to the least-significant
    # position then gives 2r and 2r-1 respectively.
    branch = x[-1]
    qc.h(branch)
    for i in range(n - 2, -1, -1):
        lower_bits = list(x[:i])
        for qubit in lower_bits:
            qc.x(qubit)
        midsize_basic_circuits.n_controlled_x(
            qc,
            [branch] + lower_bits,
            x[i],
            ancillas[:max(0, i - 1)],
        )
        for qubit in lower_bits:
            qc.x(qubit)

    for i in range(n - 1, 0, -1):
        qc.swap(x[i], x[i - 1])

    return qc

def M_2_prime(qc, L_f, x, ps, ancillas):
    # TODO: implement M_2'
    pass
