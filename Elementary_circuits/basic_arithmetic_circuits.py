from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.synthesis import SolovayKitaevDecomposition, qs_decomposition

import numpy as np
from Elementary_circuits import subarithmetic_circuits

_SOLOVAY_KITAEV = None

# TODO: check that all of the gate counts for each of the implementations are correct and consistent with my writeup

# |a>^n -> |(2**n - 1) - a>^n
def Ones_complement_inplace(qc, a):
    """Replace an n-qubit bit string with its one's complement in place.

    An X gate is applied independently to every qubit, so the basis-state
    mapping is ``|a> -> |(2**n - 1) - a>``.  The register is little-endian,
    although the operation itself is independent of qubit order.


    Gate Counts:
    - n X gates
    ----------------
    -n Clifford gates
    """
    # X maps each stored bit a[i] to a[i] XOR 1.
    for i in range(len(a)):
        qc.x(a[i])


# |a>^n|0>^n -> |a>^n|(2**n - 1) - a>^n
def Ones_complement_outofplace(qc, a, result):
    """Write the one's complement of ``a`` to a zeroed result register.

    The input register is preserved, and ``result`` must have the same width
    as ``a`` and start in |0>.

    Gate Counts:
    - n X gates
    - n CNOT gates
    ----------------
    - 2n Clifford gates
    - 0 T gates
    """
    if len(a) != len(result):
        raise ValueError("a and result must have the same length")

    for i in range(len(a)):
        qc.cx(a[i], result[i])
        qc.x(result[i])


# From section 4 of https://arxiv.org/pdf/1712.02630
# modular: |a>^n|b>^n -> |a>^n|(a + b) mod 2**n>^n
# full: |a>^n|b>^(n+1) -> |a>^n|a + b>^(n+1)
def Addition_gate_inplace(qc, a, b, modular=False, useElementaryGates=True):
    """Add little-endian register ``a`` into ``b`` while preserving ``a``.

    In modular mode, both registers contain n qubits and the mapping is
    ``|a>|b> -> |a>|(a + b) mod 2**n>``.  In full-addition mode, ``b`` has
    n + 1 qubits; when ``b[n]`` starts at zero it receives the carry-out and
    the complete n + 1 bit sum is stored in ``b``.

    The six stages below use CNOTs to encode and later uncompute carry
    dependencies, Toffoli gates to generate carries, and Peres gates to write
    sum bits while propagating carries.  ``useElementaryGates`` selects the
    Clifford+T implementations of the Toffoli and Peres gates when true.

    Gate Counts (Modular):
    - Modular: (12n - 12) H gates
    - Modular: (9n - 9) S-dagger gates
    - Modular: (18n - 18) T gates
    - Modular: (19n - 20) CNOT gates
    ----------------
    - Modular: (40n - 41) Clifford gates
    - Modular: (18n - 18) T gates

    Gate Counts (Full):
    - Full: (12n - 6) H gates
    - Full: (9n - 4) S-dagger gates
    - Full: (18n - 9) T gates
    - Full: (19n - 13) CNOT gates
    ----------------
    - Full: (40n - 23) Clifford gates
    - Full: (18n - 9) T gates
    """
    n = len(a)
    if n + int(not modular) != len(b):
        raise Exception("a and b registers are not the same length for modular addition")
    
    # Step 1: XOR the upper a bits into b to begin encoding propagate bits.
    for i in range(1, n):
        qc.cx(a[i], b[i])

    qc.barrier(label="Step 2")

    # Step 2: expose the full-mode carry target and propagate information
    # toward the most-significant end of a.
    if not modular:
        qc.cx(a[n-1], b[n])
    for i in range(n-2,0,-1):
        qc.cx(a[i], a[i+1])

    qc.barrier(label="Step 3")

    # Step 3: Toffoli gates generate the carry dependencies in a[i + 1].
    for i in range(n-1):
        qc.barrier(label="Toffoli Gate")
        subarithmetic_circuits.Toffoli_gate(qc, [a[i+1], a[i], b[i]], useElementaryGates = useElementaryGates) # using Clifford + T gates

    qc.barrier(label="Step 4")

    # Step 4: Peres gates propagate carries back toward the low end while
    # writing the corresponding sum bits into b.
    if not modular:
        subarithmetic_circuits.Peres_gate(qc, [b[n], b[n-1], a[n-1]], useElementaryGates = useElementaryGates)
    else:
        qc.cx(a[n-1], b[n-1])
    for i in range(n-2, -1,-1):
        qc.barrier(label="Peres Gate")
        subarithmetic_circuits.Peres_gate(qc, [a[i+1], b[i], a[i]], useElementaryGates = useElementaryGates)

    qc.barrier(label="Step 5")

    # Step 5: undo the temporary carry-propagation correlations within a.
    for i in range(1, n-1):
        qc.cx(a[i], a[i+1])

    qc.barrier(label="Step 6")  

    # Step 6: finish the sum XORs and restore a to its input value.
    for i in range(1, n):
        qc.cx(a[i], b[i])


# modular: |a>^n|b>^n|0>^n -> |a>^n|b>^n|(a + b) mod 2**n>^n
# full: |a>^n|b>^(n+1)|0>^(n+1) -> |a>^n|b>^(n+1)|a + b>^(n+1)
def Addition_gate_outofplace(qc, a, b, result, modular=False, useElementaryGates=True):
    """Add ``a`` and ``b`` into a zeroed result register, preserving both inputs.

    This is the out-of-place counterpart of :func:`Addition_gate`.  ``result``
    must start in |0> and have the same width as ``b``.  In modular mode all
    three registers have n qubits and the mapping is
    ``|a>|b>|0> -> |a>|b>|(a + b) mod 2**n>``.  In full-addition mode ``b``
    and ``result`` have n + 1 qubits; with ``b[n]`` initially zero, ``result``
    receives the complete n + 1 bit sum.

    CNOTs first copy ``b`` into ``result``.  The existing in-place adder then
    adds ``a`` to that copy, so neither source register is changed.

    Gate Counts (Modular):
    - Modular: (12n - 12) H gates
    - Modular: (9n - 9) S-dagger gates
    - Modular: (18n - 18) T gates
    - Modular: (20n - 20) CNOT gates
    ----------------
    - Modular: (41n - 41) Clifford gates
    - Modular: (18n - 18) T gates

    Gate Counts (Full):
    - Full: (12n - 6) H gates
    - Full: (9n - 4) S-dagger gates
    - Full: (18n - 9) T gates
    - Full: (20n - 12) CNOT gates
    ----------------
    - Full: (41n - 22) Clifford gates
    - Full: (18n - 9) T gates
    """
    n = len(a)
    expected_width = n + int(not modular)
    if n <= 1:
        raise ValueError("Addition_gate_outofplace requires at least two operand bits")
    if len(b) != expected_width or len(result) != expected_width:
        raise ValueError(
            "b and result must have length n for modular addition or n + 1 "
            "for full addition"
        )

    # Prepare result = b without modifying b.
    for i in range(len(b)):
        qc.cx(b[i], result[i])

    # result <- result + a; Addition_gate preserves its first operand a.
    Addition_gate_inplace(
        qc,
        a,
        result,
        modular=modular,
        useElementaryGates=useElementaryGates,
    )


# |a>^n|b>^n -> |a>^n|(b - a) mod 2**n>^n
def Integer_subtraction_gate_inplace(qc, a, b, useElementaryGates=True):
    """Subtract ``a`` from ``b`` in place modulo ``2**n``.

    Both registers are little-endian and have the same nonzero width.  The
    input register ``a`` is preserved while underflow in ``b`` wraps around.

    Gate Counts:
    - (12n - 12) H gates
    - (9n - 9) S gates
    - (18n - 18) T-dagger gates
    - (19n - 20) CNOT gates
    ----------------
    - (40n - 41) Clifford gates
    - (18n - 18) T gates
    """
    n = len(a)
    if n == 0 or len(b) != n:
        raise ValueError("a and b must have the same nonzero length")

    addition = QuantumCircuit(2 * n, name="modular_addition")
    Addition_gate_inplace(
        addition,
        addition.qubits[:n],
        addition.qubits[n:],
        modular=True,
        useElementaryGates=useElementaryGates,
    )
    qc.compose(
        addition.inverse(),
        qubits=list(a) + list(b),
        inplace=True,
    )


# |a>^n|b>^n|0>^n -> |a>^n|b>^n|(b - a) mod 2**n>^n
def Integer_subtraction_gate_outofplace(qc, a, b, p, useElementaryGates=True):
    """Compute ``(b - a) mod 2**n`` in a zeroed unsigned register p.

    The registers use little-endian standard binary representation.  Both
    input registers are preserved, and underflow wraps modulo ``2**n``::

        |a>|b>|0> -> |a>|b>|(b - a) mod 2**n>

    ``p`` must start in |0> so that copying ``b`` into it prepares the minuend.

    Gate Counts:
    - (12n - 12) H gates
    - (9n - 9) S gates
    - (18n - 18) T-dagger gates
    - (20n - 20) CNOT gates
    ----------------
    - (41n - 41) Clifford gates
    - (18n - 18) T gates
    """
    n = len(a)
    if n == 0 or len(b) != n or len(p) != n:
        raise ValueError("a, b, and p must have the same nonzero length")

    # Prepare p = b without modifying either input register.
    for i in range(n):
        qc.cx(b[i], p[i])

    Integer_subtraction_gate_inplace(
        qc,
        a,
        p,
        useElementaryGates=useElementaryGates,
    )


# modular: |c>^1|a>^n|b>^n|0>^1|0>^1 -> |c>^1|a>^n|(b + c*a) mod 2**n>^n|0>^1|0>^1
# full: |c>^1|a>^n|b>^n|0>^1|0>^1 -> |c>^1|a>^n|(b + c*a) mod 2**n>^n|floor((b + c*a)/2**n)>^1|0>^1
def Conditional_addition_gate_inplace(
    qc, ctrl, a, b, carry, work, modular=False, useElementaryGates=True
):
    """Conditionally add ``a`` into ``b``, optionally retaining carry-out.

    This is the seven-step Ctrl-Add circuit from Section III of
    arXiv:1706.05113.  With zeroed ``carry`` and ``work`` qubits, full mode
    stores the low n sum bits in ``b`` and the overflow bit in ``carry``.
    Modular mode stores only ``(b + c*a) mod 2**n`` in ``b`` and leaves both
    ancillary qubits zero.  The control and ``a`` are preserved.

    Registers are little-endian and must have the same width of at least two qubits.

    Gate Counts (Modular):
    - Modular: (18n - 12) H gates
    - Modular: (12n - 8) S-dagger gates
    - Modular: (27n - 18) T gates
    - Modular: (28n - 22) CNOT gates
    ----------------
    - Modular: (58n - 42) Clifford gates
    - Modular: (27n - 18) T gates

    Gate Counts (Full):
    - Full: (18n + 6) H gates
    - Full: (12n + 4) S-dagger gates
    - Full: (27n + 9) T gates
    - Full: (28n + 2) CNOT gates
    ----------------
    - Full: (58n + 12) Clifford gates
    - Full: (27n + 9) T gates
    """
    n = len(a)
    if n < 2 or len(b) != n:
        raise ValueError("a and b must have the same length of at least two")

    # |x>^1|y>^1|t>^1 -> |x>^1|y>^1|t XOR (x AND y)>^1
    def apply_toffoli(control_1, control_2, target):
        """Apply a Toffoli gate using its elementary-gate implementation.

        Gate Counts:
        - 6 H gates
        - 4 S-dagger gates
        - 9 T gates
        - 8 CNOT gates
        ----------------
        - 18 Clifford gates
        - 9 T gates
        """
        subarithmetic_circuits.Toffoli_gate(qc, [target, control_1, control_2], useElementaryGates=useElementaryGates)

    # Step 1: encode the initial propagate information in b[1:].
    for i in range(1, n):
        qc.cx(a[i], b[i])

    # Step 2: condition the top carry in full mode and ripple dependencies up a.
    if not modular:
        apply_toffoli(ctrl, a[n - 1], carry)
    for i in range(n - 2, 0, -1):
        qc.cx(a[i], a[i + 1])

    # Step 3: generate the internal carries in a[1:].
    for i in range(n - 1):
        apply_toffoli(b[i], a[i], a[i + 1])

    # Step 4: retain overflow only for full addition, then write the
    # controlled most-significant sum bit in both modes.
    if not modular:
        apply_toffoli(b[n - 1], a[n - 1], work)
        apply_toffoli(ctrl, work, carry)
        apply_toffoli(b[n - 1], a[n - 1], work)
    apply_toffoli(ctrl, a[n - 1], b[n - 1])

    # Step 5: uncompute internal carries and write lower controlled sum bits.
    for i in range(n - 2, -1, -1):
        apply_toffoli(b[i], a[i], a[i + 1])
        apply_toffoli(ctrl, a[i], b[i])

    # Step 6: restore the temporary ripple encoding within a.
    for i in range(1, n - 1):
        qc.cx(a[i], a[i + 1])

    # Step 7: finish restoring a and the propagate encoding in b.
    for i in range(1, n):
        qc.cx(a[i], b[i])


# modular: |c>^1|a>^n|b>^n|0>^n|0>^1 -> |c>^1|a>^n|b>^n|(b + c*a) mod 2**n>^n|0>^1
# full: |c>^1|a>^n|b>^n|0>^(n+1)|0>^1 -> |c>^1|a>^n|b>^n|b + c*a>^(n+1)|0>^1
def Conditional_addition_gate_outofplace(
    qc, ctrl, a, b, result, work, modular=False, useElementaryGates=True
):
    """Conditionally add ``a`` and ``b`` into ``result``, preserving inputs.

    For equal-width n-qubit, little-endian inputs, ``result`` must contain n
    zeroed qubits in modular mode or n + 1 zeroed qubits in full mode.  The
    low n result bits contain the sum; full mode also stores the carry-out in
    ``result[n]``.  ``ctrl``, ``a``, and ``b`` are preserved, and ``work`` is
    restored to zero.

    CNOTs copy ``b`` into the low result bits before the in-place conditional
    adder targets that copy.

    Gate Counts (Modular):
    - Modular: (18n - 12) H gates
    - Modular: (12n - 8) S-dagger gates
    - Modular: (27n - 18) T gates
    - Modular: (29n - 22) CNOT gates
    ----------------
    - Modular: (59n - 42) Clifford gates
    - Modular: (27n - 18) T gates

    Gate Counts (Full):
    - Full: (18n + 6) H gates
    - Full: (12n + 4) S-dagger gates
    - Full: (27n + 9) T gates
    - Full: (29n + 2) CNOT gates
    ----------------
    - Full: (59n + 12) Clifford gates
    - Full: (27n + 9) T gates
    """
    n = len(a)
    if n < 2 or len(b) != n:
        raise ValueError("a and b must have the same length of at least two")
    expected_result_width = n + int(not modular)
    if len(result) != expected_result_width:
        raise ValueError(
            "result must have length n for modular addition or n + 1 "
            "for full addition"
        )

    # Prepare the low result bits as a copy of b; the full-mode carry remains zero.
    for i in range(n):
        qc.cx(b[i], result[i])

    if modular:
        Conditional_addition_gate_inplace(
            qc,
            ctrl,
            a,
            result,
            work,
            work,
            modular=True,
            useElementaryGates=useElementaryGates,
        )
    else:
        Conditional_addition_gate_inplace(
            qc,
            ctrl,
            a,
            result[:n],
            result[n],
            work,
            modular=False,
            useElementaryGates=useElementaryGates,
        )


# |a>^n|b>^n|0>^(2*n+1) -> |a>^n|b>^n|a*b>^(2*n)|0>^1
def Integer_multiplication_gate(qc, a, b, p, useElementaryGates=True):
    """Multiply two equal-width unsigned integers into a zeroed register.

    For n-qubit, little-endian inputs this implements
    ``|a>|b>|0>^(2*n + 1) -> |a>|b>|a*b>|0>``.  The 2*n product bits occupy
    ``p[0:2*n]``; ``p[2*n]`` is temporary carry workspace and is restored to
    zero.  Both inputs are preserved.

    The circuit is a shift-and-add multiplier.  Toffoli gates form the partial
    product selected by ``b[0]``.  Each higher bit ``b[j]`` then controls an
    addition of ``a`` into the shifted slice ``p[j:j+n]``; the next two p
    qubits supply that addition's carry and clean work qubit.

    Gate Counts:
    - (18n**2 - 6n - 6) H gates
    - (12n**2 - 4n - 4) S-dagger gates
    - (27n**2 - 9n - 9) T gates
    - (28n**2 - 18n - 2) CNOT gates
    ----------------
    - (58n**2 - 28n - 12) Clifford gates
    - (27n**2 - 9n - 9) T gates
    """
    n = len(a)
    if n != len(b) or len(p) != 2 * n + 1:
        raise ValueError("a and b must have length n and p must have length 2*n + 1")

    # Paper Step 1: p[i] ^= b[0] AND a[i], forming a*b[0].
    for i in range(n):
        subarithmetic_circuits.Toffoli_gate(qc, [p[i], b[0], a[i]], useElementaryGates=useElementaryGates)

    # Paper Steps 2 and 3: when b[j] is one, add a*2**j into p.
    for j in range(1, n):
        Conditional_addition_gate_inplace(
            qc,
            b[j],
            a,
            p[j:j + n],
            p[j + n],
            p[j + n + 1],
            modular=False,
            useElementaryGates=useElementaryGates,
        )
