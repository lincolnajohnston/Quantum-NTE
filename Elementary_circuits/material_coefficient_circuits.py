"""Comparator-based block encodings of piecewise-constant coefficients."""

from numbers import Integral

import numpy as np
from qiskit import QuantumCircuit

from Elementary_circuits import midsize_basic_circuits


def _constant_comparator(qc, x, threshold, flag, bound, useElementaryGates=True):
    """Toggle ``flag`` for x >= bound in qc, restoring a clean threshold.

    Registers are little-endian. Elementary mode applies Clifford+T gates
    directly; otherwise append a single comparator gate. Returns qc.
    """
    x, threshold = list(x), list(threshold)
    n = len(x)
    if not n or len(threshold) != n:
        raise ValueError("x and threshold must have the same nonzero length")
    if not useElementaryGates:
        comparison = QuantumCircuit(2 * n + 1, name=f"geq_{bound}")
        _constant_comparator(
            comparison, comparison.qubits[:n], comparison.qubits[n:2*n],
            comparison.qubits[-1], bound,
        )
        qc.append(comparison.to_gate(), x + threshold + [flag])
        return qc

    start = len(qc.data)
    if bound == 0:
        qc.x(flag)
    elif bound < 2**n:
        for bit, qubit in enumerate(threshold):
            if (bound >> bit) & 1:
                qc.x(qubit)
        midsize_basic_circuits.integer_comparator_gate(
            qc, x, threshold, flag, useElementaryGates=True
        )
        for bit, qubit in enumerate(threshold):
            if (bound >> bit) & 1:
                qc.x(qubit)
    # Remove only barriers added by these arithmetic helpers.
    for index in range(len(qc.data) - 1, start - 1, -1):
        if qc.data[index].operation.name == "barrier":
            del qc.data[index]
    return qc


def material_coefficient_block_encoding(qc, x, ps, ancillas, blocks, alpha=None, useElementaryGates=True):
    """Append a block encoding of a piecewise-constant diagonal matrix.

    ``blocks`` contains nonoverlapping ``(start, stop, value)`` triples, with
    zero-based, half-open intervals: D[j, j] = value for start <= j < stop.
    Unspecified entries (including padding) are zero. Values may be real or
    complex. ``x`` is a nonempty little-endian address register (x[0] is the
    least-significant bit); D has dimension 2**len(x).

    ``ps`` contains exactly one post-selection qubit. ``ancillas`` contains
    at least len(x)+1 clean work qubits: one interval flag and a len(x)-qubit
    threshold register for midsize_basic_circuits.integer_comparator_gate.
    All these qubits must be distinct and belong to qc. Initialize ps and
    work qubits to |0>. Work qubits are restored to |0> on every branch.
    Projecting ps onto |0> gives D / alpha on x, i.e.
    <0|_ps <0|_work U |0>_ps |0>_work = D / alpha.

    ``alpha`` must be finite, positive, and >= max(abs(value)). By default
    it is that maximum, or 1 for the zero matrix. Returns qc, modified in
    place, following the other Elementary_circuits functions. No measurement
    is appended. Arbitrary rotations are used without Clifford+T synthesis.

    Resource Requirements:
      Gates:
        - At most four integer_comparator_gate applications per nonzero block
        - One controlled Ry and at most one phase gate per nonzero block
        - O(len(blocks) * len(x)) threshold-preparation X gates; comparator
          cost follows midsize_basic_circuits.integer_comparator_gate
      Reusable Ancillas:
        - len(x)+1 clean qubits
      Post-selection qubits:
        - 1

    Example (one-based elements 1..7 = 6.4, 8..21 = 7.7; zero padding):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(12)
        material_coefficient_block_encoding(
            qc, range(5), [5], range(6, 12),
            [(0, 7, 6.4), (7, 21, 7.7)], alpha=7.7,
        )
    """
    # making sure x, ps, and ancillas are lists of qubits and have the correct lengths and are in the right formats
    x, ps, ancillas = list(x), list(ps), list(ancillas)
    n = len(x)
    if n < 1 or len(ps) != 1 or len(ancillas) < n + 1:
        raise ValueError("Require nonempty x, exactly one ps qubit, and at least len(x)+1 ancillas")
    # Resolve integer indices as well as Qubit objects before checking aliases.
    registers = []
    for register in (x, ps, ancillas):
        resolved = []
        for qubit in register:
            if isinstance(qubit, Integral):
                if not 0 <= qubit < qc.num_qubits:
                    raise ValueError("Qubit index is outside qc")
                qubit = qc.qubits[int(qubit)]
            qc.find_bit(qubit)
            resolved.append(qubit)
        registers.append(resolved)
    x, ps, ancillas = registers
    all_qubits = x + ps + ancillas
    if len(set(all_qubits)) != len(all_qubits):
        raise ValueError("x, ps, and ancillas must contain distinct qubits")

    # Validate and process the blocks, ensuring they are non-overlapping and within the correct range.
    intervals = []
    for start, stop, value in blocks:
        if (not isinstance(start, Integral) or not isinstance(stop, Integral)
                or not 0 <= start < stop <= 2**n):
            raise ValueError("Block bounds must be integers with 0 <= start < stop <= 2**len(x)")
        value = complex(value)
        if not np.isfinite(value) or not np.isfinite(abs(value)):
            raise ValueError("Block values must be finite")
        intervals.append((int(start), int(stop), value))
    intervals.sort(key=lambda block: block[0])
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("Blocks must not overlap")
    max_value = max((abs(value) for _, _, value in intervals), default=0.0)
    alpha = float(alpha) if alpha is not None else (max_value or 1.0)
    if not np.isfinite(alpha) or alpha <= 0 or alpha < max_value:
        raise ValueError("alpha must be finite, positive, and at least max(abs(value))")

    # perform the actual coefficient encoding using the block data
    # flag qubit is 1 when the input state is within the interval defined by start and stop
    flag, threshold = ancillas[0], ancillas[1:n + 1]
    # Ry(pi)|0> = |1>: uncovered addresses have zero success amplitude.
    qc.ry(np.pi, ps[0])
    for start, stop, value in intervals:
        if value == 0:
            continue
        # [x >= start] XOR [x >= stop] is the interval membership flag.
        for bound in (start, stop):
            _constant_comparator(qc, x, threshold, flag, bound, useElementaryGates=useElementaryGates)
        theta = 2 * np.arccos(np.clip(abs(value) / alpha, 0.0, 1.0))
        qc.cry(theta - np.pi, flag, ps[0])
        # The flag phase supplies the sign or complex phase of the coefficient.
        if np.angle(value) != 0:
            qc.p(float(np.angle(value)), flag)
        for bound in (stop, start):
            _constant_comparator(qc, x, threshold, flag, bound, useElementaryGates=useElementaryGates)
    return qc
