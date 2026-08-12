from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.synthesis import SolovayKitaevDecomposition, qs_decomposition

import numpy as np
from Elementary_circuits import subarithmetic_circuits
from Elementary_circuits import basic_arithmetic_circuits

# TODO: add in the option to UseElementaryGates for all functions
# TODO: add in the n-controlled x gate implementation and tests
# TODO: add in the controlled arbitrary gate using constant overhead from https://arxiv.org/abs/1206.0758
# TODO: add in the comparator operator gates
# TODO: add in the division operator gates

# TODO: test this function
# |x>^n|y> -> |x>^n|y + (x_1 AND x_2 AND ... AND x_n)>
def n_controlled_x(qc, control_qubits, target_qubit, ancilla_qubits):
    """Apply an n-controlled X gate to the target qubit in the quantum circuit qc.
    control_qubits is a list of n control qubits, target_qubit is the target qubit, 
    and ancilla_qubits is a list of n-2 ancilla qubits.

    
    Resource Requirements:
      Gates:
        - 2n-3 Toffoli gates
        ----------------
        - 16n-24 Clifford gates
        - 14n-21 T gates
      Resuable Ancillas:
        - n-2 ancilla qubits
      Post-selection qubits:
        - 0 post-selection qubits
    """
    controls = list(control_qubits)
    ancillas = list(ancilla_qubits)
    gate_counts = [0, 0]
    num_controls = len(controls)
    required_ancillas = max(0, num_controls - 2)

    if len(ancillas) != required_ancillas:
        raise ValueError(
            f"n_controlled_x requires exactly {required_ancillas} ancilla qubits "
            f"for {num_controls} controls; received {len(ancillas)}"
        )

    # check for cases of 0, 1, or 2 controls and handle them separately
    if num_controls == 0: # simple X gate
        qc.x(target_qubit)
        return qc
    if num_controls == 1: # simple CNOT/CX gate
        qc.cx(controls[0], target_qubit)
        return qc
    if num_controls == 2: # Toffoli gate
        subarithmetic_circuits.T_count_optimized_Toffoli_gate(
            qc, controls[0], controls[1], target_qubit, gate_counts
        )
        return qc

    toffoli = subarithmetic_circuits.T_count_optimized_Toffoli_gate # uses 8 Clifford Gates, 7 T gates, and 0 ancilla qubits

    # Use the ancilla qubits to build up the AND of all control qubits, then make the target qubit flip conditional on all controls being |1>
    toffoli(qc, controls[0], controls[1], ancillas[0], gate_counts) # ancilla[0] = controls[0] AND controls[1]
    for i in range(1, num_controls - 2):
        toffoli(qc, ancillas[i - 1], controls[i + 1], ancillas[i], gate_counts) # ancilla[i] = ancilla[i-1] AND controls[i+1] = controls[0] AND ... AND controls[i+1]
    toffoli(qc, ancillas[-1], controls[-1], target_qubit, gate_counts) # target_qubit = ancilla[-1] AND controls[-1] = controls[0] AND ... AND controls[n_controls-1]

    # uncompute ancillas
    for i in range(num_controls - 3, 0, -1):
        toffoli(qc, ancillas[i - 1], controls[i + 1], ancillas[i], gate_counts)
    toffoli(qc, controls[0], controls[1], ancillas[0], gate_counts)

    return qc

# |a> -> U|a>
def arbitrary_single_qubit_gate(qc, matrix, x, epsilon=None, recursion_degree=None):
    """Approximate an arbitrary one-qubit unitary with Solovay-Kitaev synthesis.

    Resource Requirements:
      Gates:
        - O(log^3.97(1/epsilon)) Clifford gates
        - O(log^3.97(1/epsilon)) T gates
      Reusable Ancillas:
        - 0 ancilla qubits
      Post-selection qubits:
        - 0 post-selection qubits
    """

    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (2, 2):
        raise ValueError("matrix must be a 2x2 unitary")
    if epsilon is not None and recursion_degree is not None:
        raise ValueError("only one of epsilon or recursion_degree should be specified")
    if epsilon is None and recursion_degree is None:
        raise ValueError("one of epsilon or recursion_degree must be specified")
    if epsilon is not None and epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if recursion_degree is not None and recursion_degree < 0:
        raise ValueError("recursion_degree must be non-negative")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(2), atol=1e-10):
        raise ValueError("matrix must be unitary")

    target_qubit = (
        x[0]
        if isinstance(x, (list, tuple)) or hasattr(x, "__getitem__")
        else x
    )

    if recursion_degree is not None:
        approximation = SolovayKitaevDecomposition().run(matrix, recursion_degree)
        qc.compose(approximation, qubits=[target_qubit], inplace=True)
        return qc
    else:
        recursion_degree = 0
        while True:
            approximation = SolovayKitaevDecomposition().run(matrix, recursion_degree)
            approximate_matrix = Operator(approximation).data
            phase = np.angle(np.vdot(approximate_matrix, matrix))
            error = np.linalg.norm(
                matrix - np.exp(1j * phase) * approximate_matrix, ord=2
            )
            if error <= epsilon:
                qc.compose(approximation, qubits=[target_qubit], inplace=True)
                return qc
            recursion_degree += 1

# |a>^n -> U|a>^n
def arbitrary_n_qubit_gate(qc, matrix, x, epsilon=None, recursion_degree=None):
    """Approximate an arbitrary n-qubit unitary via Shannon decomposition and Solovay-Kitaev synthesis.

    Specify exactly one of ``epsilon`` or ``recursion_degree``.

    Resource Requirements:
      Gates:
        - O(4^n) CNOT and synthesized one-qubit Clifford+T gates
      Reusable Ancillas:
        - 0 ancilla qubits
      Post-selection qubits:
        - 0 post-selection qubits
    """
    matrix = np.asarray(matrix, dtype=complex)
    dimension = 2 ** len(x)

    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"matrix must have shape ({dimension}, {dimension}) for "
            f"{len(x)} qubits"
        )
    if epsilon is not None and recursion_degree is not None:
        raise ValueError("only one of epsilon or recursion_degree should be specified")
    if epsilon is None and recursion_degree is None:
        raise ValueError("one of epsilon or recursion_degree must be specified")
    if epsilon is not None and epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if recursion_degree is not None and recursion_degree < 0:
        raise ValueError("recursion_degree must be non-negative")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(dimension), atol=1e-10):
        raise ValueError("matrix must be unitary")

    decomposition = transpile(
        qs_decomposition(matrix),
        basis_gates=["u", "cx"],
        optimization_level=0,
    )
    one_qubit_count = sum(
        instruction.operation.num_qubits == 1
        for instruction in decomposition.data
    )
    per_gate_epsilon = (
        epsilon / max(one_qubit_count, 1) if epsilon is not None else None
    )

    qc.global_phase += decomposition.global_phase
    for instruction in decomposition.data:
        operation = instruction.operation
        indices = [
            decomposition.find_bit(qubit).index for qubit in instruction.qubits
        ]
        if operation.name == "cx":
            qc.cx(x[indices[0]], x[indices[1]])
        elif operation.num_qubits == 1:
            if recursion_degree is not None:
                arbitrary_single_qubit_gate(
                    qc,
                    Operator(operation).data,
                    [x[indices[0]]],
                    recursion_degree=recursion_degree,
                )
            else:
                arbitrary_single_qubit_gate(
                    qc,
                    Operator(operation).data,
                    [x[indices[0]]],
                    epsilon=per_gate_epsilon,
                )
        else:
            raise RuntimeError(
                f"unexpected gate {operation.name!r} in unitary decomposition"
            )
    return qc

def controlled_single_qubit_gate(qc, matrix, control_qubits, target_qubits, epsilon):
    pass  # Placeholder for future implementation of controlled arbitrary single-qubit gate

def controlled_arbitrary_n_qubit_gate(qc, matrix, control_qubits, target_qubits, epsilon):
    pass  # Placeholder for future implementation of controlled arbitrary n-qubit gate

def integer_division_gate(qc, dividend_qubits, divisor_qubits, quotient_qubits, remainder_qubits):
    pass  # Placeholder for future implementation of integer division gate

# |x>|0>_ps|0...0> -> |x>|[x >= M]>_ps|0...0>
def pi_projector_gate(qc, x, ps, ancillas, M):
    """Put the projector onto ``x < M`` in the post-selected zero block.

    ``x[-1]`` is the most-significant qubit.  The interval
    ``[0, M)`` is partitioned into the disjoint binary-prefix blocks obtained
    from the set bits of ``M``.  ``ps`` contains only the post-selection
    qubit; ``ancillas`` is clean workspace and is returned to ``|0>``.

    Resource Requirements:
      Gates:
        - One multi-controlled X per set bit of M, plus O(n^2) X gates
      Reusable Ancillas:
        - At most n-2 ancilla qubits
      Post-selection qubits:
        - 1 post-selection qubit
    """
    n = len(x)
    if n == 0:
        raise ValueError("x must contain at least one qubit")
    if not isinstance(M, (int, np.integer)):
        raise TypeError("M must be an integer")
    if not ps:
        raise ValueError("ps must contain a signal qubit")
    if len(ps) != 1:
        raise ValueError("ps must contain exactly one post-selection qubit")

    if M <= 0:
        qc.x(ps[0])
        return qc
    if M >= 1 << n:
        return qc

    bits = f"{int(M):0{n}b}"
    max_controls = max(i + 1 for i, bit in enumerate(bits) if bit == "1")
    required_ancillas = max(0, max_controls - 2)
    if len(ancillas) < required_ancillas:
        raise ValueError(
            f"pi_projector_gate requires at least {required_ancillas} ancilla "
            f"qubits for M={M}; received {len(ancillas)}"
        )

    # Begin with every state rejected, then toggle the disjoint blocks in
    # [0, M) back into the post-selected ps=0 subspace.
    qc.x(ps[0])
    most_to_least_significant = list(reversed(x))
    for i, bit in enumerate(bits):
        if bit != "1":
            continue

        pattern = bits[:i] + "0"
        controls = most_to_least_significant[: i + 1]
        zero_controls = [qubit for qubit, value in zip(controls, pattern) if value == "0"]
        for qubit in zero_controls:
            qc.x(qubit)
        prefix_ancillas = ancillas[:max(0, len(controls) - 2)]
        n_controlled_x(qc, controls, ps[0], prefix_ancillas)
        for qubit in zero_controls:
            qc.x(qubit)

    return qc
