from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.synthesis import SolovayKitaevDecomposition, qs_decomposition

import numpy as np
from Elementary_circuits import subarithmetic_circuits
from Elementary_circuits import basic_arithmetic_circuits

# TODO: add in the n-controlled x gate implementation and tests
# TODO: add in the controlled arbitrary gate using constant overhead from https://arxiv.org/abs/1206.0758
# TODO: add in the comparator operator gates
# TODO: add in the division operator gates

def n_controlled_x(qc, control_qubits, target_qubit):
    pass
    # TODO: implement the n-controlled x gate using just Clifford and T gates

# |a> -> U|a>
def arbitrary_single_qubit_gate(qc, matrix, x, epsilon=None, recursion_degree=None):
    """Apply an approximation of an arbitrary one-qubit unitary to qubit 'x' in the 
        quantum circuit 'qc' using the Solovay-Kitaev algorithm.  The approximation
        is guaranteed to be within 'epsilon' of the target unitary in operator norm.
    
        
        Gate Counts:
        - O(log^3.97(1/epsilon)) Clifford gates
        - O(log^3.97(1/epsilon)) T gates
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
    """Apply an approximation of an arbitrary n-qubit unitary to qubit 'x' in the 
            quantum circuit 'qc' using the Solovay-Kitaev algorithm.

            Specify exactly one of ``epsilon`` or ``recursion_degree``. When ``epsilon``
            is supplied, it is divided among the one-qubit gates in the Quantum Shannon
            decomposition. When ``recursion_degree`` is supplied, that same degree is
            passed to every one-qubit Solovay-Kitaev synthesis and no error parameter is
            used.
        
            
            Gate Counts:
            - O(log^3.97(1/epsilon)) Clifford gates
            - O(log^3.97(1/epsilon)) T gates
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
