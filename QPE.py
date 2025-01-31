# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Phase estimation circuit."""

from typing import Optional

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
import LcuFunctions

from qiskit.circuit.library import QFT
import numpy as np
import math
import fable


class PhaseEstimation(QuantumCircuit):
    r"""Phase Estimation circuit.

    In the Quantum Phase Estimation (QPE) algorithm [1, 2, 3], the Phase Estimation circuit is used
    to estimate the phase :math:`\phi` of an eigenvalue :math:`e^{2\pi i\phi}` of a unitary operator
    :math:`U`, provided with the corresponding eigenstate :math:`|\psi\rangle`.
    That is

    .. math::

        U|\psi\rangle = e^{2\pi i\phi} |\psi\rangle

    This estimation (and thereby this circuit) is a central routine to several well-known
    algorithms, such as Shor's algorithm or Quantum Amplitude Estimation.

    **References:**

    [1]: Kitaev, A. Y. (1995). Quantum measurements and the Abelian Stabilizer Problem. 1–22.
        `quant-ph/9511026 <http://arxiv.org/abs/quant-ph/9511026>`_

    [2]: Michael A. Nielsen and Isaac L. Chuang. 2011.
         Quantum Computation and Quantum Information: 10th Anniversary Edition (10th ed.).
         Cambridge University Press, New York, NY, USA.

    [3]: Qiskit
        `textbook <https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/
        quantum-phase-estimation.ipynb>`_

    """

    def __init__(
        self,
        num_evaluation_qubits: int,
        #unitary: QuantumCircuit,
        #unitary_gate: UnitaryGate,
        A_matrix,
        A_bits,
        iqft: Optional[QuantumCircuit] = None,
        name: str = "QPE",
        circuit: QuantumCircuit = None
    ) -> None:
        """
        Args:
            num_evaluation_qubits: The number of evaluation qubits.
            unitary: The unitary operation :math:`U` which will be repeated and controlled.
            iqft: A inverse Quantum Fourier Transform, per default the inverse of
                :class:`~qiskit.circuit.library.QFT` is used. Note that the QFT should not include
                the usual swaps!
            name: The name of the circuit.

        .. note::

            The inverse QFT should not include a swap of the qubit order.

        Reference Circuit:
            .. plot::

               from qiskit.circuit import QuantumCircuit
               from qiskit.circuit.library import PhaseEstimation
               from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
               unitary = QuantumCircuit(2)
               unitary.x(0)
               unitary.y(1)
               circuit = PhaseEstimation(3, unitary)
               _generate_circuit_library_visualization(circuit)
        """
        if(circuit == None):
            circuit = QuantumCircuit(qr_eval, qr_state, name=name)
        else:
            #assert(circuit.num_qubits >= unitary_gate.num_qubits + num_evaluation_qubits)
            print('circuit already made')

        if iqft is None:
            iqft = QFT(num_evaluation_qubits, inverse=True, do_swaps=False).reverse_bits()

        for i in range(num_evaluation_qubits):
            circuit.h(i)  # hadamards on evaluation qubits
        
        # TODO: make this more realistic, need to have a hermitian matrix as input, block encode it, then transform it to the exponential of that matrix, the control of which will be applied in this for loop
        if LcuFunctions.is_unitary(A_matrix):
            A_control = np.kron(np.array([[1,0],[0,0]]),np.eye(int(math.pow(2,A_bits)))) + np.kron(np.array([[0,0],[0,1]]),A_matrix)
            for j in range(num_evaluation_qubits):
                for k in range(2**j):
                    A_control_gate = UnitaryGate(A_control)
                    circuit.append(A_control_gate, list(range(num_evaluation_qubits, num_evaluation_qubits + A_bits)) + [j])
        else:
            for j in range(num_evaluation_qubits):
                for k in range(2**j):
                    fable.fable(A_matrix, circuit, epsilon=0, max_i = circuit.num_qubits-1, c_index=j)

        circuit.compose(iqft, qubits=list(range(num_evaluation_qubits)), inplace=True)  # final QFT

        super().__init__(*circuit.qregs, name=circuit.name)
        #self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)
