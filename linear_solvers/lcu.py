# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The LCU algorithm, copied the HHL file and made modifications"""

from typing import Optional, Union, List, Callable, Tuple
import numpy as np
import math

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.opflow import (
    Z,
    I,
    StateFn,
    TensoredOp,
    ExpectationBase,
    CircuitSampler,
    ListOp,
    ExpectationFactory,
    ComposedOp,
)
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance

from .linear_solver import LinearSolver, LinearSolverResult
from .matrices.numpy_matrix import NumPyMatrix
from .observables.linear_system_observable import LinearSystemObservable
from scipy.linalg import ishermitian, expm
from qiskit.quantum_info.operators import Operator

# Perform Gram-Schmidt orthogonalization to have basis of vectors including the alpha vector
# Function modified from ChatGPT suggestion
def gram_schmidt_ortho(vector):
    num_dimensions = len(vector)
    ortho_basis = np.zeros((num_dimensions, num_dimensions), dtype=float)
    
    
    # Normalize the input vector and add it to the orthogonal basis
    ortho_basis[0] = vector / np.linalg.norm(vector)
    #print(ortho_basis)

    # Gram-Schmidt orthogonalization
    for i in range(1, num_dimensions):
        ortho_vector = np.random.rand(num_dimensions)  # random initialization
        if(abs(np.log2(i) - np.ceil(np.log2(i))) < 0.00001):
            print("dimension: ", i)
        for j in range(i):
            ortho_vector -= np.dot(ortho_basis[j], ortho_vector) / np.dot(ortho_basis[j], ortho_basis[j]) * ortho_basis[j]
        ortho_basis[i] = ortho_vector / np.linalg.norm(ortho_vector)
    
    return ortho_basis

def is_unitary(matrix):
    I = matrix.dot(np.conj(matrix).T)
    return I.shape[0] == I.shape[1] and np.allclose(I, np.eye(I.shape[0]))

class LCU(LinearSolver):
    r"""Dis is LCU

    Examples:

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit
            from quantum_linear_solvers.linear_solvers.hhl import HHL
            from quantum_linear_solvers.linear_solvers.matrices import TridiagonalToeplitz
            from quantum_linear_solvers.linear_solvers.observables import MatrixFunctional

            matrix = TridiagonalToeplitz(2, 1, 1 / 3, trotter_steps=2)
            right_hand_side = [1.0, -2.1, 3.2, -4.3]
            observable = MatrixFunctional(1, 1 / 2)
            rhs = right_hand_side / np.linalg.norm(right_hand_side)

            # Initial state circuit
            num_qubits = matrix.num_state_qubits
            qc = QuantumCircuit(num_qubits)
            qc.isometry(rhs, list(range(num_qubits)), None)

            hhl = HHL()
            solution = hhl.solve(matrix, qc, observable)
            approx_result = solution.observable

    References:

        [1]: Harrow, A. W., Hassidim, A., Lloyd, S. (2009).
        Quantum algorithm for linear systems of equations.
        `Phys. Rev. Lett. 103, 15 (2009), 1–15. <https://doi.org/10.1103/PhysRevLett.103.150502>`_

        [2]: Carrera Vazquez, A., Hiptmair, R., & Woerner, S. (2020).
        Enhancing the Quantum Linear Systems Algorithm using Richardson Extrapolation.
        `arXiv:2009.04484 <http://arxiv.org/abs/2009.04484>`_

    """

    def __init__(
        self,
        epsilon: float = 1e-2,
        expectation: Optional[ExpectationBase] = None,
        quantum_instance: Optional[Union[Backend, QuantumInstance]] = None,
    ) -> None:
        r"""
        Args:
            epsilon: Error tolerance of the approximation to the solution, i.e. if :math:`x` is the
                exact solution and :math:`\tilde{x}` the one calculated by the algorithm, then
                :math:`||x - \tilde{x}|| \le epsilon`.
            expectation: The expectation converter applied to the expectation values before
                evaluation. If None then PauliExpectation is used.
            quantum_instance: Quantum Instance or Backend. If None, a Statevector calculation is
                done.
        """
        super().__init__()

        self._epsilon = epsilon
        # Tolerance for the different parts of the algorithm as per [1]
        self._epsilon_r = epsilon / 3  # conditioned rotation
        self._epsilon_s = epsilon / 3  # state preparation
        self._epsilon_a = epsilon / 6  # hamiltonian simulation

        self._scaling = None  # scaling of the solution

        self._sampler = None
        self.quantum_instance = quantum_instance

        self._expectation = expectation

        # For now the default reciprocal implementation is exact
        self._exact_reciprocal = True
        # Set the default scaling to 1
        self.scaling = 1

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Get the quantum instance.

        Returns:
            The quantum instance used to run this algorithm.
        """
        return None if self._sampler is None else self._sampler.quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Optional[Union[QuantumInstance, Backend]]
    ) -> None:
        """Set quantum instance.

        Args:
            quantum_instance: The quantum instance used to run this algorithm.
                If None, a Statevector calculation is done.
        """
        if quantum_instance is not None:
            self._sampler = CircuitSampler(quantum_instance)
        else:
            self._sampler = None

    @property
    def scaling(self) -> float:
        """The scaling of the solution vector."""
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: float) -> None:
        """Set the new scaling of the solution vector."""
        self._scaling = scaling

    @property
    def expectation(self) -> ExpectationBase:
        """The expectation value algorithm used to construct the expectation measurement from
        the observable."""
        return self._expectation

    @expectation.setter
    def expectation(self, expectation: ExpectationBase) -> None:
        """Set the expectation value algorithm."""
        self._expectation = expectation

    def _get_delta(self, n_l: int, lambda_min: float, lambda_max: float) -> float:
        """Calculates the scaling factor to represent exactly lambda_min on nl binary digits.

        Args:
            n_l: The number of qubits to represent the eigenvalues.
            lambda_min: the smallest eigenvalue.
            lambda_max: the largest eigenvalue.

        Returns:
            The value of the scaling factor.
        """
        formatstr = "#0" + str(n_l + 2) + "b"
        lambda_min_tilde = np.abs(lambda_min * (2**n_l - 1) / lambda_max)
        # floating point precision can cause problems
        if np.abs(lambda_min_tilde - 1) < 1e-7:
            lambda_min_tilde = 1
        binstr = format(int(lambda_min_tilde), formatstr)[2::]
        lamb_min_rep = 0
        for i, char in enumerate(binstr):
            lamb_min_rep += int(char) / (2 ** (i + 1))
        return lamb_min_rep

    def _calculate_norm(self, qc: QuantumCircuit) -> float:
        """Calculates the value of the euclidean norm of the solution.

        Args:
            qc: The quantum circuit preparing the solution x to the system.

        Returns:
            The value of the euclidean norm of the solution.
        """
        # Calculate the number of qubits
        nb = qc.qregs[0].size
        nl = qc.qregs[1].size
        na = qc.num_ancillas

        # Create the Operators Zero and One
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2

        # Norm observable
        observable = one_op ^ TensoredOp((nl + na) * [zero_op]) ^ (I ^ nb)
        norm_2 = (~StateFn(observable) @ StateFn(qc)).eval()

        return np.real(np.sqrt(norm_2) / self.scaling)

    def _calculate_observable(
        self,
        solution: QuantumCircuit,
        ls_observable: Optional[LinearSystemObservable] = None,
        observable_circuit: Optional[QuantumCircuit] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]], int, float], float]
        ] = None,
    ) -> Tuple[float, Union[complex, List[complex]]]:
        """Calculates the value of the observable(s) given.

        Args:
            solution: The quantum circuit preparing the solution x to the system.
            ls_observable: Information to be extracted from the solution.
            observable_circuit: Circuit to be applied to the solution to extract information.
            post_processing: Function to compute the value of the observable.

        Returns:
            The value of the observable(s) and the circuit results before post-processing as a
             tuple.
        """
        # Get the number of qubits
        nb = solution.qregs[0].size
        nl = solution.qregs[1].size
        na = solution.num_ancillas

        # if the observable is given construct post_processing and observable_circuit
        if ls_observable is not None:
            observable_circuit = ls_observable.observable_circuit(nb)
            post_processing = ls_observable.post_processing

            if isinstance(ls_observable, LinearSystemObservable):
                observable = ls_observable.observable(nb)

        # in the other case use the identity as observable
        else:
            observable = I ^ nb

        # Create the Operators Zero and One
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2

        is_list = True
        if not isinstance(observable_circuit, list):
            is_list = False
            observable_circuit = [observable_circuit]
            observable = [observable]

        expectations: Union[ListOp, ComposedOp] = []
        for circ, obs in zip(observable_circuit, observable):
            circuit = QuantumCircuit(solution.num_qubits)
            circuit.append(solution, circuit.qubits)
            circuit.append(circ, range(nb))

            ob = one_op ^ TensoredOp((nl + na) * [zero_op]) ^ obs
            expectations.append(~StateFn(ob) @ StateFn(circuit))

        if is_list:
            # execute all in a list op to send circuits in batches
            expectations = ListOp(expectations)
        else:
            expectations = expectations[0]

        # check if an expectation converter is given
        if self._expectation is not None:
            expectations = self._expectation.convert(expectations)
        # if otherwise a backend was specified, try to set the best expectation value
        elif self._sampler is not None:
            if is_list:
                op = expectations.oplist[0]
            else:
                op = expectations
            self._expectation = ExpectationFactory.build(
                op, self._sampler.quantum_instance
            )

        if self._sampler is not None:
            expectations = self._sampler.convert(expectations)

        # evaluate
        expectation_results = expectations.eval()

        # apply post_processing
        result = post_processing(expectation_results, nb, self.scaling)

        return result, expectation_results

    def construct_circuit(
        self,
        matrix: Union[List, np.ndarray, QuantumCircuit],
        vector: Union[List, np.ndarray, QuantumCircuit],
        neg_vals: Optional[bool] = True,
    ) -> QuantumCircuit:
        """Construct the HHL circuit.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            neg_vals: States whether the matrix has negative eigenvalues. If False the
            computation becomes cheaper.

        Returns:
            The HHL circuit.

        Raises:
            ValueError: If the input is not in the correct format.
            ValueError: If the type of the input matrix is not supported.
        """
        # State preparation circuit - default is qiskit
        if isinstance(vector, QuantumCircuit):
            nb = vector.num_qubits
            vector_circuit = vector
        elif isinstance(vector, (list, np.ndarray)):
            if isinstance(vector, list):
                vector = np.array(vector)
            nb = int(np.log2(len(vector)))
            vector_circuit = QuantumCircuit(nb)
            # pylint: disable=no-member
            vector_circuit.isometry(
                vector / np.linalg.norm(vector), list(range(nb)), None
            )

        # If state preparation is probabilistic the number of qubit flags should increase
        nf = 1

        # Hamiltonian simulation circuit - default is Trotterization
        if isinstance(matrix, QuantumCircuit):
            matrix_circuit = matrix
        elif isinstance(matrix, (list, np.ndarray)):
            if isinstance(matrix, list):
                matrix = np.array(matrix)

            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square!")
            if np.log2(matrix.shape[0]) % 1 != 0:
                raise ValueError("Input matrix dimension must be 2^n!")
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError("Input matrix must be hermitian!")
            if matrix.shape[0] != 2**vector_circuit.num_qubits:
                raise ValueError(
                    "Input vector dimension does not match input "
                    "matrix dimension! Vector dimension: "
                    + str(vector_circuit.num_qubits)
                    + ". Matrix dimension: "
                    + str(matrix.shape[0])
                )
            matrix_circuit = NumPyMatrix(matrix, evolution_time=2 * np.pi)
        else:
            raise ValueError(f"Invalid type for matrix: {type(matrix)}.")

        # Set the tolerance for the matrix approximation
        if hasattr(matrix_circuit, "tolerance"):
            matrix_circuit.tolerance = self._epsilon_a

        # check if the matrix can calculate the condition number and store the upper bound
        if (
            hasattr(matrix_circuit, "condition_bounds")
            and matrix_circuit.condition_bounds() is not None
        ):
            kappa = matrix_circuit.condition_bounds()[1]
        else:
            kappa = 1
        # Update the number of qubits required to represent the eigenvalues
        # The +neg_vals is to register negative eigenvalues because
        # e^{-2 \pi i \lambda} = e^{2 \pi i (1 - \lambda)}
        nl = max(nb + 1, int(np.ceil(np.log2(kappa + 1)))) + neg_vals

        # check if the matrix can calculate bounds for the eigenvalues
        if (
            hasattr(matrix_circuit, "eigs_bounds")
            and matrix_circuit.eigs_bounds() is not None
        ):
            lambda_min, lambda_max = matrix_circuit.eigs_bounds()
            # Constant so that the minimum eigenvalue is represented exactly, since it contributes
            # the most to the solution of the system. -1 to take into account the sign qubit
            delta = self._get_delta(nl - neg_vals, lambda_min, lambda_max)
            # Update evolution time
            matrix_circuit.evolution_time = (
                2 * np.pi * delta / lambda_min / (2**neg_vals)
            )
            # Update the scaling of the solution
            self.scaling = lambda_min
        else:
            delta = 1 / (2**nl)
            print("The solution will be calculated up to a scaling factor.")

        if self._exact_reciprocal:
            reciprocal_circuit = ExactReciprocal(nl, delta, neg_vals=neg_vals)
            # Update number of ancilla qubits
            na = matrix_circuit.num_ancillas
        else:
            # Calculate breakpoints for the reciprocal approximation
            num_values = 2**nl
            constant = delta
            a = int(round(num_values ** (2 / 3)))

            # Calculate the degree of the polynomial and the number of intervals
            r = 2 * constant / a + np.sqrt(np.abs(1 - (2 * constant / a) ** 2))
            degree = min(
                nb,
                int(
                    np.log(
                        1
                        + (
                            16.23
                            * np.sqrt(np.log(r) ** 2 + (np.pi / 2) ** 2)
                            * kappa
                            * (2 * kappa - self._epsilon_r)
                        )
                        / self._epsilon_r
                    )
                ),
            )
            num_intervals = int(np.ceil(np.log((num_values - 1) / a) / np.log(5)))

            # Calculate breakpoints and polynomials
            breakpoints = []
            for i in range(0, num_intervals):
                # Add the breakpoint to the list
                breakpoints.append(a * (5**i))

                # Define the right breakpoint of the interval
                if i == num_intervals - 1:
                    breakpoints.append(num_values - 1)

            reciprocal_circuit = PiecewiseChebyshev(
                lambda x: np.arcsin(constant / x), degree, breakpoints, nl
            )
            na = max(matrix_circuit.num_ancillas, reciprocal_circuit.num_ancillas)


        # Do LCU routine (https://arxiv.org/pdf/1511.02306.pdf), equation 18
        A_mat_size = len(matrix)
        J = 64
        K = 16
        num_unitaries = 2 * J * K
        num_LCU_bits = math.ceil(np.log2(num_unitaries))
        ql = QuantumRegister(num_LCU_bits)  # LCU ancilla zero bits
        cl = ClassicalRegister(num_LCU_bits)  # right hand side and solution
        y_max = 3
        z_max = 3
        delta_y = y_max / J
        delta_z = z_max / K

        # Initialise the quantum registers
        qb = QuantumRegister(nb)  # right hand side and solution
        if na > 0:
            qa = AncillaRegister(na)  # ancilla qubits
        #qf = QuantumRegister(nf)  # flag qubits

        if na > 0:
            qc = QuantumCircuit(qb, ql, qa)
        else:
            qc = QuantumCircuit(qb, ql)

        # State preparation
        qc.append(vector_circuit, qb[:])

        M = np.zeros((A_mat_size, A_mat_size))
        U = np.zeros((A_mat_size * pow(2,num_LCU_bits), A_mat_size * pow(2,num_LCU_bits)), dtype=complex)
        alphas = np.zeros(pow(2,num_LCU_bits))
        alpha = 0

        for j in range(J):
            print("J: ", j)
            y = j * delta_y
            for k in range(-K,K):
                #condition_mat = np.zeros((pow(2,num_LCU_bits), pow(2,num_LCU_bits)))
                #condition_mat[j * J + (k + K),j * J + (k + K)] = 1
                z = k * delta_z
                alpha_temp = (1) / math.sqrt(2 * math.pi) * delta_y * delta_z * z * math.exp(-z*z/2)
                uni_mat = (1j) * expm(-(1j) * matrix * y * z)
                assert(is_unitary(uni_mat))
                M_temp = alpha_temp * uni_mat
                #U_temp = U_temp + np.kron(condition_mat, uni_mat)
                if(alpha_temp < 0): # if alpha is negative, incorporate negative phase into U unitary
                    alpha_temp *= -1
                    U[A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1),A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1)] = -1 * uni_mat
                else:
                    U[A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1),A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1)] = uni_mat
                alpha += alpha_temp
                alphas[2 * j * K + (k + K)] = alpha_temp
                M = M + M_temp
        #print(U)
        matrix_invert = np.linalg.inv(matrix)
        print(matrix_invert)
        #print("prob: ", math.pow(np.linalg.norm(np.matmul(matrix, vector/ np.linalg.norm(vector))),2))
        print(M)
        V = gram_schmidt_ortho(alphas)
        V = np.kron(V, np.eye(A_mat_size))
        W = np.matmul(np.conj(V).T, np.matmul(U , V))


        # check if W is unitary
        #print("V is unitary: ", is_unitary(V))
        #print("U is unitary: ", is_unitary(U))
        #print("W is unitary: ", is_unitary(W))

        LCU = Operator(W)
        qc.unitary(LCU, ql[:] + qb[:], label='lcu')
        #qc.measure(ql[:],cl[:])
        #qc.append(M_circuit, qb[:])
        return qc

    def solve(
        self,
        matrix: Union[List, np.ndarray, QuantumCircuit],
        vector: Union[List, np.ndarray, QuantumCircuit],
        observable: Optional[
            Union[
                LinearSystemObservable,
                List[LinearSystemObservable],
            ]
        ] = None,
        observable_circuit: Optional[
            Union[QuantumCircuit, List[QuantumCircuit]]
        ] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]], int, float], float]
        ] = None,
    ) -> LinearSolverResult:
        """Tries to solve the given linear system of equations.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            observable: Optional information to be extracted from the solution.
                Default is the probability of success of the algorithm.
            observable_circuit: Optional circuit to be applied to the solution to extract
                information. Default is `None`.
            post_processing: Optional function to compute the value of the observable.
                Default is the raw value of measuring the observable.

        Raises:
            ValueError: If an invalid combination of observable, observable_circuit and
                post_processing is passed.

        Returns:
            The result object containing information about the solution vector of the linear
            system.
        """
        # verify input
        if observable is not None:
            if observable_circuit is not None or post_processing is not None:
                raise ValueError(
                    "If observable is passed, observable_circuit and post_processing cannot be set."
                )

        solution = LinearSolverResult()
        solution.state = self.construct_circuit(matrix, vector)
        solution.euclidean_norm = self._calculate_norm(solution.state)

        if isinstance(observable, List):
            observable_all, circuit_results_all = [], []
            for obs in observable:
                obs_i, circ_results_i = self._calculate_observable(
                    solution.state, obs, observable_circuit, post_processing
                )
                observable_all.append(obs_i)
                circuit_results_all.append(circ_results_i)
            solution.observable = observable_all
            solution.circuit_results = circuit_results_all
        elif observable is not None or observable_circuit is not None:
            solution.observable, solution.circuit_results = self._calculate_observable(
                solution.state, observable, observable_circuit, post_processing
            )

        return solution
