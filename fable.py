#!/usr/bin/env python
# -*- coding: utf-8 -*-

# code copy and pasted from https://github.com/QuantumComputingLab/fable and modified
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

def gray_code(b):
    '''Gray code of b.
    Args:
        b: int:
            binary integer
    Returns:
        Gray code of b.
    '''
    return b ^ (b >> 1)


def gray_permutation(a):
    '''Permute the vector a from binary to Gray code order.

    Args:
        a: vector
            1D NumPy array of size 2**n
    Returns:
        vector:
            Gray code permutation of a
    '''
    b = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        b[i] = a[gray_code(i)]
    return b


def sfwht(a):
    '''Scaled Fast Walsh-Hadamard transform of input vector a.

    Args:
        a: vector
            1D NumPy array of size 2**n.
    Returns:
        vector:
            Scaled Walsh-Hadamard transform of a.
    '''
    n = int(np.log2(a.shape[0]))
    for h in range(n):
        for i in range(0, a.shape[0], 2**(h+1)):
            for j in range(i, i+2**h):
                x = a[j]
                y = a[j + 2**h]
                a[j] = (x + y) / 2
                a[j + 2**h] = (x - y) / 2
    return a


def compute_control(i, n):
    '''Compute the control qubit index based on the index i and size n.'''
    if i == 4**n:
        return 1
    return 2*n - int(np.log2(gray_code(i-1) ^ gray_code(i)))


def compressed_uniform_rotation(a, ry=True):
    '''Compute a compressed uniform rotation circuit based on the thresholded
    vector a.

    Args:
        a: vector:
            A thresholded vector a a of dimension 2**n
        ry: bool
            uniform ry rotation if true, else uniform rz rotation
    Returns:
        circuit
            A qiskit circuit representing the compressed uniform rotation.
    '''
    n = int(np.log2(a.shape[0])/2)
    circ = QuantumCircuit(2*n + 1)

    i = 0
    while i < a.shape[0]:
        parity_check = 0

        # add the rotation gate
        if a[i] != 0:
            if ry:
                circ.ry(a[i], 0)
            else:
                circ.rz(a[i], 0)

        # loop over sequence of consecutive zeros
        while True:
            ctrl = compute_control(i+1, n)
            # toggle control bit
            parity_check = (parity_check ^ (1 << (ctrl-1)))
            i += 1
            if i >= a.shape[0] or a[i] != 0:
                break

        # add CNOT gates
        for j in range(1, 2*n+1):
            if parity_check & (1 << (j-1)):
                circ.cx(j, 0)

    return circ

def get_logn(a):
    n, m = a.shape
    if n != m:
        k = max(n, m)
        a = np.pad(a, ((0, k - n), (0, k - m)))
        n = k
    logn = int(np.ceil(np.log2(n)))
    return logn


def fable(a, circ = None, epsilon=None):
    '''FABLE - Fast Approximate BLock Encodings.

    Args:
        a: array
            matrix to be block encoded.
        epsilon: float >= 0
            (optional) compression threshold.
    Returns:
        circuit: qiskit circuit
            circuit that block encodes A
        alpha: float
            subnormalization factor
    '''
    epsm = np.finfo(a.dtype).eps
    alpha = np.linalg.norm(np.ravel(a), np.inf)
    if alpha > 1:
        alpha = alpha + np.sqrt(epsm)
        a = a/alpha
    else:
        alpha = 1.0

    n, m = a.shape
    logn = get_logn(a)
    if n < 2**logn:
        a = np.pad(a, ((0, 2**logn - n), (0, 2**logn - n)))
        n = 2**logn

    a = np.ravel(a)

    if all(np.abs(np.imag(a)) < epsm):  # real data
        a = gray_permutation(
                sfwht(
                    2.0 * np.arccos(np.real(a))
                )
            )
        # threshold the vector
        if epsilon:
            a[abs(a) <= epsilon] = 0
        # compute circuit
        OA = compressed_uniform_rotation(a)
    else:  # complex data
        # magnitude
        a_m = gray_permutation(
                sfwht(
                    2.0 * np.arccos(np.abs(a))
                )
            )
        if epsilon:
            a_m[abs(a_m) <= epsilon] = 0

        # phase
        a_p = gray_permutation(
                sfwht(
                    -2.0 * np.angle(a)
                )
            )
        if epsilon:
            a_p[abs(a_p) <= epsilon] = 0

        # compute circuit
        OA = compressed_uniform_rotation(a_m).compose(
                compressed_uniform_rotation(a_p, ry=False)
            )

    if(circ == None):
        circ = QuantumCircuit(2*logn + 1)
    else:
        circ = circ.reverse_bits()
        assert(circ.num_qubits >= 2*logn + 1)

    # diffusion on row indices
    for i in range(logn):
        circ.h(i+1)

    # matrix oracle
    circ = circ.compose(OA)

    # swap register
    for i in range(logn):
        circ.swap(i+1,  i+logn+1)

    # diffusion on row indices
    for i in range(logn):
        circ.h(i+1)

    # reverse bits because of little-endiannes
    circ = circ.reverse_bits()

    return circ, alpha