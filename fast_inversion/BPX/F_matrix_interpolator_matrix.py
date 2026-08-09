import sys
import os
sys.path.append(os.getcwd())
from helpers.ProblemData import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
from scipy.sparse import csr_matrix, coo_matrix
import scipy as sp
import math
import itertools

# returns the interpolator from l to l+1 ((2**(l+1)-1) x (2**l - 1) matrix)
def get_increment_interpolator(l):
    N = 2**l
    Inc = np.zeros((2*N-1, N-1))
    for i in range(N-1):
        Inc[2*i:2*i+3,i] = [0.5,1,0.5]
    return Inc


# return the interpolator matrix from l to L ((2**L-1) x (2**l - 1) matrix)
def get_interpolator(l, L):
    Inc_master = get_increment_interpolator(l)
    for i in range(l+1, L):
        Inc = get_increment_interpolator(i)
        Inc_master = Inc @ Inc_master
    return Inc_master

# ChatGPT function to perform a swap in a quantum circuit given a list of the new permutation of the qubits
def qubit_permutation_matrix(perm):
    """
    Construct the full 2^n x 2^n unitary matrix that permutes qubits.

    perm[i] = j  means qubit at index i is moved to index j.
    """
    n = len(perm)
    dim = 2**n
    U = np.zeros((dim, dim), dtype=int)

    # Precompute inverse permutation because output bits at position k
    # come from input bit at position perm_inv[k]
    perm_inv = np.argsort(perm)

    for col in range(dim):
        # Convert basis index to bit array (LSB = qubit 0)
        bits = [(col >> i) & 1 for i in range(n)]

        # Permute bits using inverse permutation
        new_bits = [bits[perm_inv[i]] for i in range(n)]

        # Convert permuted bitstring back to an integer index
        row = sum(new_bits[i] << i for i in range(n))

        U[row, col] = 1

    return U

# ChatGPT function to fill in an incomplete matrix with other orthonormal columns
def complete_from_zero_columns(M, completed_cols, tol=1e-12, random_state=None, verbose=False):
    """
    Given an n x n matrix M where some columns are already filled and the
    other columns are all zeros, fill the zero columns with vectors that
    are orthonormal to each other and to the provided completed columns.
    The provided completed columns are kept exactly as provided (not reorthonormalized).
    
    Parameters
    ----------
    M : (n, n) ndarray
        Square matrix. Some columns should be nonzero (the 'completed' ones),
        and the remaining columns should be exactly zero (these will be filled).
    completed_cols : sequence of ints
        Indices of columns (0-based) that are already filled and should be preserved.
    tol : float
        Numerical tolerance for orthonormality / zero checks.
    random_state : None | int | numpy.random.Generator
        If int, used as seed for RNG for random candidate vectors. If None, RNG is random.
    verbose : bool
        If True, print diagnostic messages.
    
    Returns
    -------
    Q : (n, n) ndarray
        Matrix with the same first dimension n, with columns completed so Q is an
        orthonormal basis. Provided completed columns are identical to M[:, completed_cols].
    
    Raises
    ------
    ValueError
        If M is not square, if completed_cols contain invalid indices, if any
        of the "zero" columns is not (close to) zero, or if the provided completed
        columns are not mutually orthonormal (within tol).
    """
    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError("M must be 2D.")
    n, n2 = M.shape
    if n != n2:
        raise ValueError("M must be square (n x n).")
    completed_cols = list(completed_cols)
    if any((c < 0 or c >= n) for c in completed_cols):
        raise ValueError("completed_cols contains out-of-range column index.")
    # Check zero columns: columns not in completed_cols must be (close to) zero
    zero_indices = [i for i in range(n) if i not in completed_cols]
    for i in zero_indices:
        if np.linalg.norm(M[:, i]) > tol:
            raise ValueError(f"Column {i} is not zero but is not listed in completed_cols.")
    # Get the provided columns matrix and verify orthonormality
    if completed_cols:
        Q0 = M[:, completed_cols].astype(complex if np.iscomplexobj(M) else float)
        # Check linear independence/orthonormality: Q0^* Q0 should be identity
        gram = Q0.conj().T @ Q0
        I_k = np.eye(len(completed_cols), dtype=gram.dtype)
        if np.linalg.norm(gram - I_k) > tol * max(1.0, np.linalg.norm(gram)):
            raise ValueError("Provided completed columns are not mutually orthonormal within tol.")
    else:
        # No prefilled columns
        Q0 = np.zeros((n, 0), dtype=M.dtype)
    # RNG
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    # Helper: project a candidate off the span of 'current_cols' and return normalized vector or None
    def project_and_normalize(v, current_cols):
        # subtract projection onto current_cols
        if current_cols.shape[1] > 0:
            coeffs = current_cols.conj().T @ v           # (k,)
            w = v - current_cols @ coeffs
        else:
            w = v
        nw = np.linalg.norm(w)
        if nw > tol:
            return w / nw
        return None
    # Build matrix of accepted columns, starting with the provided ones (kept exact)
    accepted = Q0.copy()
    # Fill columns in ascending index order
    new_cols = {}
    # deterministic pass over standard basis vectors (in order)
    for e in np.eye(n, dtype=accepted.dtype):
        if len(new_cols) >= len(zero_indices):
            break
        w = project_and_normalize(e, accepted)
        if w is not None:
            # find next zero column index to fill
            idx = zero_indices[len(new_cols)]
            new_cols[idx] = w
            # append to accepted and continue
            accepted = np.column_stack([accepted, w])
    # If still need columns, use random vectors
    tries = 0
    max_tries = 5000
    while len(new_cols) < len(zero_indices) and tries < max_tries:
        if np.iscomplexobj(M):
            v = rng.normal(size=n) + 1j * rng.normal(size=n)
        else:
            v = rng.normal(size=n)
        w = project_and_normalize(v, accepted)
        if w is not None:
            idx = zero_indices[len(new_cols)]
            new_cols[idx] = w
            accepted = np.column_stack([accepted, w])
        tries += 1
    if len(new_cols) < len(zero_indices):
        # fallback: perform eigen-decomposition of projector onto orthogonal complement
        if verbose:
            print("Random attempts exhausted; falling back to eig on projector.")
        P = np.eye(n, dtype=accepted.dtype) - accepted @ accepted.conj().T
        eigvals, eigvecs = np.linalg.eigh(P)
        # choose eigenvectors with largest eigenvalues (should be near 1)
        idxs = np.argsort(-eigvals)
        for ei in idxs:
            if len(new_cols) >= len(zero_indices):
                break
            if eigvals[ei] > tol * 1e2:
                idx = zero_indices[len(new_cols)]
                new_cols[idx] = eigvecs[:, ei]
                accepted = np.column_stack([accepted, eigvecs[:, ei]])
    # Final check: ensure we have all indices filled
    if len(new_cols) < len(zero_indices):
        raise RuntimeError("Could not find enough orthogonal vectors to complete the basis.")
    # Assemble result: start from copy of M, fill columns
    Q = M.copy().astype(complex if np.iscomplexobj(M) else float)
    for idx, colvec in new_cols.items():
        # ensure the placed vector is exactly what's in 'accepted' (normalized)
        Q[:, idx] = colvec
    # Sanity check: orthonormality of entire matrix
    err = np.linalg.norm(Q.conj().T @ Q - np.eye(n))
    if verbose:
        print(f"Final orthonormality error ||Q^* Q - I||_F = {err:.3e}")
    if err > 1e-8:
        # small numerical issues could remain; attempt to re-orthonormalize *only* the new columns
        if verbose:
            print("Re-orthonormalizing newly created columns to reduce numerical error (preserving original columns).")
        # take out provided columns and re-orthonormalize the complement submatrix against the provided ones
        K = len(completed_cols)
        # Build matrix with exactly the provided columns first, then the new columns in the same order as zero_indices
        complement_cols = np.column_stack([Q[:, j] for j in zero_indices])
        # orthonormalize complement while keeping them orthogonal to provided columns
        # We'll do QR on the projected complement:
        # 1) project complement onto orthogonal complement of provided columns (should already be)
        if K > 0:
            proj_off = complement_cols - Q[:, completed_cols] @ (Q[:, completed_cols].conj().T @ complement_cols)
        else:
            proj_off = complement_cols
        # 2) QR
        Qc, R = np.linalg.qr(proj_off, mode='reduced')
        # 3) put back into Q in their original indices
        for i, col_idx in enumerate(zero_indices):
            Q[:, col_idx] = Qc[:, i]
        err2 = np.linalg.norm(Q.conj().T @ Q - np.eye(n))
        if verbose:
            print(f"After re-orthonormalizing complement, error = {err2:.3e}")
        if err2 > 1e-8:
            raise RuntimeError("Could not achieve a numerically orthonormal matrix to desired tolerance.")
    return Q

# return the P matrix that is used in the contruction of the Interpolator matrix
# two registers are used, the first, |j> stays in its original state, while the
# second undergoes the transformation |0> -> (0.5|2j> + 1|2j+1> + 0.5|2j+2>)
# the first register has 2^a qubits, the second register has 2^(a+1)
def get_P(a):
    N_a = 1 << a # same as 2**a but for bit notation
    N_P = N_a * (2*N_a)
    P = np.zeros((N_P, N_P))
    for j in range(N_a): # index of the first register
        P[j*(2*N_a) + 2*j,j*(2*N_a)] = 1/2
        P[j*(2*N_a) + 2*j+1,j*(2*N_a)] = 1/np.sqrt(2)
        P[(j*(2*N_a) + (2*j+2)%(2*N_a)),j*(2*N_a)] = 1/2
    return P


# return the Q matrix that is used in the contruction of the Interpolator matrix
# two registers are used, the first, |j> stays in its original state, while the
# second undergoes the transformation |0> -> (1/sqrt(2)|floor(j/2)> + 1/sqrt(2)|floor(j/2)-1>) if
# j%2 == 0 else |0> -> |j/2>
# the first register has 2^(a+1) qubits, the second register has 2^a
def get_Q(a):
    N_a = 1 << a # same as 2**a but for bit notation
    N_Q = N_a * (2*N_a)
    Q = np.zeros((N_Q, N_Q))
    for j in range(2*N_a): # index of the first register
        test = int(j/2 - 1) % N_a
        if j%2==0:
            Q[j*(N_a) + int(j/2),j*(N_a)] = 1/np.sqrt(2)
            Q[j*(N_a) + int(j/2 - 1) % N_a,j*(N_a)] = 1/np.sqrt(2)
        else:
            Q[j*(N_a) + int((j-1)/2),j*(N_a)] = 1
    return Q



l = 3
L = 6
#incrementer = get_increment_interpolator(l)
#print(l)

#inc_master = get_interpolator(l, L)
#print(inc_master)

# test creating an offset state from ChatGPT suggestion
'''input_state = np.array([1,0,0,0])
theta_1 = math.pi/3
ry1 = np.array([[math.cos(theta_1/2), -math.sin(theta_1/2)], [math.sin(theta_1/2), math.cos(theta_1/2)]])
ry1 = np.kron(ry1, np.eye(2))

theta_2 = 2 * math.acos(1/math.sqrt(3))
ry2 = np.array([[math.cos(theta_2/2), -math.sin(theta_2/2)], [math.sin(theta_2/2), math.cos(theta_2/2)]])
cry2 = np.eye(4)
cry2[:2,:2] = ry2

offset_mat = cry2 @ ry1
print(offset_mat)'''

# Test the P and Q method for making the interpolator matrix
a = 3
N_a = 1 << a
P = get_P(a)
P = complete_from_zero_columns(P, 2*N_a*np.array(range(N_a)))
print(P)
Q = get_Q(a)
Q = complete_from_zero_columns(Q, N_a*np.array(range(2*N_a)))
print(Q)

SWAP = qubit_permutation_matrix(list(range(a,2*a+1)) + list(range(a)))

Int = np.transpose(Q) @ SWAP @ P

state = np.zeros((2**(2*a+1)))
