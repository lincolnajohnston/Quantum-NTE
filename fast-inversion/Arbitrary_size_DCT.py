import sys
import os
sys.path.append(os.getcwd())
from helpers.ProblemData import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
from scipy.fftpack import dct, dst
import math
import cmath

# use the DCT implementation to create DCT of arbitrary size.
# Assume we have access to an arbitrary size QFT.
# Goal of this script is to find methods to apply the T_N matrix efficiently
# and make sure the DCT matrix created is correct

def get_T_N(N):
    T_N_mat = np.zeros((2*N,2*N), dtype = complex)
    T_N_mat[0,0] = 1
    T_N_mat[N,N] = 1
    T_N_mat[1:N,1:N] = np.diag(1/math.sqrt(2) * np.ones(N-1))
    T_N_mat[1:N,N+1:2*N] = np.diag(1j/math.sqrt(2) * np.ones(N-1))
    for i in range(N-1):
        T_N_mat[N+1+i,N-1-i] = 1 / math.sqrt(2)
        T_N_mat[N+1+i,2*N-1-i] = -1j / math.sqrt(2)
    return T_N_mat

def get_V_N(N):
    V_N_mat = np.zeros((2*N,2*N), dtype = complex)
    V_N_mat[:N,:N] = np.diag(1 / math.sqrt(2) * np.ones(N))
    V_N_mat[0:N,N:2*N] = np.diag(1 / math.sqrt(2) * np.ones(N))
    for i in range(N):
        V_N_mat[N+i,N-1-i] = 1 / math.sqrt(2)
        V_N_mat[N+i,2*N-1-i] = -1 / math.sqrt(2)
    return V_N_mat

def get_U_N(N):
    U_N_mat = np.zeros((2*N,2*N), dtype = complex)
    U_N_mat[0,0] = 1
    U_N_mat[N,2*N-1] = 1
    omega = np.exp(2j * math.pi / (4 * N))
    omega_conj = omega.conjugate()
    for i in range(1,N):
        U_N_mat[i,i] = (1 / math.sqrt(2)) * np.power(omega_conj,i) # top left submatrix
        U_N_mat[i,N-1+i] = (-1j / math.sqrt(2)) * np.power(omega_conj,i) # top right submatrix
        U_N_mat[2*N-i,i] = 1 / math.sqrt(2) * np.power(omega,i) # bottom left submatrix
        U_N_mat[2*N-i,N-1+i] = 1j / math.sqrt(2) * np.power(omega,i) # bottom right submatrix
    U_N_mat[N,2*N-1] = -1
    return U_N_mat

# ChatGPT matrix to return the QFT matrix
def get_QFT_matrix(N: int, dtype=np.complex128) -> np.ndarray:
    """
    Return the exact N×N Quantum Fourier Transform (QFT) matrix.

    Parameters
    ----------
    N : int
        Dimension of the Hilbert space (must be ≥ 1).
    dtype : numpy.dtype, optional
        Complex dtype for the result (default: np.complex128).

    Returns
    -------
    F : np.ndarray
        The QFT matrix with entries F_{j,k} = ω^{j·k} / √N,
        where ω = exp(2πi / N).

    Notes
    -----
    * Time/space cost is O(N²); build lazily or use FFT-style
      decompositions for large N when you only need the action
      of the QFT, not the full matrix.
    """
    if N < 1 or not isinstance(N, int):
        raise ValueError("N must be a positive integer.")

    omega = np.exp(2j * np.pi / N).astype(dtype)     # primitive N-th root of unity
    j = np.arange(N, dtype=dtype)                    # shape (N,)
    k = j[:, None]                                   # column vector shape (N,1)
    F = omega ** (j * k) / np.sqrt(N, dtype=dtype)   # outer product exponentiation
    return F

# ChatGPT function to create any of the 4 types of DCT matrix
def dct_matrix(N: int, typ: int = 2, norm: str | None = "ortho",
               dtype=float) -> np.ndarray:
    r"""
    Return the $N \times N$ real matrix $D$ whose multiplication
    $y = D\,x$ performs a 1-D Discrete Cosine Transform of the requested type.

    Parameters
    ----------
    N   : int
          Size of the transform (signal length).
    typ : {1, 2, 3, 4}, default 2
          Which DCT flavour to construct.
    norm: {None, "ortho"}, optional
          • None  – raw, un-scaled definition (inverse differs by factors).  
          • "ortho" – orthonormal scaling so $D$ is orthogonal
            (unitary up to round-off for real data).
    dtype: NumPy dtype, default float
          Floating type of the returned array.

    Returns
    -------
    D : ndarray, shape (N, N)
        The DCT transform matrix.

    Notes
    -----
    • For large N this costs $O(N^2)$ memory & time; apply the fast FFT-based
      routines instead when you only need the coefficients.
    • Orthonormal factors follow SciPy 1.12’s `scipy.fft.dct`.  In that
      convention  
          – DCT-II and -III are mutual inverses,  
          – DCT-I and -IV are self-inverse.  """
    if typ not in {1, 2, 3, 4}:
        raise ValueError("typ must be 1, 2, 3, or 4")

    k = np.arange(N, dtype=dtype)[:, None]   # column indices (rows)
    n = np.arange(N, dtype=dtype)[None, :]   # row indices (columns)

    if typ == 1:                 # DCT-I
        D = np.cos(np.pi/(N-1) * k * n)
        if norm == "ortho":
            D[[0, -1], :] *= 1 / np.sqrt(2)
            D *= np.sqrt(2/(N-1))

    elif typ == 2:               # DCT-II (the common “DCT”)
        D = np.cos(np.pi/N * (n + 0.5) * k)
        if norm == "ortho":
            D[0, :] *= 1 / np.sqrt(2)
            D *= np.sqrt(2/N)

    elif typ == 3:               # DCT-III (inverse of II up to scale)
        D = np.cos(np.pi/N * n * (k + 0.5))
        if norm == "ortho":
            D[:, 0] *= 1 / np.sqrt(2)
            D *= np.sqrt(2/N)

    else:                        # typ == 4, DCT-IV (self-inverse)
        D = np.cos(np.pi/N * (n + 0.5) * (k + 0.5))
        if norm == "ortho":
            D *= np.sqrt(2/N)

    return D.astype(dtype)


 # DCT and DST Type 1 Implementation
N = 8
T_N = get_T_N(N)
F_2N = get_QFT_matrix(2*N)

DCT_1_test = np.conjugate(T_N.T) @ F_2N @ T_N
#DCT_1 = dct_matrix(N,typ=1)
#DCT_1_error = DCT_1_test[0:N+1,0:N+1] - DCT_1

# DCT and DST Type 2 Implementation
V_N = get_V_N(N)
U_N = get_U_N(N)

DCT_2_test = np.conjugate(U_N.T) @ F_2N @ V_N
DCT_2 = dct_matrix(N,typ=2)
DCT_2_error = DCT_2_test[0:N,0:N] - DCT_2

print("Total error in DCT decomposition: ", np.sum(np.abs(DCT_2_error)))
print("done")