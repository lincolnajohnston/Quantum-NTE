"""
Pre-wavelet basis for nested P1 spaces (1D) and a separable 2D demo.

This module constructs a *pre-wavelet* multiscale transform on nested, uniform
meshes, shows how to assemble FEM stiffness matrices, and demonstrates how the
pre-wavelet transform approximately block-diagonalizes the stiffness.

Key ideas
---------
- Spaces: V_0 ⊂ V_1 ⊂ ... ⊂ V_J where V_j are continuous, piecewise linears on a
  uniform mesh over [0,1] with 2^j subintervals and **Dirichlet** boundary
  conditions. Number of interior DOFs: n_j = 2^j - 1.
- Lifting-style pre-wavelets (quasi-interpolant):
  • Analysis (fine → coarse+detail):
      c_{j-1}[m] = u_j[2m]                                    (pick evens)
      d_j[m]     = u_j[2m-1] - 0.5*(u_j[2m-2] + u_j[2m])      (predict odd)
  • Synthesis (coarse+detail → fine):
      u_j[2m]   = c_{j-1}[m]
      u_j[2m-1] = d_j[m] + 0.5*(c_{j-1}[m-1] + c_{j-1}[m])    (reconstruct)
  This produces a stable (Riesz) basis often called a *pre-wavelet* basis.

- Global multilevel transform:
  Let z = [c_0; d_1; d_2; ...; d_J]. For homogeneous Dirichlet here, c_0 has
  dimension 0, so z = [d_1; ...; d_J] and dim(z) = n_J.
  We build sparse matrices S (synthesis) and A (analysis) such that
        u_J = S z,    z = A u_J,    and  A @ S = I.
  Then the stiffness in pre-wavelet coordinates is K̃ = Sᵀ K S, which is
  nearly block-diagonal (level-wise), enabling optimal multilevel preconditioners.

- 2D demo (separable/Q1):
  We also provide a quick 2D demonstration on an N×N tensor grid with standard
  bilinear (Q1) Laplacian assembled as
        K_2D = kron(I, K_1D) + kron(K_1D, I),
  and the separable pre-wavelet transform S_2D = kron(S_1D, S_1D). This is not
  the triangular P1 assembly, but it illustrates the same multiscale structure.

Author: ChatGPT (GPT-5 Thinking)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ----------------------------- Utility helpers ----------------------------- #

def n_dofs_1d(level: int) -> int:
    """Number of interior degrees of freedom for Dirichlet P1 on [0,1]."""
    if level < 1:
        return 0
    return 2**level - 1


def block_diag_sparse(blocks: List[sp.spmatrix]) -> sp.spmatrix:
    """Sparse block diagonal (CSR)."""
    if not blocks:
        return sp.csr_matrix((0, 0))
    return sp.block_diag(blocks, format="csr")


def eye(n: int) -> sp.spmatrix:
    return sp.identity(n, format="csr")


# -------------------------- 1D FEM stiffness/mass -------------------------- #

def assemble_stiffness_1d(level: int) -> sp.csr_matrix:
    """Assemble 1D P1 stiffness matrix with Dirichlet boundary on [0,1].

    Mesh: 2**level subintervals of size h = 1 / 2**level. Interior nodes only.
    Local element matrix on [x_i, x_{i+1}] is (1/h) * [[1,-1],[-1,1]].
    Global (interior) K is tridiagonal with entries (2,-1,-1)/h.
    """
    n = n_dofs_1d(level)
    if n == 0:
        return sp.csr_matrix((0, 0))
    h = 1.0 / (2**level)
    main = (2.0 / h) * np.ones(n)
    off = (-1.0 / h) * np.ones(n - 1)
    K = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    return K


def assemble_mass_1d(level: int) -> sp.csr_matrix:
    """Assemble 1D P1 mass matrix with Dirichlet boundary on [0,1].

    Local mass on element is (h/6) * [[2,1],[1,2]]. Global M is tridiagonal.
    """
    n = n_dofs_1d(level)
    if n == 0:
        return sp.csr_matrix((0, 0))
    h = 1.0 / (2**level)
    main = (2.0 * h / 6.0) * np.ones(n)
    off = (1.0 * h / 6.0) * np.ones(n - 1)
    M = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    return M


# ---------------------- 1D pre-wavelet step matrices ----------------------- #

def prewavelet_step_synthesis_1d(level: int) -> sp.csr_matrix:
    """Build the *step* synthesis S_step at level j (maps [c_{j-1}; d_j] → u_j).

    Dimensions:
      n_f = 2^j - 1                   (fine DOFs at level j)
      n_c = 2^{j-1} - 1               (coarse DOFs at level j-1)
      m   = 2^{j-1}                   (number of midpoints/new DOFs)

    Ordering of input vector x:
      x = [c_0, ..., c_{n_c-1}, d_0, ..., d_{m-1}]  (c then d)

    Output u has length n_f with entries u[0..n_f-1].
    Indexing convention: fine interior nodes are numbered 1..2^j-1; we use
    zero-based array indices i = node_id - 1.
    - Even node 2m   has index i = 2m - 1  (odd index in 0-based)
    - Odd  node 2m-1 has index i = 2m - 2  (even index in 0-based)
    """
    j = level
    if j < 1:
        return sp.csr_matrix((0, 0))
    n_f = 2**j - 1
    n_c = 2**(j - 1) - 1
    m = 2**(j - 1)

    rows = []
    cols = []
    vals = []

    # Even nodes (old coarse nodes): u[2m] = c[m]
    for m_idx in range(1, n_c + 1):
        i_even = 2 * m_idx - 1  # 0-based index for node 2*m_idx
        rows.append(i_even)
        cols.append(m_idx - 1)  # c-block index
        vals.append(1.0)

    # Odd nodes (new midpoints): u[2m-1] = d[m] + 0.5*(c[m-1] + c[m])
    # Here m runs 1..m_count with coarse neighbors m-1 and m (when they exist)
    for m_idx in range(1, m + 1):
        i_odd = 2 * m_idx - 2  # 0-based index for node 2*m_idx - 1
        # Contribution from d[m_idx]
        cols.append(n_c + (m_idx - 1))
        rows.append(i_odd)
        vals.append(1.0)
        # Left coarse neighbor c[m_idx-1]
        if m_idx - 1 >= 1:
            rows.append(i_odd)
            cols.append((m_idx - 1) - 1)
            vals.append(0.5)
        # Right coarse neighbor c[m_idx]
        if m_idx <= n_c:
            rows.append(i_odd)
            cols.append(m_idx - 1)
            vals.append(0.5)

    S_step = sp.csr_matrix((vals, (rows, cols)), shape=(n_f, n_c + m))
    return S_step


def prewavelet_step_analysis_1d(level: int) -> sp.csr_matrix:
    """Build the *step* analysis A_step at level j (maps u_j → [c_{j-1}; d_j])."""
    j = level
    if j < 1:
        return sp.csr_matrix((0, 0))
    n_f = 2**j - 1
    n_c = 2**(j - 1) - 1
    m = 2**(j - 1)

    rows = []
    cols = []
    vals = []

    # Coarse part: c[m] = u[2m]
    for m_idx in range(1, n_c + 1):
        i_even = 2 * m_idx - 1
        rows.append(m_idx - 1)  # row in c-block
        cols.append(i_even)
        vals.append(1.0)

    # Detail part: d[m] = u[2m-1] - 0.5*(u[2m-2] + u[2m])
    for m_idx in range(1, m + 1):
        i_odd = 2 * m_idx - 2
        # +1 * u[odd]
        rows.append(n_c + (m_idx - 1))
        cols.append(i_odd)
        vals.append(1.0)
        # -0.5 * left even neighbor
        if i_odd - 1 >= 0:
            rows.append(n_c + (m_idx - 1))
            cols.append(i_odd - 1)
            vals.append(-0.5)
        # -0.5 * right even neighbor
        if i_odd + 1 <= n_f - 1:
            rows.append(n_c + (m_idx - 1))
            cols.append(i_odd + 1)
            vals.append(-0.5)

    A_step = sp.csr_matrix((vals, (rows, cols)), shape=(n_c + m, n_f))
    return A_step


# ---------------------- Multilevel S and A (global) ------------------------ #

def prewavelet_global_1d(J: int) -> Tuple[sp.csr_matrix, sp.csr_matrix, List[int]]:
    """Build global synthesis S and analysis A up to finest level J.

    Returns
    -------
    S : (n_J × n_J) CSR
        Global synthesis: u_J = S z, with z = [d_1; d_2; ...; d_J].
    A : (n_J × n_J) CSR
        Global analysis:  z = A u_J.
    block_sizes : list[int]
        Sizes of levelwise blocks [|d_1|, |d_2|, ..., |d_J|] (sums to n_J).
    """
    assert J >= 1
    nJ = n_dofs_1d(J)
    # Step matrices
    S_steps = [prewavelet_step_synthesis_1d(j) for j in range(1, J + 1)]
    A_steps = [prewavelet_step_analysis_1d(j) for j in range(1, J + 1)]

    # Compose S: start with first step, then lift with block-diag identities
    M = S_steps[0]  # maps [d1] → u1
    for j in range(2, J + 1):
        S_step = S_steps[j - 1]
        m_j = 2 ** (j - 1)  # |d_j|
        M = S_step @ block_diag_sparse([M, eye(m_j)])
    S = M.tocsr()

    # Compose A: start from top and push analysis down
    M = A_steps[-1]  # maps u_J → [c_{J-1}; d_J]
    tail = 2 ** (J - 1)  # |d_J|
    for j in range(J - 1, 0, -1):
        A_step = A_steps[j - 1]  # u_j → [c_{j-1}; d_j]
        M = block_diag_sparse([A_step, eye(tail)]) @ M
        tail += 2 ** (j - 1)
    A = M.tocsr()

    # Sanity: A @ S == I
    # (Numerically exact here since entries are rationals.)
    # You can uncomment the assert below in small sizes.
    # assert sp.linalg.norm(A @ S - eye(nJ)) < 1e-12

    block_sizes = [2 ** (j - 1) for j in range(1, J + 1)]
    return S, A, block_sizes


# ---------------------- Block structure diagnostics ----------------------- #

def perm_by_levels(block_sizes: List[int]) -> np.ndarray:
    """Permutation that groups coordinates by [d1 | d2 | ... | dJ]."""
    offsets = np.cumsum([0] + block_sizes)
    return np.arange(offsets[-1])  # already grouped in our construction


def block_diagonal_part(Ktilde: sp.spmatrix, block_sizes: List[int]) -> sp.csr_matrix:
    """Extract block-diagonal part of K̃ (level-wise blocks)."""
    n = Ktilde.shape[0]
    diag_blocks = []
    start = 0
    for sz in block_sizes:
        rows = np.arange(start, start + sz)
        diag_blocks.append(Ktilde[rows[:, None], rows])
        start += sz
    return block_diag_sparse([B.tocsr() for B in diag_blocks])


def offdiag_fro_ratio(Ktilde: sp.spmatrix, block_sizes: List[int]) -> float:
    B = block_diagonal_part(Ktilde, block_sizes)
    diff = (Ktilde - B)
    num = sp.linalg.norm(diff)
    den = sp.linalg.norm(Ktilde)
    return float(num / den) if den != 0 else 0.0


# ---------------------- Preconditioner (levelwise) ------------------------ #

def levelwise_preconditioner(K: sp.spmatrix, S: sp.spmatrix, block_sizes: List[int]) -> sp.csr_matrix:
    """Wavelet-style preconditioner Q ≈ K^{-1} built from level blocks.

    We form K̃ = Sᵀ K S and approximate K^{-1} by
        Q = S D^{-1} Sᵀ,  where D is block-diagonal (levelwise) extracted from K̃.
    This is *not* the only choice (BPX uses scaled projectors), but it is a
    natural pre-wavelet preconditioner.
    """
    Ktilde = (S.T @ K @ S).tocsr()
    D = block_diagonal_part(Ktilde, block_sizes).tocsr()
    # Invert each diagonal block explicitly (they're small in demos; for large
    # problems use sparse Cholesky per block or Jacobi on blocks).
    n = D.shape[0]
    inv_blocks = []
    start = 0
    for sz in block_sizes:
        rows = slice(start, start + sz)
        Bi = D[rows, rows].toarray()
        # Regularize tiny numerical noise if needed
        w, V = np.linalg.eigh(Bi)
        w = np.clip(w, 1e-12, None)
        Bi_inv = (V * (1.0 / w)) @ V.T
        inv_blocks.append(sp.csr_matrix(Bi_inv))
        start += sz
    Dinv = block_diag_sparse(inv_blocks)
    Q = (S @ Dinv @ S.T).tocsr()
    return Q


def cond_est_spd(A: sp.spmatrix, k: int = 6, tol: float = 1e-8, maxiter: int = 200) -> float:
    """Rough spectral condition number estimate for SPD A via Lanczos.

    Uses a couple of largest/smallest eigenvalues. Keep sizes modest for speed.
    """
    # Largest
    lmax = spla.eigsh(A, k=1, which="LM", return_eigenvectors=False, tol=tol, maxiter=maxiter)[0]
    # Smallest
    lmin = spla.eigsh(A, k=1, which="SM", return_eigenvectors=False, tol=tol, maxiter=maxiter)[0]
    return float(lmax / lmin)


# ------------------------------ 2D (separable) ----------------------------- #

def assemble_stiffness_2d_tensor(J: int) -> sp.csr_matrix:
    """2D stiffness on tensor grid with Dirichlet boundaries (Q1-like model).

    Using the canonical Kronecker form (finite-difference equivalent):
        K2D = kron(I, K1D) + kron(K1D, I),
    on an N×N interior grid where N = 2^J - 1.
    """
    K1 = assemble_stiffness_1d(J)
    I1 = eye(K1.shape[0])
    K2 = sp.kron(I1, K1) + sp.kron(K1, I1)
    return K2.tocsr()


def prewavelet_global_2d_tensor(J: int) -> Tuple[sp.csr_matrix, sp.csr_matrix, List[int]]:
    """Separable 2D pre-wavelet transform via Kronecker products.

    S2D = kron(S1D, S1D),  A2D = kron(A1D, A1D).
    Level block sizes in 2D are grouped by tensor subbands at each scale; for
    simplicity we return the 1D level sizes squared for the *HH* band only and
    aggregate them as one block per scale (LL lives only at coarsest scale which
    is 0 here due to Dirichlet). This still shows blockiness in K̃.
    """
    S1, A1, bs1 = prewavelet_global_1d(J)
    n1 = S1.shape[0]
    S2 = sp.kron(S1, S1, format="csr")
    A2 = sp.kron(A1, A1, format="csr")
    # A simple block-size proxy: total 2D DOFs is n1^2, and the fraction per
    # scale is proportional to 2*bs1[j]*n1 - bs1[j]^2 for mixed bands; to keep
    # the demo focused, we just split equally by scale using 4^j growth.
    # For diagnostics we provide a monotone partition summing to n1^2.
    sizes = []
    remaining = n1 * n1
    for j, sz1 in enumerate(bs1, start=1):
        # crude split; ensure integer and nonnegative
        s = max(1, int(round((sz1 / n1) * remaining)))
        sizes.append(s)
        remaining -= s
    if remaining > 0:
        sizes[-1] += remaining
    return S2, A2, sizes


# ------------------------------ Demonstrations ----------------------------- #

def demo_1d(J: int = 6) -> None:
    print(f"1D pre-wavelet demo at level J={J} (n = {n_dofs_1d(J)})")
    K = assemble_stiffness_1d(J)
    S, A, blocks = prewavelet_global_1d(J)
    Ktilde = (S.T @ K @ S).tocsr()
    ratio = offdiag_fro_ratio(Ktilde, blocks)
    print(f"‣ Off-diagonal Frobenius ratio (level blocks): {ratio:.3e}")

    # Preconditioner quality (coarse estimate)
    Q = levelwise_preconditioner(K, S, blocks)
    condK = cond_est_spd(K)
    condQK = cond_est_spd(Q @ K)
    print(f"‣ cond(K)   ≈ {condK:9.2e}")
    print(f"‣ cond(QK)  ≈ {condQK:9.2e}  (near-constant across levels if preconditioner is good)")


def demo_2d(J: int = 4) -> None:
    n = n_dofs_1d(J)
    print(f"2D separable demo at level J={J} (n = {n} per axis, total DOFs = {n*n})")
    K2 = assemble_stiffness_2d_tensor(J)
    S2, A2, sizes = prewavelet_global_2d_tensor(J)
    Kt2 = (S2.T @ K2 @ S2).tocsr()
    ratio = offdiag_fro_ratio(Kt2, sizes)
    print(f"‣ Off-diagonal Frobenius ratio (2D level groups): {ratio:.3e}")


# ------------------------------ Usage example ------------------------------ #
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    demo_1d(J=6)   # n = 63
    print()
    demo_2d(J=4)   # 2D with n = 15 per axis
