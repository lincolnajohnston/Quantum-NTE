import sys
import numpy as np
import os
sys.path.append(os.getcwd())
import scipy as sp
import scipy.sparse as spsp
from scipy.sparse import csr_matrix, coo_matrix
import itertools
from fast_inversion.BPX import FEM_BPX_helpers as FEM

# get C_l for vacuum boundary condition, with D being the total number of dimensions and D_p being the dimension of which we take the derivative
def getC_l_v_deriv(D, D_p, l):
    pi_l_C_l = csr_matrix((D*2**(D*(l+1)), (2**l + 1)**D), dtype=float)
    pi_l_C_l_s = np.array([1])
    for _ in range(1,D_p):
        pi_l_C_l_s = np.kron(pi_l_C_l_s, FEM.get_R_l_1D_v(l))
    pi_l_C_l_s = np.kron(pi_l_C_l_s, FEM.get_C_l_1D_v(l))
    for _ in range(D_p+1, D+1):
        pi_l_C_l_s = np.kron(pi_l_C_l_s, FEM.get_R_l_1D_v(l))
    rows, cols = np.nonzero(pi_l_C_l_s)
    pi_l_C_l_s_data = pi_l_C_l_s[rows, cols]

    rows_g = rows + (D_p-1)*2**(D*(l+1))
    cols_g = cols

    update = coo_matrix((pi_l_C_l_s_data, (rows_g, cols_g)), shape=pi_l_C_l.shape)
    pi_l_C_l += update.tocsr() 
    
    #pi_l_C_l[(s-1)*2**(D*(l+1)):s*2**(D*(l+1)), :] = pi_l_C_l_s
    pi_l_star_new = FEM.jk_interleave_permutation_matrix(l, D, sparse=False)
    pi_l_new = np.array([pi_l_star_new + 2**(D*l+D) * d for d in range(D)]).flatten()
    C_l = pi_l_C_l[pi_l_new, :]
    return C_l

D = 1
L = 4

########### CREATE ALL OF THE COMPONENT MATRICES FOR THE PRECONDITIONED DIRICHLET AND VACUUM SYSTEM ###########

# 0 Dirichlet BC stiffness matrix
C_L = FEM.getC_l(D, L).toarray()
S = np.transpose(C_L) @ C_L

# Robin surface integral matrix
C_L_r = FEM.getC_l_r(D, L).toarray()
R_v_test = np.transpose(C_L_r) @ C_L_r

# Vacuum BC stiffness matrix
C_L_v = FEM.getC_l_v(D, L).toarray()
S_v = np.transpose(C_L_v) @ C_L_v
R_v = FEM.get_Robin_matrix_v_brute_force(L, D) # extra Robin term for Vacuum BC becasue the surface integral doesn't go to 0
S_v = S_v # if you include the Robin matrix in the preconditioned system the condition number is significantly higher (from the few tests I did)

# Vacuum BC stiffness matrix made from adding the individual second order derivative stiffness matrices
S_v_test = np.zeros((len(S_v), len(S_v)))
for D_p in range(1,D+1):
    C_L_v_deriv = getC_l_v_deriv(D, D_p, L).toarray()
    S_v_deriv = np.transpose(C_L_v_deriv) @ C_L_v_deriv
    S_v_test += S_v_deriv

# Preconditioned Dirichlet BC stiffness matrix
F = FEM.get_F(D, L).toarray()
G_F = np.transpose(F) @ S @ F

# Preconditioned Vacuum BC stiffness matrix
F_v = FEM.get_F_v(D, L).toarray()
S_v_F = F_v.T @ S_v @ F_v # preconditioned stiffness matrix
R_v_F = F_v.T @ R_v @ F_v # preconditioned Robin surface integral matrix
G_v_F = S_v_F + R_v_F

C_Fr = FEM.get_C_F_r(D, L).toarray()
#C_Fr = C_L_r @ F_v # directly creating C_F using F_v and C_L_r which we already made instead of crafting it level by level
R_v_CF = C_Fr.T @ C_Fr

# trying to use code to find C_{f,v}
'''eigvals, Q = np.linalg.eigh(R_v)
# numerical cleanup: remove tiny negative values from roundoff
eigvals = np.maximum(eigvals, 0)
C_Fv = np.diag(np.sqrt(eigvals)) @ Q.T'''

# Test if S_v_F^{+} S_v_F = I (which would be true if S_v_F is full row rank)
identity_S =  S @ np.linalg.pinv(S) # S is full row rank
identity_S_v =  S_v @ np.linalg.pinv(S_v) # S_v is not full row rank
identity_F_v =  F_v @ np.linalg.pinv(F_v) # F_v is full row rank

# 0 Dirichlet BC CF matrix to create the preconditioned system
C_F = FEM.get_C_F(D, L).toarray()
G_CF = C_F.T @ C_F

# Vacuum BC CF matrix to create the preconditioned system
C_F_v = FEM.get_C_F_v(D, L).toarray()
G_v_CF = C_F_v.T @ C_F_v

G_error = G_v_CF - G_v_F
G_error_scalar = np.linalg.norm(G_error) # this is basically 0 so pretty sure C_F and F are implemented correctly (unless I made the same errors in both implementations)

G_v_error = G_v_CF - G_v_F
G_v_error_scalar = np.linalg.norm(G_v_error) # this is basically 0 so pretty sure C_F_v and F_v are implemented correctly (unless I made the same errors in both implementations)

### Mass matrices ###
abs_matrix = np.ones(2**(L*D)).reshape([2**L] * D)
A = FEM.get_mass_matrix_brute_force(L, D, abs_matrix).toarray()
A_v = FEM.get_mass_matrix_v_brute_force(L, D, abs_matrix).toarray()

########### FIND SINGULAR VALUES OF THESE MATRICES ###########
_, S_sing_vals, _ = np.linalg.svd(S)
_, G_F_sing_vals, _ = np.linalg.svd(G_F)
_, S_v_sing_vals, _ = np.linalg.svd(S_v)
_, G_v_F_sing_vals, _ = np.linalg.svd(G_v_F)

# remove the zero singular values
S_sing_vals = S_sing_vals[abs(S_sing_vals) > 1E-10] # the Dirichlet stiffness matrix should be invertible so this shouldn't do anything
G_F_sing_vals = G_F_sing_vals[abs(G_F_sing_vals) > 1E-10]
S_v_sing_vals = S_v_sing_vals[abs(S_v_sing_vals) > 1E-10] # The vacuum stiffness matrix will likely be singular so this will actually remove values
G_v_F_sing_vals = G_v_F_sing_vals[abs(G_v_F_sing_vals) > 1E-10]


# Dirichlet BC matrices
S_max_sing = np.max(S_sing_vals)
S_min_sing = np.min(S_sing_vals)
S_cond = S_max_sing / S_min_sing
print("S maximum singular value: ", S_max_sing)
print("S minimum singular value norm: ", S_min_sing)
print("S condition number: ", S_cond, "\n")

G_F_max_sing = np.max(G_F_sing_vals)
G_F_min_sing = np.min(G_F_sing_vals)
G_F_cond = G_F_max_sing / G_F_min_sing
print("G_F maximum singular value: ", G_F_max_sing)
print("G_F minimum singular value norm: ", G_F_min_sing)
print("G_F condition number: ", G_F_cond, "\n\n")


# Vacuum BC matrices
S_v_max_sing = np.max(S_v_sing_vals)
S_v_min_sing = np.min(S_v_sing_vals)
S_v_cond = S_v_max_sing / S_v_min_sing
print("S_v maximum singular value: ", S_v_max_sing)
print("S_v minimum singular value norm: ", S_v_min_sing)
print("S_v condition number: ", S_v_cond, "\n")

G_v_F_max_sing = np.max(G_v_F_sing_vals)
G_v_F_min_sing = np.min(G_v_F_sing_vals)
G_v_F_cond = G_v_F_max_sing / G_v_F_min_sing
print("G_v_F maximum singular value: ", G_v_F_max_sing)
print("G_v_F minimum singular value norm: ", G_v_F_min_sing)
print("G_v_F condition number: ", G_v_F_cond)


print("done")