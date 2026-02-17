import sys
import os
sys.path.append(os.getcwd())
from helpers.ProblemData import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import math

# Create the matrices needed to apply the inverse of the diffusion term in the diffusion equation using the Woodbury correction matrix
# Assume a 1-D, 2-material problem with constant D and sigma_a within each material, and constant h (cell width)
# throughout the problem domain

# google AI created this function
def create_tridiagonal(n, a, b, c):
    """
    Creates an n x n tridiagonal matrix.

    Args:
        n: The size of the matrix (number of rows and columns).
        a: List of subdiagonal elements (length n-1).
        b: List of main diagonal elements (length n).
        c: List of superdiagonal elements (length n-1).

    Returns:
        A NumPy array representing the tridiagonal matrix.
    """
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = b[i]
        if i > 0:
            matrix[i, i - 1] = a[i - 1]
        if i < n - 1:
            matrix[i, i + 1] = c[i]
    return matrix

#return the edge-averaged diffusion coefficient (at the edge of a cell)
# which is the harmonic mean of the diffusion coefficient divided by the cell width, dx
def get_edge_av_diff_coef(D1, D2, dx1, dx2):
    return 2 * (D1/dx1) * (D2/dx2) / (D1/dx1 + D2/dx2)


#n_list = np.array([1]) # big matrix size
#n_list = np.array(range(10,300,10))
n_list = np.array([8,16,32,64,128])
h = 1/n_list
k = 2 # size of small matrix (correction for boundary between materials)
m = 4 # location of material boundary (first cell of new material)
L_diffs_L1 = np.zeros(len(n_list))
L_diffs_L2 = np.zeros(len(n_list))
L_diffs_max_sing = np.zeros(len(n_list))
R_norms_L1 = np.zeros(len(n_list))
R_norms_L2 = np.zeros(len(n_list))
R_norms_max_sing = np.zeros(len(n_list))
C_conds = np.zeros(len(n_list)) # condition numbers for the C correction matrix
C_inv_conds = np.zeros(len(n_list)) # condition numbers for the inverse of the C correction matrix
D0 = 7
D1 = 8
sigma_a_0 = 12
sigma_a_1 = 19
k_0 = 0.5
k_1 = 0.5

for n_i,n in enumerate(n_list):
    D_01 = get_edge_av_diff_coef(D0, D1, h[n_i], h[n_i])
    h2 = h[n_i]*h[n_i]
    L = 1/(h2) * create_tridiagonal(n,-1*np.ones(n-1), 2*np.ones(n), -1*np.ones(n-1)) # O(log(n))
    L_inv = np.linalg.inv(L) # O(log(n)), can do DCT/DST matrices inverses in same time as forward operators

    D_mat = np.diag(np.concatenate((D0*np.ones(m),D1*np.ones(n-m)))) # O(M), M is number of materials, O(1) in term of n
    D_mat_inv = np.diag(np.concatenate((1/D0*np.ones(m),1/D1*np.ones(n-m)))) # O(M), M is number of materials, O(1) in term of n
    A_mat = D_mat @ L # O(Mlog(n)) to encode quantumly
    A_mat_inv = D_mat_inv @ L_inv # O(Mlog(n)) to encode quantumly

    sigma_a_mat = np.diag(np.concatenate((sigma_a_0*np.ones(m),sigma_a_1*np.ones(n-m))))
    sigma_a_mat_prime = sigma_a_mat.copy()
    sigma_a_mat_prime[m-1,m-1] *= (1-k_0)
    sigma_a_mat_prime[m,m] *= (1-k_1)

    V_mat = np.zeros((k,n))
    U_mat = np.zeros((n,k))
    for i in range(k):
        V_mat[i,m+i] = 1 
        U_mat[m+i,i] = 1
    A_inv_sub = V_mat @ A_mat_inv @ U_mat
    _, s1, _ = np.linalg.svd(A_inv_sub)
    A_inv_sub_inv = np.linalg.inv(A_inv_sub)
    _, s2, _ = np.linalg.svd(A_inv_sub_inv)

    C_mat = np.array([[-D0/h2 + D_01/h[n_i] + k_0*sigma_a_mat[m-1,m-1],D0/h2-D_01/h[n_i]],[D1/h2-D_01/h[n_i],-D1/h2 + D_01/h[n_i] + k_1*sigma_a_mat[m,m]]]) # each element in this 2x2 matrix scales as O(1/h^2)
    C_mat_inv = np.linalg.inv(C_mat) # elements in this matrix are O(1) w.r.t. h as h->0
    C_conds[n_i] = np.linalg.cond(C_mat) # condition number of C matrix scales as O(h^2), this also implies the condition number of C_mat_inv scales as O(h^2), whch means both will be ill-conditioned as h->0

    P_mat_inv = C_mat_inv + A_inv_sub
    P_mat = np.linalg.inv(P_mat_inv)

    P_C_dif_mat = P_mat - C_mat
    print(P_C_dif_mat)

    Q = (U_mat@P_mat@V_mat) @ A_mat_inv
    R = A_mat_inv @ Q
    R_norm_L1 = np.max(R)
    R_norm_L2 = np.linalg.norm(R)
    R_norm_max_sing = np.linalg.norm(R, ord=2)

    Q_approx = (U_mat@C_mat@V_mat) @ A_mat_inv
    #Q_approx = (U_mat@np.linalg.inv(A_inv_sub)@V_mat) @ A_mat_inv
    R_approx = A_mat_inv @ Q_approx

    L_final_mat = A_mat_inv - R
    L_final_approx_mat = A_mat_inv - R_approx
    L_diff = L_final_mat - L_final_approx_mat # same as (R_approx - R)
    L_diff_L1_norm = np.max(L_diff)
    L_diff_L2_norm = np.linalg.norm(L_diff)
    L_diff_max_sing = np.linalg.norm(L_diff, ord=2)

    #print("A_inv: ", A_mat_inv)
    #print("L_final", L_final_mat)
    #print("L_final_approx", L_final_approx_mat)
    #print("L_diff:", L_diff)
    print("L1 norm of difference of exact Woodbury inversion and inversion with P=C:", L_diff_L1_norm)
    print("L2 norm of difference of exact Woodbury inversion and inversion with P=C:", L_diff_L2_norm)
    print("matrix norm (maximum singular value) of difference of exact Woodbury inversion and inversion with P=C:", L_diff_max_sing)
    print("L1 norm of correction to A_inv introduced by C: ", R_norm_L1 )
    print("L2 norm of correction to A_inv introduced by C: ", R_norm_L2 )
    print("matrix norm (maximum singular value) of correction to A_inv introduced by C: ", R_norm_max_sing )
    L_diffs_L1[n_i] = L_diff_L1_norm
    R_norms_L1[n_i] = R_norm_L1
    L_diffs_L2[n_i] = L_diff_L2_norm
    R_norms_L2[n_i] = R_norm_L2
    L_diffs_max_sing[n_i] = L_diff_max_sing
    R_norms_max_sing[n_i] = R_norm_max_sing


#plt.loglog(h,L_diffs_L1) # scales as O(h^6)
#plt.loglog(h,R_norms_L1) # scales as O(h^4)
#plt.loglog(h,L_diffs_L2) # scales as O(h^5)
#plt.loglog(h,R_norms_L2) # scales as O(h^3)
plt.loglog(h,L_diffs_max_sing) # scales as O(h^5)
plt.loglog(h,R_norms_max_sing) # scales as O(h^3)
plt.loglog(h,1E-5*h)
plt.loglog(h,1E-5*h*h)
plt.loglog(h,1E-5*h*h*h)
plt.loglog(h,1E-3*h*h*h*h)
plt.loglog(h,1E-1*h*h*h*h*h)
plt.loglog(h,1E1*h*h*h*h*h*h)
#plt.loglog(h,1E-5*np.exp(h))
plt.legend(["L_diffs","R_norms", "O(h^1)", "O(h^2)", "O(h^3)","O(h^4)","O(h^5)","O(h^6)"])
plt.figure()

plt.loglog(n_list, C_inv_conds)
plt.xlabel("n")
plt.ylabel("condition number of the C matrix")
plt.show()

print("done")