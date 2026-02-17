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
import FEM_BPX_helpers as FEM

# trying to visualize the matrices from "Quantum Realization of the Finite Element Method" by Deiml M, Peterseim D and make 
# sure they can be applied effectively and work as preconditioners

# the domain goes from 0 to 1
D_min = 2
D_max = 2
L_min = 2
L_max = 7
D_vals = np.array(range(D_min,D_max + 1))
L_vals = np.array(range(L_min,L_max + 1)) # number of levels of BPX preconditioner
D_and_L = [D_vals, L_vals]
FSF_conds = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
S_conds = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
C_F_conds = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
FSF_max_sings = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
S_max_sings = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
C_F_max_sings = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
FSF_min_sings = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
S_min_sings = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
C_F_min_sings = np.zeros((D_max - D_min + 1, L_max - L_min + 1))

mat_L = 2
diffusion_mat_small = [np.diag(np.random.rand(2**(D*mat_L))) for D in D_vals]
#diffusion_mat_small = [np.eye(int(2**(D*mat_L))) for D in D_vals] # all ones diffusion coefficients

absorption_vec_small = [np.random.rand(2**(D*mat_L)) for D in D_vals] # random absorption cross sections
#absorption_vec_small = [np.ones(int(2**(D*mat_L))) for D in D_vals] # all ones absorption cross sections

for combo in itertools.product(*D_and_L):
    D = combo[0]
    L = combo[1]
    # find C_L (C_l for the finest level)
    C_L = FEM.getC_l(D, L)
    F = FEM.get_F(D, L)
    C_F = C_L @ F

    F_prime = FEM.get_F_prime(D,L)
    C = FEM.get_C(D, L)
    F_2 = F_prime @ C
    F_prime_norm = np.linalg.norm(F_prime.toarray(), ord=2)
    C_norm = np.linalg.norm(C.toarray(), ord=2)
    F_2_norm = np.linalg.norm(F_2.toarray(), ord=2)

    F_prime_cond = np.linalg.cond(F_prime.toarray())
    C_cond = np.linalg.cond(C.toarray())
    F_2_cond = np.linalg.cond(F_2.toarray())

    print("F norm: ", F_prime_norm)
    print("C norm: ", C_norm)
    print("F_2 norm: ", F_2_norm)

    print("\n F cond: ", F_prime_cond)
    print("C cond: ", C_cond)
    print("F_2 cond: ", F_2_cond)

    # testing norm of each section of F and how that compares to the norm of the entire F
    '''CF_col_sections = [(2**l-1)**D for l in range(1,L+1)]
    F_norm = np.linalg.norm(F.toarray(), ord=2)
    print("F norm: ", F_norm)
    for l in range(L):
        F_sub = F[:,sum(CF_col_sections[0:l]):sum(CF_col_sections[0:l+1])]
        F_sub_norm = np.linalg.norm(F_sub.toarray(), ord=2)
        print("Level ", l, " F sub norm: ", F_sub_norm)'''

    diffusion_mat = np.kron(diffusion_mat_small[D-D_min], np.eye(2**(D*(L - mat_L))))
    #diffusion_mat = np.diag(np.concatenate((3*np.ones(2**(D*L-1)),5*np.ones(2**(D*L-1))))) # matrix of diffusion coefficients
    D_A = np.kron(diffusion_mat, np.eye(D))

    S = np.transpose(C_L) @ np.kron(D_A, np.eye(2**D)) @ C_L
    FSF = np.transpose(F) @ S @ F # preconditioned system using the F matrix
    test = F @ np.linalg.pinv(F.toarray())

    FSF_inv = np.linalg.pinv(FSF)
    S_qc_inv = F @ FSF_inv @ np.transpose(F)
    S_inv = np.linalg.inv(S)
    S_inv_error_mat = S_qc_inv - S_inv
    S_inv_error = np.linalg.norm(S_inv_error_mat, ord=np.inf)
    print("Error between S^-1 on the QC and the actual S^-1: ", S_inv_error)
    F_rank = np.linalg.matrix_rank(F.toarray())
    S_rank = np.linalg.matrix_rank(S)

    # check if S and FSF are normal matrices
    '''T1 = np.transpose(S) @ S
    T2 = S @ np.transpose(S)
    print("S is " + ("" if np.linalg.norm(T1-T2) < 1E-10 else "not ") + "normal for D=" + str(D) + " and L="+str(L))

    T1 = np.transpose(FSF) @ FSF
    T2 = FSF @ np.transpose(FSF)
    print("FSF is " + ("" if np.linalg.norm(T1-T2) < 1E-10 else "not ") + "normal for D=" + str(D) + " and L="+str(L))'''

    _, FSF_sing_vals, _ = np.linalg.svd(FSF)
    _, S_sing_vals, _ = np.linalg.svd(S)
    _, C_F_sing_vals, _ = np.linalg.svd(C_F.toarray())

    # remove the zero singular values
    S_sing_vals = S_sing_vals[abs(S_sing_vals) > 1E-12]
    FSF_sing_vals = FSF_sing_vals[abs(FSF_sing_vals) > 1E-12]
    C_F_sing_vals = C_F_sing_vals[abs(C_F_sing_vals) > 1E-12]


    FSF_max_sing = np.max(FSF_sing_vals)
    FSF_min_sing = np.min(FSF_sing_vals)
    FSF_cond = FSF_max_sing / FSF_min_sing
    #print("FSF norm: ", FSF_norm)
    #print("FSF_inv norm: ", FSF_inv_norm)
    #print("FSF cond: ", FSF_cond)

    S_max_sing = np.max(S_sing_vals)
    S_min_sing = np.min(S_sing_vals)
    S_cond = S_max_sing / S_min_sing
    #print("\nS norm: ", S_norm)
    #print("S_inv norm: ", S_inv_norm)
    #print("S cond: ", S_cond)

    C_F_max_sing = np.max(C_F_sing_vals)
    C_F_min_sing = np.min(C_F_sing_vals)
    C_F_cond = C_F_max_sing / C_F_min_sing
    #print("\nS norm: ", S_norm)
    #print("S_inv norm: ", S_inv_norm)
    #print("S cond: ", S_cond)

    # store the condition numbers
    FSF_conds[D-D_min,L-L_min] = FSF_cond
    S_conds[D-D_min,L-L_min] = S_cond
    C_F_conds[D-D_min,L-L_min] = C_F_cond

    # store the matrix norms
    FSF_max_sings[D-D_min,L-L_min] = FSF_max_sing
    S_max_sings[D-D_min,L-L_min] = S_max_sing
    C_F_max_sings[D-D_min,L-L_min] = C_F_max_sing

    # store the matrix norms of the inverse of the matrice
    FSF_min_sings[D-D_min,L-L_min] = FSF_min_sing
    S_min_sings[D-D_min,L-L_min] = S_min_sing
    C_F_min_sings[D-D_min,L-L_min] = C_F_min_sing
    print("-------------------------\n")

# plot and print out the condition numbers of FSF, S, and C_F
for D in range(D_min, D_max + 1):
    plt.semilogy(L_vals, FSF_conds[D-D_min,:])
    plt.title("Condition Numbers vs L")
    plt.xlabel("L")
    print("FSF d = " + str(D) + " Maximum Singular Values: ", FSF_max_sings[D-D_min,:])
    print("FSF d = " + str(D) + " Minimum Singular Values: ", FSF_min_sings[D-D_min,:])
    print("FSF d = " + str(D) + " condition numbers: ", FSF_conds[D-D_min,:])

for D in range(D_min, D_max + 1):
    plt.semilogy(L_vals, S_conds[D-D_min,:])
    print("S d = " + str(D) + " Maximum Singular Values: ", S_max_sings[D-D_min,:])
    print("S d = " + str(D) + " Minimum Singular Values: ", S_min_sings[D-D_min,:])
    print("S d = " + str(D) + " condition numbers: ", S_conds[D-D_min,:])

for D in range(D_min, D_max + 1):
    print("C_F d = " + str(D) + " Maximum Singular Values: ", C_F_max_sings[D-D_min,:])
    print("C_F d = " + str(D) + " Minimum Singular Values: ", C_F_min_sings[D-D_min,:])
    print("C_F d = " + str(D) + " condition numbers: ", C_F_conds[D-D_min,:])

plt.legend(["FSF matrix D=" + str(d) for d in range(D_min, D_max + 1)] + ["S matrix D=" + str(d) for d in range(D_min, D_max + 1)])
plt.figure()

# plot the minimum and maximum singular values of FSF (preconditioned system) and S (unpreconditioned system)
for D in range(D_min, D_max + 1):
    plt.semilogy(L_vals, FSF_max_sings[D-D_min,:])
    plt.semilogy(L_vals, FSF_min_sings[D-D_min,:])
    plt.title("Condition Numbers vs L")
    plt.xlabel("L")

for D in range(D_min, D_max + 1):
    plt.semilogy(L_vals, S_max_sings[D-D_min,:])
    plt.semilogy(L_vals, S_min_sings[D-D_min,:])


plt.legend(["FSF matrix Maximum Singular Values D=" + str(d) for d in range(D_min, D_max + 1)] + ["FSF matrix Minimum Singular Values D=" + str(d) for d in range(D_min, D_max + 1)] + ["S matrix Maximum Singular Values  D=" + str(d) for d in range(D_min, D_max + 1)] + ["S matrix Minimum Singular Values D=" + str(d) for d in range(D_min, D_max + 1)])
plt.show()