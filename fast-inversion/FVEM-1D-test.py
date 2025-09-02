import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
import math

#  Use example 6.1 from "Multilevel preconditioning for the finite volume method" by Xu Y 
# to precondition a FVEM matrix to make the condition number grow as O(1) with discretization level

# TODO: Use BPX preconditioner instead of the wavelet preconditioner
# TODO: See if we can implement these preconditioners efficiently on a quantum circuit
# TODO: Make sure there is conservation of particles like in other finite volume methods:
    # Each variation equation is made from a conservation law: a(u, v) = (f,v).   a(u,v) is an integral of the 
    # diffusion of particles over the boundaries of the test space, (f,v) is the integral of the source function over the test space volume
# TODO: how are piecewise constant diffusion coefficients handled? (instead of a continuously varying function)
    # Case 5 from section 6 of the paper shows that discontinuous diffusion coefficients with the BPX preconditioner will be alright, the preconditioner still works the same


# Returns the matrix that does a basis transform from the wavelet basis to the hat function nodal basis
def getL(n_min, n_max):
    N_max = int(math.pow(2,n_max)) # size of most fine wavelet set
    N_min = int(math.pow(2,n_min)) # size of most coarse wavelet set
    L = np.zeros((N_max-1,N_max-N_min)) # Basis transformation matrix from wavelet to nodal hat function basis

    s = 0 # leftmost row index in stencil
    j = 1 # jump between discrete points on wavelet stencil grid (in terms of number of points on finest grid)
    ref = 0 # current column
    for i in range(n_max-n_min):
        N_cur = int(N_max * math.pow(2,-i-1)) # size of wavelet set at the current level
        dilation_radius = int(2**i - 1) # radius of points at which the fine nodal basis is needed to represent the current wavelet
        dilation_list = 1/(2**i) * np.array(list(range(1,2**(i)+1,1)) + list(range(2**(i)-1,0,-1)))

        # special case for most coarse wavelet (only one point)
        if (n_min == 0 and i == n_max -1):
            for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
                L[s + 0*j + off,ref+0] += dilation_list[dil_index]
            continue
            

        # column 0 (left boundary):
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s + 0*j + off,ref+0] += dilation_list[dil_index] * 9/10
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s + 1*j + off,ref+0] += dilation_list[dil_index] * -3/5
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s + 2*j + off,ref+0] += dilation_list[dil_index] * 1/10

        # columns [1,N-2] (interior points):
        for l in range(1,N_cur-1):
            for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
                L[s+j*(2*l-2) + off,ref+l] += dilation_list[dil_index] * 1/10
            for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
                L[s+j*(2*l-1) + off,ref+l] += dilation_list[dil_index] * -3/5
            for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
                L[s+j*(2*l) + off,ref+l] += dilation_list[dil_index] * 1
            for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
                L[s+j*(2*l+1) + off,ref+l] += dilation_list[dil_index] * -3/5
            for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
                L[s+j*(2*l+2) + off,ref+l] += dilation_list[dil_index] * 1/10

        # column N-1 (right boundary):
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s+j*(2*N_cur-4) + off,ref+N_cur-1] += dilation_list[dil_index] * 1/10
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s+j*(2*N_cur-3) + off,ref+N_cur-1] += dilation_list[dil_index] * -3/5
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s+j*(2*N_cur-2) + off,ref+N_cur-1] += dilation_list[dil_index] * 9/10

        s += j
        j *= 2 # double jump size for next (coarser) grid
        ref += N_cur

    return L

# ChatGPT function (with modifications) to get the L transform matrix
def prewavelet_transform_matrix(N0: int, J: int):
    """
    Build the sparse (here dense for simplicity) matrix W that maps fine-grid
    nodal coefficients in V_J to pre-wavelet detail coefficients on levels 1..J.
    Size: rows = N0*(2**J - 1), cols = nJ = 2**J * N0 - 1.
    """
    nJ = 2**J * N0 - 1
    rows = []
    meta = []  # (j, ell) -> row index

    # stencil weights
    left  = [(0, 0.9), (1, -0.6), (2, 0.1)]
    mid   = [(-2, 0.1), (-1, -0.6), (0, 1.0), (1, -0.6), (2, 0.1)]
    right = [(-2, 0.1), (-1, -0.6), (0, 0.9)]

    for j in range(1, J+1):
        s = 2**(J - j)                 # stride on the fine grid
        start = 2**(J-j) - 1
        L = (2**(j-1)) * N0            # number of wavelets on level j

        # l = 1 (left boundary on level j)
        r = np.zeros(nJ)
        for m, w in left:
            k = start + s*m               # 1-based to 0-based
            r[k] = w
        meta.append((j, 1))
        rows.append(r)

        # 2 <= l <= L-1 (interior)
        for ell in range(2, L):
            r = np.zeros(nJ)
            c = s*(2*ell - 1) - 1      # center (0-based)
            for off, w in mid:
                r[c + off*s] = w
            meta.append((j, ell))
            rows.append(r)

        # l = L (right boundary on level j)
        r = np.zeros(nJ)
        base = s*(2*L - 1) - 1
        for off, w in right:
            r[base + off*s] = w
        meta.append((j, L))
        rows.append(r)

    W = np.vstack(rows)
    return np.transpose(W), meta


# (NxN) is size of D_inv matrix
# n is the number of indices for sets of wavelet basis matrices
def getD_inv(n_min, n_max, coarse_to_fine=False):
    N_max = int(math.pow(2,n_max))
    N_min = int(math.pow(2,n_min))
    D_inv = np.zeros((N_max - N_min,N_max - N_min))
    m = 0
    n_list = range(n_min, n_max) if coarse_to_fine else range(n_max-1,n_min-1,-1)
    for i in n_list:
        for j in range(int(math.pow(2,i))):
            D_inv[m,m] = math.pow(2,-i)
            m += 1
    return D_inv

# get the stiffness matrix for the diffusion equation
# N is the size of the system
# size of p_vals must be N/(2^m) where m is an integer
def getA(N, p_vals_coarse):
    p_vals = np.kron(p_vals_coarse, np.ones(int(N / len(p_vals_coarse))))
    N_A = N-1
    A_n = np.zeros((N_A,N_A))
    for i in range(N_A):
        A_n[i,i] = p_vals[i] + p_vals[i+1]
        if i > 0:
            A_n[i,i-1] = -p_vals[i]
        if i < N_A-1:
            A_n[i,i+1] = -p_vals[i+1]
    
    return A_n

a = 0
b = 1

n_mat = 2
N_mat = 2**n_mat
#f_vals = 10 * np.random.random(N_mat - 1)

#p_vals = np.random.random(N_mat)
#p_vals = [0.1, 0.5, 1.5, 0.3] # use for n_mat = 2

n_list = list(range(3,11))
#n_list = [5,6,7,8] # number of qubits to represent the number of FV regions
#n_list = [8]
A_n_cond_list = np.zeros(len(n_list))
A_n_tilde_cond_list = np.zeros(len(n_list))
for n_i,n in enumerate(n_list):
    N = int(math.pow(2,n)) # max index of points defining domain
    N_A = N-1 # number of points excluding boundaries (size of the A matrix)
    dx = (b - a) / (N)
    p_vals = np.ones(N) + np.sqrt((2*np.array(range(1,N+1))-1) / (2*N)) # p_vals to match example 6.1 from the FVEM paper (the A_n condition numbers do match)

    # right side of the equation, is on the test space not the trial space, TODO: need to figure out how to do this correctly
    f_n = dx * np.ones(N_A)

    trial_space_pts = np.linspace(a,b,N+1) # points at the center of the nodal (hat function) basis
    test_space_pts = [a] + list(np.linspace(a+dx/2,b-dx/2,N)) + [b] # points at the center of the piecewise constant basis

    # value of diffusion coefficient in each colume defined by the trial space points
    #p_vals = 3 * np.ones(N)

    A_n = getA(N,p_vals)

    A_n_cond = np.linalg.cond(A_n)
    print("A_N condition number: ", A_n_cond)
    A_n_cond_list[n_i] = A_n_cond

    c = np.linalg.inv(A_n) @ f_n # solve the system for the coefficients on the functions in the trial basis

    n_min = 0

    # do the wavelet preconditioning
    #L = getL(int((N_A-1)/2+1))
    L = getL(n_min,n) # basis change from wavelet to hat function (hat to wavelet for basis vectors, wavelet to hats for vector components)

    L_inv = np.linalg.inv(L) # exact inverse of the wavelet to nodal basis
    '''phi_T = np.array([0.9, -0.6, 0.1] + [0]*(N-4))
    L_inv_trans = np.transpose(L_inv)
    oueabg = phi_T @ L_inv_trans'''
    L_fixed = L

    L_norm = np.linalg.norm(L, ord=2)
    L_inv_norm = np.linalg.norm(L_inv, ord=2)
    L_cond = np.linalg.cond(L)
    print("L_norm: ", L_norm)
    print("L_inv_norm: ", L_inv_norm)
    print("L_cond: ", L_cond)
    #L_inv = np.transpose(L) # "approximate inverse" of the wavelet to nodal basis

    D_inv = getD_inv(n_min,n, coarse_to_fine=False)
    D = np.linalg.inv(D_inv)

    P_1 = D_inv @ np.linalg.inv(L_fixed)
    P_2 = L_fixed @ D_inv # kind of like the inverse of the preconditioner, but not really
    P_norm = np.linalg.norm(P_1, ord=2)
    P_2_norm = np.linalg.norm(P_2, ord=2)
    P_1_cond = np.linalg.cond(P_1)
    P_2_cond = np.linalg.cond(P_2)
    print("P_1_norm: ", P_norm)
    print("P_2_norm: ", P_2_norm)
    print("P_1_cond: ", P_1_cond)
    print("P_2_cond: ", P_2_cond)

    # Using coarse to fine indexing for the wavelet bases, from the ChatGPT wavelet transform function
    #L, meta = prewavelet_transform_matrix(2**n_min, n - n_min)
    #D_inv = getD_inv(n_min,n, coarse_to_fine=True)

    # test the L matrix (basis change from hats to wavelets)
    #test_vec = np.arange(len(L))
    test_vec = np.zeros(len(L[0]))
    test_vec[1] = 1
    trans_test_vec = L @ test_vec

    A_n_tilde = P_1 @ A_n @ P_2 # Use exact inverse of basis change matrix to get maximum decrease in condition number
    A_n_tilde_inv = np.linalg.inv(A_n_tilde)

    A_n_tilde_norm = np.linalg.norm(A_n_tilde, ord=2)
    A_n_tilde_inv_norm = np.linalg.norm(A_n_tilde_inv, ord=2)
    A_n_tilde_cond = np.linalg.cond(A_n_tilde)
    print("A_n_tilde norm: ", A_n_tilde_norm)
    print("A_n_tilde_inv norm: ", A_n_tilde_inv_norm)
    print("A_n_tilde condition number: ", A_n_tilde_cond)
    A_n_tilde_cond_list[n_i] = A_n_tilde_cond

    f_n_tilde = D_inv @ L_inv @ f_n

    I = np.transpose(L) @ L
    #print(I)

    #c_tilde = np.linalg.solve(A_n_tilde, f_n_tilde) # solve the preconditioned system
    c_tilde = np.linalg.solve(A_n_tilde, f_n_tilde) # solve the preconditioned system

    #c = L @ c_tilde
    c_precond = L @ D_inv @ c_tilde

    # Use coefficients on the trial basis functions to recreate the solution in the domain
    #u = np.zeros((N+1, N+1))
    #for i in range(1,N):
    #    u[i] += c[i-1]

    u_original = np.array([0] + list(c) + [0])
    u_precond = np.array([0] + list(c_precond) + [0])

    #normalize solutions
    u_original = u_original / np.linalg.norm(u_original)
    u_precond = u_precond / np.linalg.norm(u_precond)

    '''plt.plot(trial_space_pts, u_original)
    plt.plot(trial_space_pts, u_precond)
    plt.legend(["Original Solution", "Preconditioned solution"])
    plt.title("Solutions for diffusion equation with n=" + str(n))
    plt.show()'''

plt.semilogy(n_list, A_n_cond_list)
plt.semilogy(n_list, A_n_tilde_cond_list)
plt.legend(["A_n condition number", "A_n_tilde condition number"])
plt.show()
print("done")