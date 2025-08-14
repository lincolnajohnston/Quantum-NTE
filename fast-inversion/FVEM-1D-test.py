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

# L matrix where the psi values are defined all according to the most fine phi discretization
'''def getL(N):
    L = np.zeros((2*N-1,N))

    # column 0:
    L[0,0] = 9/10
    L[1,0] = -3/5
    L[2,0] = 1/10

    # columns [1,N-2]
    for l in range(1,N-1):
        L[2*l-2,l] = 1/10
        L[2*l-1,l] = -3/5
        L[2*l,l] = 1
        L[2*l+1,l] = -3/5
        L[2*l+2,l] = 1/10

    # Column N-1
    L[2*N-4,N-1] = 1/10
    L[2*N-3,N-1] = -3/5
    L[2*N-2,N-1] = 9/10

    return L'''

# L matrix where the psi points are defined only according to the phi discretization with the same level of fineness as psi
'''def getL(N):
    L = np.zeros((2*N-1,2*N-1))

    s = 0
    j = 1
    ref = 0
    for i in range(int(math.log2(N))):
        N_cur = int(N * math.pow(2,-i))
        # column 0:
        L[s+0*j,ref+0] = 9/10
        L[s+1*j,ref+0] = -3/5
        L[s+2*j,ref+0] = 1/10

        # columns [1,N-2]
        for l in range(1,N_cur-1):
            L[s+j*(2*l-2),ref+l] = 1/10
            L[s+j*(2*l-1),ref+l] = -3/5
            L[s+j*(2*l),ref+l] = 1
            L[s+j*(2*l+1),ref+l] = -3/5
            L[s+j*(2*l+2),ref+l] = 1/10

        # Column N-1
        L[s+j*(2*N_cur-4),ref+N_cur-1] = 1/10
        L[s+j*(2*N_cur-3),ref+N_cur-1] = -3/5
        L[s+j*(2*N_cur-2),ref+N_cur-1] = 9/10

        s += j
        j *= 2
        ref += N_cur

    L[N-1,ref] = 1
    return L'''

# Same as previous function, just making sure it was implemented correctly, adding an n_min
def getL(n_min, n_max):
    N_max = int(math.pow(2,n_max))
    N_min = int(math.pow(2,n_min))
    #L = np.zeros((N_max-1,N_max-1))
    L = np.zeros((N_max-1,N_max-N_min))

    s = 0
    j = 1
    ref = 0
    for i in range(n_max-n_min):
        N_cur = int(N_max * math.pow(2,-i-1))
        dilation_range = int(2**(i+1) - 1) # range of finest hat functions needed to define each wavelet function
        dilation_radius = int(2**i - 1)
        dilation_list = 1/(2**i) * np.array(list(range(1,2**(i)+1,1)) + list(range(2**(i)-1,0,-1)))
        # column 0:
        #L[s+0*j,ref+0] = 9/10
        #L[s+1*j,ref+0] = -3/5
        #L[s+2*j,ref+0] = 1/10

        # column 0:
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s + 0*j + off,ref+0] += dilation_list[dil_index] * 9/10
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s + 1*j + off,ref+0] += dilation_list[dil_index] * -3/5
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s + 2*j + off,ref+0] += dilation_list[dil_index] * 1/10

        # columns [1,N-2]
        for l in range(1,N_cur-1):
            #L[s+j*(2*l-2),ref+l] = 1/10
            #L[s+j*(2*l-1),ref+l] = -3/5
            #L[s+j*(2*l),ref+l] = 1
            #L[s+j*(2*l+1),ref+l] = -3/5
            #L[s+j*(2*l+2),ref+l] = 1/10

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

        # Column N-1
        #L[s+j*(2*N_cur-4),ref+N_cur-1] = 1/10
        #L[s+j*(2*N_cur-3),ref+N_cur-1] = -3/5
        #L[s+j*(2*N_cur-2),ref+N_cur-1] = 9/10

        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s+j*(2*N_cur-4) + off,ref+N_cur-1] += dilation_list[dil_index] * 1/10
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s+j*(2*N_cur-3) + off,ref+N_cur-1] += dilation_list[dil_index] * -3/5
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s+j*(2*N_cur-2) + off,ref+N_cur-1] += dilation_list[dil_index] * 9/10

        s += j
        j *= 2
        ref += N_cur

    # set all of the remaining rows (levels below the set lowest level) to 1
    '''for i in range(n_max-n_min, n_max):
        N_cur = int(N_max * math.pow(2,-i-1))

        # columns [1,N-2]
        for l in range(0,N_cur):
            L[s+j*(2*l),ref+l] = 1

        s += j
        j *= 2
        ref += N_cur'''

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
    n_list = range(n_min, n) if coarse_to_fine else range(n-1,n_min-1,-1)
    for i in n_list:
        for j in range(int(math.pow(2,i))):
            D_inv[m,m] = math.pow(2,-i)
            m += 1
    return D_inv

# (NxN) is size of D_inv matrix
# n is the number of indices for sets of wavelet basis matrices
'''def getD_inv(n_min, n_max):
    N = 2**(n_max - n_min)
    D_inv = np.zeros((N,N))
    for n in range(n_min, n_max + 1):
        jump = int(math.pow(2,n_max - n + 1))
        start = int(math.pow(2,n_max - n)) - 1
        for j in range(start,N,jump):
            D_inv[j,j] = jump / math.pow(2,n_max - n_min)
    return D_inv'''

a = 0
b = 1

n = 8 # number of qubits to represent the number of FV regions
N = int(math.pow(2,n)) # max index of points defining domain
N_A = N-1 # number of points excluding boundaries (size of the A matrix)
dx = (b - a) / (N)

f = 1
f_n = f * dx * np.ones(N_A)

trial_space_pts = np.linspace(a,b,N+1)
test_space_pts = [a] + list(np.linspace(a+dx/2,b-dx/2,N)) + [b]
#print(trial_space_pts)
#print(test_space_pts)

#p_vals = np.random.rand(N)
p_vals = 3 * np.ones(N) # value of diffusion coefficient in each finite volume

A_n = np.zeros((N_A,N_A))
for i in range(N_A):
    A_n[i,i] = p_vals[i] + p_vals[i+1]
    if i > 0:
        A_n[i,i-1] = -p_vals[i]
    if i < N-2:
        A_n[i,i+1] = -p_vals[i+1]

A_n_cond = np.linalg.cond(A_n)
print("A_N condition number: ", A_n_cond)

c = np.linalg.inv(A_n) @ f_n # solve the system for the coefficients on the functions in the trial basis

n_min = 1

# do the wavelet preconditioning
#L = getL(int((N_A-1)/2+1))
L = getL(n_min,n) # basis change from wavelet to hat function
D_inv = getD_inv(n_min,n, coarse_to_fine=False)

# Using coarse to fine indexing for the wavelet bases, from the ChatGPT wavelet transform function
#L, meta = prewavelet_transform_matrix(2**n_min, n - n_min)
#D_inv = getD_inv(n_min,n, coarse_to_fine=True)

# test the L matrix (basis change from hats to wavelets)
#test_vec = np.arange(len(L))
test_vec = np.zeros(len(L[0]))
test_vec[1] = 1
trans_test_vec = L @ test_vec

#A_n_tilde = D_inv @ np.transpose(L) @ A_n @ L @ D_inv # Use the transpose of the basis change matrix because it is easier to calculte (but doesnt precondition as well)
A_n_tilde = D_inv @ np.linalg.pinv(L) @ A_n @ L @ D_inv # Use exact inverse of basis change matrix to get maximum decrease in condition number
#A_n_tilde = np.transpose(L) @ A_n @ L

A_n_tilde_cond = np.linalg.cond(A_n_tilde)
print("A_n_tilde condition number: ", A_n_tilde_cond)

#f_n_tilde = D_inv @ np.transpose(L) @ f_n
f_n_tilde = np.transpose(L) @ f_n
#c_tilde = np.linalg.solve(A_n_tilde, f_n_tilde) # solve the preconditioned system
c_tilde = np.linalg.solve(A_n_tilde, f_n_tilde) # solve the preconditioned system
c = L @ c_tilde
#c = L @ D_inv @ c_tilde

# Use coefficients on the trial basis functions to recreate the solution in the domain
u = np.zeros((N+1, N+1))
for i in range(1,N):
    u[i] += c[i-1]

plt.plot(trial_space_pts, u)
plt.show()

print("done")