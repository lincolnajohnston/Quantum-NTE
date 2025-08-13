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
    L = np.zeros((N_max-1,N_max-1))

    s = 0
    j = 1
    ref = 0
    for i in range(n_max-n_min):
        N_cur = int(N_max * math.pow(2,-i-1))
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

    for i in range(n_max-n_min, n_max):
        N_cur = int(N_max * math.pow(2,-i-1))

        # columns [1,N-2]
        for l in range(0,N_cur):
            L[s+j*(2*l),ref+l] = 1

        s += j
        j *= 2
        ref += N_cur

    return L


# (NxN) is size of D_inv matrix
# n is the number of indices for sets of wavelet basis matrices
def getD_inv(N, n):
    D_inv = np.zeros((N,N))
    m = 0
    for i in range(n-1,-1,-1):
        for j in range(int(math.pow(2,i))):
            D_inv[m,m] = math.pow(2,-i)
            m += 1
    return D_inv

# (NxN) is size of D_inv matrix
# n is the number of indices for sets of wavelet basis matrices
'''def getD_inv(N, n_min, n_max):
    D_inv = np.zeros((N,N))
    for n in range(n_min, n_max + 1):
        jump = int(math.pow(2,n_max - n + 1))
        start = int(math.pow(2,n_max - n)) - 1
        for j in range(start,N,jump):
            D_inv[j,j] = jump / math.pow(2,n_max - n_min)
    return D_inv'''

a = 0
b = 1

n = 5 # number of qubits to represent the number of FV regions
N = int(math.pow(2,n)) # max index of points defining domain
N_A = N-1 # number of points excluding boundaries (size of the A matrix)
dx = (b - a) / (N)

f = 1
f_n = f * dx * np.ones(N_A)

trial_space_pts = np.linspace(a,b,N+1)
test_space_pts = [a] + list(np.linspace(a+dx/2,b-dx/2,N)) + [b]
print(trial_space_pts)
print(test_space_pts)

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

# do the wavelet preconditioning
#L = getL(int((N_A-1)/2+1))
L = getL(1,n)

D_inv = getD_inv(N_A,n)
#D_inv = getD_inv(N_A, 1, n)

A_n_tilde = D_inv @ np.transpose(L) @ A_n @ L @ D_inv
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