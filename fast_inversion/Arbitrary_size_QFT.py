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

N = 10
U_shift = np.zeros((N,N))
for i in range(N):
    U_shift[(i+1) % N, i] = 1

U_shift_eigvals, U_shift_eigvecs = np.linalg.eig(U_shift)
U_shift_eig_phases = np.copy(U_shift_eigvals)
for i in range(N):
    U_shift_eig_phases[i] = cmath.phase(U_shift_eigvals[i]) / math.pi
    if U_shift_eig_phases[i] < 0:
        U_shift_eig_phases[i] = U_shift_eig_phases[i] + 2

sorted_eigvals = np.array([x for _, x in sorted(zip(U_shift_eig_phases, U_shift_eigvals))])
sorted_eigvecs = np.array([x for _, x in sorted(zip(U_shift_eig_phases, np.transpose(U_shift_eigvecs)))])

# adjust eigenvectors by a phase
for i in range(N):
    sorted_eigvecs[i,:] *= -1 * np.conjugate(sorted_eigvecs[i,0]) / np.abs(sorted_eigvecs[i,0])

print(U_shift)
print("\n\n")
print(np.round(np.transpose(U_shift_eigvecs),3))