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

M = 13
N = 16

A = np.random.randint(0, 10, size=(M, M))
B = np.random.randint(0, 10, size=(M, M))
C = np.kron(A, B)

A_p = np.pad(A, pad_width=((0, N-M), (0, N-M)), mode='constant', constant_values=0)
B_p = np.pad(B, pad_width=((0, N-M), (0, N-M)), mode='constant', constant_values=0)

C_p = np.kron(A_p, B_p)
print("A_prime: ", A_p)
print("B_prime: ", B_p)
print("C_prime: ", C_p)

P_row = np.zeros((N*N, N*N), dtype=int)
# set ones in P_row (short version) (map from column to row a.k.a. input to output)
for c in range(N*N): # c is the column number
    if c % N < M:
        P_row[c + math.floor(c/N) * (M - N),c] = 1
    elif c % N >= M:
        P_row[c + math.floor(c/N) * (-M) - M + M*M,c] = 1

C_2 = P_row @ C_p @ np.transpose(P_row)
print("C_2 (permuted kronecker product of the zero-padded block encodings:", C_2)
print("C:", C)

print("Is the upper (M^2 x M^2) corner of the permuted C_2 equal to C: ", str(np.allclose(C, C_2[:M*M,:M*M])))