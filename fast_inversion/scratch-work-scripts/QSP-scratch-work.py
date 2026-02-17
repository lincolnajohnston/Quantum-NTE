import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
from scipy.sparse import csr_matrix, coo_matrix
import scipy as sp
import math
import itertools

Z = np.array([[1, 0],[0,-1]])
phi = 1.5
rz = sp.linalg.expm(phi*1j*Z)
print(rz)