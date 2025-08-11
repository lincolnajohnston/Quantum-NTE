import sys
import os
sys.path.append(os.getcwd())
from helpers.ProblemData import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import math

# return the y value (as fraction of max height) at position x for a triangle wave starting at x_min and ending at x_max
def triangle_wave(x: float, x_min, x_max) -> float:
    if x < x_min or x > x_max:
        return 0
    dx = x_max - x_min
    r = x - x_min
    return 1.0 - 2.0 * abs(r - dx/2) / dx


# the domain goes from 0 to 1
L = 4
n_fine = int(math.pow(2,L))

h_fine = 1/n_fine


F = np.zeros((n_fine-1,n_fine * 2 - L - 2))
fine_x_vals = np.linspace(h_fine,1-h_fine,n_fine-1)
col = 0
for l in range(1,L+1):
    n_coarse = int(math.pow(2,l))
    h_coarse = 1/n_coarse
    #triangle_height_weight = math.pow(2,l-L)
    triangle_height_weight = 1
    for i in range(n_coarse - 1):
        weights = [math.pow(2,-l/2) * triangle_height_weight * triangle_wave(x,h_coarse*i,h_coarse*(i+2)) for x in fine_x_vals]
        F[:,col] =  weights
        col += 1
print("F condition number:", np.linalg.cond(F))
print("F matrix norm: ", np.linalg.norm(F))
print(F)
