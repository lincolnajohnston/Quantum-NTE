import sys
import os
sys.path.append(os.getcwd())
from helpers.ProblemData import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import math

# Use the decomposition of the A matrix (piecewise constant diffution coefficient with harmonic mean at material boundaries) to apply A in O(logN) time
# which can then be inverted quantumly