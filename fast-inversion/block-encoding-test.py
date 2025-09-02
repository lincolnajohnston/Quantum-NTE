import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
import math
import cmath

from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation, CXGate, XGate, ZGate, QFT, HGate, RYGate, U1Gate, IntegerComparator, DraperQFTAdder
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_histogram
import fable

# just a test script to review block-encoding

