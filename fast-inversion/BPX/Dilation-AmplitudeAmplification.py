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
import cmath
import itertools

# Can amplitude amplification make the success probability of N applications of E be only linearly dependent on alpha, not 2^alpha

# dilator matrix that does the transformation E|x> = 0.5|(x-offset) mod N> + 1|x> + 0.5|(x+offset) mod N>
# BE=true returns the entire unitary matrix that would block encode E
# gap_ancillas is the number of ancillas placed between the main register and the LCU ancillas when BE=true
def get_E(N, offset = 1, BE=False, prefix_ancillas = 0, gap_ancillas=0):
    if BE == False:
        matrix = np.zeros((N, N))
        matrix[0:N,0:N] += np.diag(np.ones(N)) # diagonal terms, identity matrix O(1) time to apply

        # |x> -> 0.5|(x-offset) mod N>
        # |x> -> 1|(x-offset) mod N> is unitary and can be implemented in O(polylog(N) time)
        matrix[0:N,0:N] += np.diag(0.5 * np.ones(N-offset), k=offset)  # |x> -> |x-offset> term when x >= offset
        matrix[0:N,0:N] += np.diag(0.5 * np.ones(offset), k=offset-N) # |x> -> |x-offset> term when x < offset

        # |x> -> 0.5|(x+offset) mod N>
        # |x> -> 1|(x+offset) mod N> is unitary and can be implemented in O(polylog(N) time)
        matrix[0:N,0:N] += np.diag(0.5 * np.ones(offset), k=N-offset)  # |x> -> |x+offset> term when x >= N - offset
        matrix[0:N,0:N] += np.diag(0.5 * np.ones(N-offset), k=-offset) # |x> -> |x+offset> term when x < N - offset

        # Each of the three terms can be implemented as separate unitary matrices. Then LCU can be used to sum them, with alpha values of 0.5, 1, and 0.5
        # the time complexity of doing LCU is O(alpha) (which I think comes from the post-selection procedure, which you can do amplitude amplification for 
        # which takes O(alpha) rotations to get a high probability of success).
        # So alpha=2, E/2 can be block-encoded in O(polylog(N) + alpha) = O(polylog(N)) time
        return matrix
    else:
        # Use LCU to block encode
        gap_ancilla_N = int(2**gap_ancillas)
        total_N = gap_ancilla_N*N
        V = np.array([[1/math.sqrt(2), 1/math.sqrt(2), 0, 0],
                      [1/2, -1/2, 0, 1/math.sqrt(2)],
                      [1/2, -1/2, 0, -1/math.sqrt(2)],
                      [0, 0, 1, 0]])
        V_inv = np.transpose(V)
        I_N = np.eye(total_N)
        E_matrix = np.eye(4*total_N)

        # apply the V matrix
        E_matrix = np.kron(V, I_N) @ E_matrix

        # apply the 3 summands of the LCU controlled on the ancillas V was applied to
        # identity matrix
        ident_control = np.eye(4*total_N)
        ident_control[:total_N,:total_N] = np.eye(total_N)
        E_matrix = ident_control @ E_matrix

        # positive integer shift matrix
        int_plus = np.diag(np.ones(N-offset), k=-offset) + np.diag(np.ones(offset), k=N-offset)
        int_plus = np.kron(np.eye(gap_ancilla_N), int_plus)
        int_plus_control = np.eye(4*total_N)
        int_plus_control[total_N:2*total_N,total_N:2*total_N] = int_plus
        E_matrix = int_plus_control @ E_matrix

        # negative integer shift matrix
        int_minus = np.diag(np.ones(N-offset), k=offset) + np.diag(np.ones(offset), k=offset-N)
        int_minus = np.kron(np.eye(gap_ancilla_N), int_minus)
        int_minus_control = np.eye(4*total_N)
        int_minus_control[2*total_N:3*total_N,2*total_N:3*total_N] = int_minus
        E_matrix = int_minus_control @ E_matrix

        # apply the inverse of V
        E_matrix = np.kron(V_inv, I_N) @ E_matrix

        return np.kron(np.eye(int(2**prefix_ancillas)), E_matrix)

# returns the F_{u,s} matrix, which only has the ones of section s of the F matrix
def get_F_us_1D(L, s):
    section_length_list = [int(math.pow(2,n_p)-1) for n_p in range(1,L+1)]
    N = int(math.pow(2,L))
    Fu = np.zeros((N,N))
    row_jump = 2**(L-s-1)
    row_offset = row_jump - 1
    section_size = section_length_list[s]
    for col in range(section_size):
        Fu[row_offset + col*row_jump, col] = 1
    return Fu

L = 4 # total levels
'''l = 3 # current level section
Nl = int(2**l)
NL = int(2**L)'''

# single E matrix
'''g = 1 # offset for the E matrix
n = L+1 # number of qubits E is applied to
n_total = n+2 # total number of qubits in circuit including ancillas
E = get_E(2**n, offset=g, BE=True)'''


# multiple E matrices applied in series
offset_max = 3
offsets = [int(2**i) for i in range(offset_max)]
n = L+1 # number of qubits E is applied to
n_total = n+2*offset_max # total number of qubits in circuit including ancillas
for i in range(offset_max):
    g = offsets[i]
    E_g = get_E(2**n, offset=g, BE=True, prefix_ancillas = 2*(offset_max - i - 1), gap_ancillas=2*i)
    E = E @ E_g if i>0 else E_g

E_inv = np.transpose(E)
# add 3 ancilla qubits, first 2 for the V ancillas, next one to remove problems coming from the integer shift matrices being periodic
circuit_unitary = np.eye(int(2**(n_total)))
circuit_unitary = E @ circuit_unitary

# make S_chi matrix that is applied to the 3 ancillas from the LCU (2 qubits) + extra 1 qubits to remove the periodic values
phi = 1.00*math.pi
post_select_bits = 2*offset_max + 1
S_chi = np.eye(int(2**post_select_bits), dtype=np.complex_)
S_chi[0,0] = cmath.exp(1j * phi)
S_chi = np.kron(S_chi, np.eye(int(2**L)))

# make S_0 matrix that is applied to the 3 ancillas from the LCU (2 qubits) + extra 1 qubits to remove the periodic values
S_0 = np.eye(int(2**n_total), dtype=np.complex_)
input_state = 0
S_0[input_state,input_state] = cmath.exp(1j * phi)
#S_0 = np.kron(S_0, np.eye(int(2**L)))

good_norm = np.linalg.norm(circuit_unitary[:int(2**L),input_state])
good_angle = math.asin(good_norm)
print("Norm of \"good\" state: ", good_norm)
print("Angle of good state: ", good_angle)

# Grover iteration
n_G = 5 # number of Grover iterations
for i in range(n_G):
    # Grover iteration using premade rotations
    '''P = np.zeros((int(2**3), int(2**3)))
    P[0,0] = 1
    P = np.kron(P,np.eye(int(2**L)))
    R_g = np.eye(int(2**n_total)) - 2*P

    R_psi = np.eye(int(2**n_total)) - 2*(E @ P @ E_inv)
    circuit_unitary = R_g @ circuit_unitary
    circuit_unitary = R_psi @ circuit_unitary'''

    # Grover iteration using original block encoding
    circuit_unitary = S_chi @ circuit_unitary
    circuit_unitary = E_inv @ circuit_unitary
    circuit_unitary = S_0 @ circuit_unitary
    circuit_unitary = E @ circuit_unitary

    good_norm = np.linalg.norm(circuit_unitary[:int(2**L),input_state])
    good_angle = math.asin(good_norm)
    print("Norm of \"good\" state: ", good_norm)
    print("Angle of good state: ", good_angle)

    print("end of Grover iteration ", str(i))
