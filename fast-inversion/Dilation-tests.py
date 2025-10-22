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
import itertools

# For both state interpolation and application of the BPX preconditioner,
# we tried to use a strategy of refining the vector/matrix one level at a time
# by applying a "dilator" that does the operation |x>|0> -> 0.5 |2x-1> + 1 |2x> + 0.5 |2x+1>

# However, this dilator needs to be block-encoded, and will have an alpha of about 2,
# which means if these alphas multiplicatively combine, the alpha for the block-encoding for
# the entire dilation process will be 2^L, which I think would be bad because block-encoding error
# sclaes with alpha??? (but I think this is an errorless block-encoding?), but also post-selection can 
# take O(alpha) time so this could also be bad. Hopefully through this
# script writing out how each of these dilators would be block-encoded I can understand if and why
# this method of interpolation/dilation is bad

# dilator matrix that does the transformation E|x> = 0.5|(x-offset) mod N> + 1|x> + 0.5|(x+offset) mod N>
# BE=true returns the entire unitary matrix that would block encode E
def get_E(N, offset = 1, BE=False):
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
        V = np.array([[1/math.sqrt(2), 1/math.sqrt(2), 0, 0],
                      [1/2, -1/2, 0, 1/math.sqrt(2)],
                      [1/2, -1/2, 0, -1/math.sqrt(2)],
                      [0, 0, 1, 0]])
        V_inv = np.transpose(V)
        I_N = np.eye(N)
        matrix = np.eye(N*4)

        # apply the V matrix
        matrix = np.kron(V, I_N) @ matrix

        # apply the 3 summands of the LCU controlled on the ancillas V was applied to
        # identity matrix
        ident_control = np.eye(4*N)
        ident_control[:N,:N] = np.eye(N)
        matrix = ident_control @ matrix

        # positive integer shift matrix
        int_plus = np.diag(np.ones(N-offset), k=-offset) + np.diag(np.ones(offset), k=N-offset)
        int_plus_control = np.eye(4*N)
        int_plus_control[N:2*N,N:2*N] = int_plus
        matrix = int_plus_control @ matrix

        # negative integer shift matrix
        int_minus = np.diag(np.ones(N-offset), k=offset) + np.diag(np.ones(offset), k=offset-N)
        int_minus_control = np.eye(4*N)
        int_minus_control[2*N:3*N,2*N:3*N] = int_minus
        matrix = int_minus_control @ matrix

        # apply the inverse of V
        matrix = np.kron(V_inv, I_N) @ matrix

        return matrix

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

l = 4
L = 9
Nl = int(2**l)
NL = int(2**L)

# create some arbitrary input state represeting the coarse solution
np.random.seed(931986)
input_state = np.random.random(Nl)
input_state[-1] = 0 # last value is zero because actual size of input state is 2**l - 1
input_state = input_state / np.linalg.norm(input_state)

# put input_state on finer grid by just adding qubits to the register initialized to |1>
zero_state = np.zeros(int(2**(L-l)))
zero_state[-1] = 1
input_state_fine = np.kron(input_state, zero_state)

post_selection_probability = 1 # update the probability of post selecting the correct result (without amplitude amplification)

# add 3 ancilla qubits, first 2 for the V ancillas, next one to remove problems coming from the integer shift matrices being periodic
zero_state = np.zeros(8)
zero_state[0] = 1
q_state = np.kron(zero_state, input_state_fine)
for i in range(L-l-1, -1, -1):
    E = get_E(2*NL, offset=int(2**i), BE=True) 
    q_state = E @ q_state # apply the dilation

    # you must post-select |00> for the V ancilla qubits and |0> for the extra qubit in E
    # the probability of this happening is multiplied into post_selection_probability
    # and then q_state is updated to reflect that this postselection happened and the first
    # 3 qubits have to be in the |000> state
    sub_state_norm = np.linalg.norm(q_state[:int(2**L)])
    post_selection_probability *= sub_state_norm**2
    q_state_temp = np.zeros(2**(L+3))
    q_state_temp[:int(2**L)] = q_state[:int(2**L)] / sub_state_norm
    q_state = q_state_temp

print("success probability: ", post_selection_probability)
plt.plot(np.arange(2**(-l), 1, 2**(-l)), input_state[:-1])

plt.plot(np.arange(2**(-L), 1, 2**(-L)), q_state[:int(2**L - 1)])
plt.legend(["coarse", "fine"])
plt.show()