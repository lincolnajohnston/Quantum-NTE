import numpy as np
import os
from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian
import time
import ProblemData
import LcuFunctions
import math


from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
np.set_printoptions(threshold=np.inf)

def getFluxAtPosition(pos, g, sol, n_x, n_y, delta_x, delta_y):
    x = pos[0]
    y = pos[1]
    x0 = (1/2-n_x/2) * delta_x
    y0 = (1/2-n_y/2) * delta_y
    x_index = math.floor((x-x0) / delta_x)
    y_index = math.floor((y-y0) / delta_y)
    xm = (x-x0) % delta_x
    ym = (y-y0) % delta_y

    flux_lower_left = sol[g,x_index,y_index]
    flux_upper_left = sol[g,x_index,y_index+1]
    flux_lower_right = sol[g,x_index+1,y_index]
    flux_upper_right = sol[g,x_index+1,y_index+1]

    flux_left = flux_lower_left + (flux_upper_left - flux_lower_left) * ym/delta_y
    flux_right = flux_lower_right + (flux_upper_right - flux_lower_right) * ym/delta_y

    return flux_left + (flux_right - flux_left) * xm/delta_x

def getPositionFromIndex(i, j, n_x, n_y, delta_x, delta_y):
    return [(1/2-n_x/2 + i) * delta_x, (1/2-n_y/2 + j) * delta_y]


start = time.perf_counter()

sim_path = 'simulations/test_1G_diffusion/input_files/'
ndim = 2

# Run the most fine simulation which will be treated as the true solution
ref_n = 128
ref_input_file = 'input'+str(ref_n)+'.txt'
# create ProblemData object, select geometry type
ref_data = ProblemData.ProblemData(sim_path + ref_input_file, 'homo')

# make A matrix and b vector
if ref_data.sim_method == "sp3":
    A_mat_size = 2 * (ref_data.n_x) * (ref_data.n_y) * ref_data.G
    A_matrix, b_vector = ref_data.sp3_construct_A_matrix(A_mat_size) 
elif ref_data.sim_method == "diffusion":
    A_mat_size = (ref_data.n_x) * (ref_data.n_y) * ref_data.G
    if ndim == 1:
        A_matrix, b_vector = ref_data.diffusion_construct_FD_A_matrix_1D(A_mat_size)
    elif ndim == 2:
        A_matrix, b_vector = ref_data.diffusion_construct_FD_A_matrix(A_mat_size)

ref_sol_vec = np.linalg.solve(A_matrix, b_vector)

ref_sol_vec = ref_sol_vec[:int(ref_data.G * ref_data.n_x * ref_data.n_y)]
ref_sol_vec.resize((ref_data.G, ref_data.n_x,ref_data.n_y))
xticks = np.round(np.array(range(ref_data.n_x))*ref_data.delta_x - (ref_data.n_x - 1)*ref_data.delta_x/2,3)
yticks = np.round(np.array(range(ref_data.n_y))*ref_data.delta_y - (ref_data.n_y - 1)*ref_data.delta_y/2,3)
for g in range(ref_data.G):
    ax = sns.heatmap(ref_sol_vec[g,:,:], linewidth=0.5, xticklabels=xticks, yticklabels=yticks)
    ax.invert_yaxis()
    plt.title("Reference Real Solution, Group " + str(g))
    #plt.savefig('real_sol_g' + str(g) + '.png')
    plt.figure()

#n_vals = [40,45,50,55,60,64,70,80,90,100,110,120,128,140]
n_vals = [40,45,50,55,60,64,70,80,90,100]

# do the most finely discretized solution first

input_files = ['input'+str(n_vals[i])+'.txt' for i in range(len(n_vals))]
integrals = np.zeros((len(input_files), 10))
l_inf = np.zeros(len(n_vals))
l_2 = np.zeros(len(n_vals))
for sim_index,input_file in enumerate(input_files):
    print("Simulating Input file ", input_file)

    # create ProblemData object, select geometry type
    data = ProblemData.ProblemData(sim_path + input_file, 'homo')

    # make A matrix and b vector
    if data.sim_method == "sp3":
        A_mat_size = 2 * (data.n_x) * (data.n_y) * data.G
        A_matrix, b_vector = data.sp3_construct_A_matrix(A_mat_size) 
    elif data.sim_method == "diffusion":
        A_mat_size = (data.n_x) * (data.n_y) * data.G
        if ndim == 1:
            A_matrix, b_vector = data.diffusion_construct_FD_A_matrix_1D(A_mat_size)
        elif ndim == 2:
            A_matrix, b_vector = data.diffusion_construct_FD_A_matrix(A_mat_size)

    classical_sol_vec = np.linalg.solve(A_matrix, b_vector)

    classical_sol_vec = classical_sol_vec[:int(data.G * data.n_x * data.n_y)]
    classical_sol_vec.resize((data.G, data.n_x,data.n_y))

    #classical_RR = classical_sol_vec * data.delta_x * data.delta_y

    classical_sol_error = np.zeros(classical_sol_vec.shape)
    for g in range(data.G):
        for i in range(data.n_x):
            for j in range(data.n_y):
                classical_sol_error[g,i,j] = classical_sol_vec[g,i,j] - getFluxAtPosition(getPositionFromIndex(i, j, data.n_x, data.n_y, data.delta_x, data.delta_y), g, ref_sol_vec, ref_data.n_x, ref_data.n_y, ref_data.delta_x, ref_data.delta_y)
    l_inf[sim_index] = np.max(np.abs(classical_sol_error))
    l_2[sim_index] = np.linalg.norm(classical_sol_error) / math.sqrt(data.G * data.n_x * data.n_y)
    for i in range(sim_index,int(math.log2(data.n_x)) + 1):
        integrals[sim_index, i-sim_index] = np.sum(classical_sol_vec[:,:2**i,:2**i]) / (2**(i*ndim))

    xticks = np.round(np.array(range(data.n_x))*data.delta_x - (data.n_x - 1)*data.delta_x/2,3)
    yticks = np.round(np.array(range(data.n_y))*data.delta_y - (data.n_y - 1)*data.delta_y/2,3)
    '''for g in range(data.G):
        ax = sns.heatmap(classical_sol_vec[g,:,:], linewidth=0.5, xticklabels=xticks, yticklabels=yticks)
        ax.invert_yaxis()
        plt.title("Classical Solution, Group " + str(g))
        #plt.savefig('real_sol_g' + str(g) + '.png')
        plt.figure()
    for g in range(data.G):
        ax = sns.heatmap(classical_sol_error[g,:,:], linewidth=0.5, xticklabels=xticks, yticklabels=yticks)
        ax.invert_yaxis()
        plt.title("Classical Solution Error, Group " + str(g))
        #plt.savefig('real_sol_g' + str(g) + '.png')
        plt.figure()'''
print(integrals)

'''for i in range(10):
    if(integrals[-1,i] == 0):
        break
    plt.plot(n_vals, np.abs(integrals[:,i]-integrals[-1,i]))
    plt.xlabel("n_x")
    plt.ylabel("Error of integral over some region")
    plt.legend(["integral width: "+str(n_vals[i]/n_vals[-1])+" of domain" for i in range(len(n_vals))])
plt.figure()
for i in range(10):
    if(integrals[-1,i] == 0):
        break
    plt.plot(1/np.array(n_vals), np.abs(integrals[:,i]-integrals[-1,i]))
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("Error of integral over some region")
    plt.legend(["integral width: "+str(n_vals[i]/n_vals[-1])+" of domain" for i in range(len(n_vals))])'''
for i in range(len(n_vals)):
    plt.plot(n_vals, l_inf)
    plt.xlabel("n_x")
    plt.ylabel("L_inf error")
plt.figure()
for i in range(len(n_vals)):
    plt.plot(1/np.array(n_vals), l_inf)
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("L_inf error")
plt.figure()
for i in range(len(n_vals)):
    plt.plot(n_vals, l_2)
    plt.xlabel("n_x")
    plt.ylabel("L_2 error")
plt.figure()
for i in range(len(n_vals)):
    plt.plot(1/np.array(n_vals), l_2)
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("L_2 error")
plt.show()