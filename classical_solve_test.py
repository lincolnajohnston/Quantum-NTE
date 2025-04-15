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
import scipy.sparse as sp


from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
np.set_printoptions(threshold=np.inf)

# python script description: Classically solve the linear system created by ProblemData discretizing
# a diffusion of SP3 problem. Vary the N (number of discretization units, assumed to be finite volume for now) and find the L2 norm of the
# error in the solution created (treating a very finely discretized solution as the true solution)
# This shows how the error introduced from  discretization scales with N and delta x

# find the solution at a continuous point, pos, on a discretized grid with solution given
# by "sol". If interpolate=True, interpolate between discrete grid values, otherwise, assume
# "sol" is constant within each finite volume
def getFluxAtPosition(pos, g, sol, n_x, n_y, delta_x, delta_y, interpolate=False):
    x = pos[0]
    y = pos[1]
    x0 = (1/2-n_x/2) * delta_x
    y0 = (1/2-n_y/2) * delta_y
    x_index = math.floor((x-x0) / delta_x)
    y_index = math.floor((y-y0) / delta_y)
    xm = (x-x0) % delta_x
    ym = (y-y0) % delta_y

    if not interpolate:
        return sol[g,x_index,y_index]

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

# input variables
#sim_path = 'simulations/test_1G_diffusion/input_files/'
ndim = 2
L_x = 3.2
L_y = 3.2
geometry_type = "homo"
sim_method = "diffusion"
G = 1
xs_folder = "XS-1group-constD"

# set the values of discretization level, N, that will be used for creating ProblemData objects and then solved as a linear system
#n_vals = [40,41,42,43,44,45,46,47,48,49,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,144,146,148,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,490,510,530,550,600]
#n_vals = [40,41,42,43,44,45,46,47,48,49,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,144,146,148,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,260,270,280,290,300,310,320,330,340,350,370,400]
#n_vals = [20,25,30,35,40,45,50,55,60,64,70,80,90,100,110,120,130,150,170,190,210,230,250]
n_vals = list(range(20,200,5))
n_vals = [10,40,45,50,55,60,64,70,80,90,100]
n_vals = np.array(n_vals)

# Run the most fine simulation which will be treated as the true solution
ref_n = 200
ref_input_file = 'input'+str(ref_n)+'.txt'
# create ProblemData object, select geometry type
ref_data = ProblemData.ProblemData(geom_string = geometry_type, input_data = {"n_x":ref_n, "n_y":ref_n, "delta_x":L_x/ref_n, "delta_y":L_y/ref_n, "sim_method":sim_method, "G":G, "xs_folder":xs_folder})

# make A matrix and b vector fro reference solution
if ref_data.sim_method == "sp3":
    A_mat_size = 2 * (ref_data.n_x) * (ref_data.n_y) * ref_data.G
    A_matrix, b_vector = ref_data.sp3_construct_A_matrix(A_mat_size) 
elif ref_data.sim_method == "diffusion":
    A_mat_size = (ref_data.n_x) * (ref_data.n_y) * ref_data.G
    if ndim == 1:
        A_matrix, b_vector = ref_data.diffusion_construct_FD_A_matrix_1D(A_mat_size)
    elif ndim == 2:
        A_matrix, b_vector = ref_data.diffusion_construct_sparse_FD_A_matrix(A_mat_size)
A_matrix_construction_time = time.perf_counter()
print("made A matrix in " + str(A_matrix_construction_time - start) + " sec")

# solve the reference solution system of equations
ref_sol_vec = sp.linalg.spsolve(A_matrix, b_vector)
A_matrix_construction_time = time.perf_counter()
print("solved A matrix in " + str(A_matrix_construction_time - start) + " sec")
ref_sol_vec = ref_sol_vec[:int(ref_data.G * ref_data.n_x * ref_data.n_y)]
ref_sol_vec.resize((ref_data.G, ref_data.n_x,ref_data.n_y))

# plot the reference flux solution
'''xticks = np.round(np.array(range(ref_data.n_x))*ref_data.delta_x - (ref_data.n_x - 1)*ref_data.delta_x/2,3)
yticks = np.round(np.array(range(ref_data.n_y))*ref_data.delta_y - (ref_data.n_y - 1)*ref_data.delta_y/2,3)
for g in range(ref_data.G):
    ax = sns.heatmap(ref_sol_vec[g,:,:], linewidth=0.5, xticklabels=xticks, yticklabels=yticks)
    ax.invert_yaxis()
    plt.title("Reference Real Solution, Group " + str(g))
    #plt.savefig('real_sol_g' + str(g) + '.png')
    plt.figure()'''

#input_files = ['input'+str(n_vals[i])+'.txt' for i in range(len(n_vals))]
l_inf = np.zeros(len(n_vals))
l_2 = np.zeros(len(n_vals))
for n_index,n in enumerate(n_vals):
    print("Simulating n =  ", n)

    # create ProblemData object, select geometry type
    data = ProblemData.ProblemData(geom_string=geometry_type, input_data = {"n_x":n, "n_y":n, "delta_x":L_x/n, "delta_y":L_y/n, "sim_method":sim_method, "G":G, "xs_folder":xs_folder})

    # make A matrix and b vector
    if data.sim_method == "sp3":
        A_mat_size = 2 * (data.n_x) * (data.n_y) * data.G
        A_matrix, b_vector = data.sp3_construct_A_matrix(A_mat_size) 
    elif data.sim_method == "diffusion":
        A_mat_size = (data.n_x) * (data.n_y) * data.G
        if ndim == 1:
            A_matrix, b_vector = data.diffusion_construct_FD_A_matrix_1D(A_mat_size)
        elif ndim == 2:
            A_matrix, b_vector = data.diffusion_construct_sparse_FD_A_matrix(A_mat_size)

    classical_sol_vec = sp.linalg.spsolve(A_matrix, b_vector)

    classical_sol_vec = classical_sol_vec[:int(data.G * data.n_x * data.n_y)]
    classical_sol_vec.resize((data.G, data.n_x,data.n_y))

    # solution error on the coarse grid
    '''classical_sol_error = np.zeros(classical_sol_vec.shape)
    for g in range(data.G):
        for i in range(data.n_x):
            for j in range(data.n_y):
                classical_sol_error[g,i,j] = classical_sol_vec[g,i,j] - getFluxAtPosition(getPositionFromIndex(i, j, data.n_x, data.n_y, data.delta_x, data.delta_y), g, ref_sol_vec, ref_data.n_x, ref_data.n_y, ref_data.delta_x, ref_data.delta_y)'''
    
    # solution error on the fine grid
    classical_sol_error = np.zeros(ref_sol_vec.shape)
    for g in range(ref_data.G):
        for i in range(ref_data.n_x):
            for j in range(ref_data.n_y):
                classical_sol_error[g,i,j] = getFluxAtPosition(getPositionFromIndex(i, j, ref_data.n_x, ref_data.n_y, ref_data.delta_x, ref_data.delta_y), g, classical_sol_vec, data.n_x, data.n_y, data.delta_x, data.delta_y) - ref_sol_vec[g,i,j]
    l_inf[n_index] = np.max(np.abs(classical_sol_error))
    l_2[n_index] = np.linalg.norm(classical_sol_error) / math.sqrt(data.G * data.n_x * data.n_y)

    # plot the coarse grid flux solution and its error compared to the fine grid
    '''xticks = np.round(np.array(range(data.n_x))*data.delta_x - (data.n_x - 1)*data.delta_x/2,3)
    yticks = np.round(np.array(range(data.n_y))*data.delta_y - (data.n_y - 1)*data.delta_y/2,3)
    for g in range(data.G):
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

# plot the error in the coarse grids
print("n vals: ", n_vals)
print("l_inf: ", l_inf)
print("l_2: ", l_2)

plt.plot(n_vals, l_inf)
plt.xlabel("n_x")
plt.ylabel("L_inf error")
plt.figure()

plt.plot(2*L_x/np.array(n_vals), l_inf)
plt.xlabel(r'$\Delta x$')
plt.ylabel("L_inf error")
plt.figure()

plt.plot(n_vals, l_2)
plt.xlabel("n_x")
plt.ylabel("L_2 error")
plt.figure()

plt.plot(2*L_x/np.array(n_vals), l_2)
plt.xlabel(r'$\Delta x$')
plt.ylabel("L_2 error")
#coef = (2*L_x/n_vals[0])**2 / l_2[0]
#plt.plot(2*L_x/n_vals,(2*L_x/n_vals)**2 / coef) # plot a (delta-x)^2 curve that assumes L_x = L_y

plt.show()