import numpy as np
import math
import os
from Material import Material
import scipy.sparse as sp



class ProblemData:
    def __init__(self, input_file="", geom_string='4_pin', input_data={}):
        if input_file == "":
            self.n_x = input_data["n_x"]
            self.n_pts_x = self.n_x + 2
            self.n_y = input_data["n_y"]
            self.n_pts_y = self.n_y + 2
            self.delta_x = input_data["delta_x"]
            self.delta_y = input_data["delta_y"]
            self.sim_method = input_data["sim_method"]
            self.G = input_data["G"]
            self.xs_folder = input_data["xs_folder"]
        else:
            self.read_input(input_file)
        self.initialize_BC()
        self.initialize_geometry(geom_string)
        self.initialize_materials()

    def read_input(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        self.n_x = int(lines[0].strip())
        self.n_pts_x = self.n_x + 2
        self.n_y = int(lines[1].strip())
        self.n_pts_y = self.n_y + 2
        self.delta_x = float(lines[2].strip())
        self.delta_y = float(lines[3].strip())
        self.sim_method = lines[4].strip()
        self.G = int(lines[5].strip())
        self.xs_folder = lines[6].strip()
    
    def initialize_BC(self):
        if self.sim_method == "sp3":
            top_I_3 = 0
            bottom_I_3 = 0
            right_I_3 = 0
            left_I_3 = 0
            self.top_x_I_3 = np.ones(self.n_pts_x) * top_I_3
            self.bottom_x_I_3 = np.zeros(self.n_pts_x) * bottom_I_3
            self.right_y_I_3 = np.zeros(self.n_pts_y) * right_I_3
            self.left_y_I_3 = np.zeros(self.n_pts_y) * left_I_3

            top_I_1 = 0
            bottom_I_1 = 0
            right_I_1 = 0
            left_I_1 = 0
            self.top_x_I_1 = np.ones(self.n_pts_x) * top_I_1
            self.bottom_x_I_1 = np.zeros(self.n_pts_x) * bottom_I_1
            self.right_y_I_1 = np.zeros(self.n_pts_y) * right_I_1
            self.left_y_I_1 = np.zeros(self.n_pts_y) * left_I_1
        elif self.sim_method == "diffusion":
            self.beta = 0.5 # related to albedo constant

    # return B.C.s at edge of problem domain
    def get_I_1_value(self, index):
        i = index[0]
        j = index[1]
        if (i == 0):
            return self.left_y_I_1[j]
        if (i == self.n_x-1):
            return self.right_y_I_1[j]
        if (j == 0):
            return self.bottom_x_I_1[i]
        if (j == self.n_y-1):
            return self.top_x_I_1[i]
        raise Exception("tried to get BC on non-boundary node")
    
    # return B.C.s at edge of problem domain
    def get_I_3_value(self, index):
        i = index[0]
        j = index[1]
        if (i == 0):
            return self.left_y_I_3[j]
        if (i == self.n_x-1):
            return self.right_y_I_3[j]
        if (j == 0):
            return self.bottom_x_I_3[i]
        if (j == self.n_y-1):
            return self.top_x_I_3[i]
        raise Exception("tried to get BC on non-boundary node")

    # get averaged diffusion coefficient in either the "x" or "y" direction for interior points
    # lower_index is lower index in the direction of the averaged diffusion coefficient
    # set_index is the other dimension index
    def get_av_D(self, direction, lower_index, set_index, g):
        if direction == "x":
            D_lower = self.materials[self.material_matrix[lower_index,set_index]].D[g]
            D_upper = self.materials[self.material_matrix[lower_index+1,set_index]].D[g]
            delta = self.delta_x
        elif direction == "y":
            D_lower = self.materials[self.material_matrix[set_index,lower_index]].D[g]
            D_upper = self.materials[self.material_matrix[set_index,lower_index+1]].D[g]
            delta = self.delta_y
        return 2 * (D_lower/delta) * (D_upper/delta) / (D_lower/delta + D_upper/delta)


    def get_av_D2(self, direction, lower_index, set_index, g):
        if direction == "x":
            D_lower = self.materials[self.material_matrix[lower_index,set_index]].D2[g]
            D_upper = self.materials[self.material_matrix[lower_index+1,set_index]].D2[g]
            delta = self.delta_x
        elif direction == "y":
            D_lower = self.materials[self.material_matrix[set_index,lower_index]].D2[g]
            D_upper = self.materials[self.material_matrix[set_index,lower_index+1]].D2[g]
            delta = self.delta_y
        return 2 * (D_lower/delta) * (D_upper/delta) / (D_lower/delta + D_upper/delta)


    def get_edge_D(self, x_i, y_i, g, delta):
        D = self.materials[self.material_matrix[x_i,y_i]].D[g]
        return 2 * (self.beta/2) * (D/delta) / (self.beta/2 + (D/delta))
        # Set material data at each finite difference point, O(N)
    

    def initialize_geometry(self, geom_string):
        self.material_matrix = np.empty((self.n_x, self.n_y), dtype=object)
        x_range = self.n_x * self.delta_x
        y_range = self.n_y * self.delta_y

        fuel_radius = min(x_range,y_range)/4

        for i in range(self.n_x):
            for j in range(self.n_y):
                x_val = (i + 0.5) * self.delta_x - x_range/2
                y_val = (j + 0.5) * self.delta_y - y_range/2

                # homogeneous fuel
                if geom_string == 'homo':
                    self.material_matrix[i,j] = "fuel"
                
                # fuel at center
                elif geom_string == '1_pin':
                    if (math.sqrt(x_val * x_val + y_val * y_val) < fuel_radius):
                        # use fuel XSs
                        self.material_matrix[i,j] = "fuel"
                        
                    else:
                        # use moderator XSs
                        self.material_matrix[i,j] = "water"

                elif geom_string == '4_pin':
                    # 4 fuel pins
                    if (math.sqrt(math.pow(abs(x_val)-x_range/4,2) + math.pow(abs(y_val)-y_range/4,2)) < fuel_radius):
                        # use fuel XSs
                        self.material_matrix[i,j] = "fuel"
                    else:
                        # use moderator XSs
                        self.material_matrix[i,j] = "water"
        return

    def initialize_materials(self):
        self.materials = {}
        for _, dirs, _ in os.walk(self.xs_folder):
            for mat in dirs:
                xs_file = self.xs_folder + "/" + mat + "/xs.txt"
                self.materials[mat] = Material(mat, xs_file, self.G, self.n_x, self.n_y)
                
    # use finite volume method to construct the A matrix representing the diffusion equation in the form Ax=b, O(N)
    def diffusion_construct_A_matrix(self, A_mat_size):
        fd_order = 2
        A_matrix = np.zeros((A_mat_size, A_mat_size))
        b_vector = np.zeros(A_mat_size)
        for g in range(self.G):
            for x_i in range(self.n_x):
                for y_i in range(self.n_y):
                    mat = self.materials[self.material_matrix[x_i,y_i]]
                    i = self.unroll_index([g, x_i, y_i])
                    if(x_i == 0): # left BC, normal vector = (-1,0)
                        J_x_minus = self.get_edge_D(x_i ,y_i, g, self.delta_x) * self.delta_y
                    else:
                        J_x_minus = self.get_av_D("x",x_i-1,y_i, g) * self.delta_y
                        A_matrix[i,self.unroll_index([g, x_i-1, y_i])] =  -J_x_minus # (i-1,j) terms
                    if(x_i == self.n_x - 1): # right BC, normal vector = (1,0)
                        J_x_plus = self.get_edge_D(x_i ,y_i, g, self.delta_x) * self.delta_y
                    else:
                        J_x_plus = self.get_av_D("x",x_i,y_i, g) * self.delta_y
                        A_matrix[i,self.unroll_index([g, x_i+1, y_i])] =  -J_x_plus # (i+1,j) terms
                    if(y_i == 0): # bottom BC, normal vector = (0,-1)
                        J_y_minus = self.get_edge_D(x_i ,y_i, g,self.delta_y)* self.delta_x
                    else:
                        J_y_minus = self.get_av_D("y",y_i-1,x_i, g) * self.delta_x
                        A_matrix[i,self.unroll_index([g, x_i, y_i-1])] =  -J_y_minus # (i,j-1) terms
                    if(y_i == self.n_y - 1): # right BC, normal vector = (0,1)
                        J_y_plus = self.get_edge_D(x_i ,y_i, g,self.delta_y) * self.delta_x
                    else:
                        J_y_plus = self.get_av_D("y",y_i,x_i, g) * self.delta_x
                        A_matrix[i,self.unroll_index([g, x_i, y_i+1])] =  -J_y_plus # (i,j+1) terms
                    A_matrix[i,i] = J_x_minus + J_x_plus + J_y_minus + J_y_plus + (mat.sigma_t[g]) * self.delta_x * self.delta_y
                    for g_p in range(self.G): # group to group scattering and fission terms
                        A_matrix[i,self.unroll_index([g_p, x_i, y_i])] += -(mat.sigma_sgg[g_p, g] + mat.chi[g] * mat.nu_sigma_f[g_p]) * self.delta_x * self.delta_y
                    b_vector[i] = mat.Q[g] * self.delta_x * self.delta_y
        #for row in range(len(A_matrix)):
        #    print("row: ", A_matrix[row])
        return A_matrix, b_vector

    # use finite difference method to construct the A matrix representing the diffusion equation in the form Ax=b, assume 0 flux at boundary conditions for simplicity
    def diffusion_construct_FD_A_matrix(self, A_mat_size):
        fd_order = 2
        A_matrix = np.zeros((A_mat_size, A_mat_size))
        b_vector = np.zeros(A_mat_size)
        for g in range(self.G):
            for x_i in range(self.n_x):
                for y_i in range(self.n_y):
                    mat = self.materials[self.material_matrix[x_i,y_i]]
                    i = self.unroll_index([g, x_i, y_i])
                    
                    # combination of double derivative term on current element and total cross section term
                    A_matrix[i,i] += 2 * mat.D / self.delta_x**2 + 2 * mat.D / self.delta_y**2 + mat.sigma_t[g]
                    
                    # double derivative term in x and y direction with periodic boundary conditions
                    A_matrix[i, self.unroll_index([g, (x_i-1)%self.n_x, y_i])] += -mat.D / self.delta_x**2
                    A_matrix[i, self.unroll_index([g, (x_i+1)%self.n_x, y_i])] += -mat.D / self.delta_x**2

                    # double derivative term in y direction
                    A_matrix[i, self.unroll_index([g, x_i, (y_i-1)%self.n_y])] += -mat.D / self.delta_y**2
                    A_matrix[i, self.unroll_index([g, x_i, (y_i+1)%self.n_y])] += -mat.D / self.delta_y**2

                    # double derivative term in x direction
                    '''if(x_i > 0):
                        A_matrix[i, self.unroll_index([g, x_i-1, y_i])] += -mat.D / self.delta_x**2
                    if(x_i < self.n_x - 1):
                        A_matrix[i, self.unroll_index([g, x_i+1, y_i])] += -mat.D / self.delta_x**2

                    # double derivative term in y direction
                    if(y_i > 0):
                        A_matrix[i, self.unroll_index([g, x_i, y_i-1])] += -mat.D / self.delta_y**2
                    if(y_i < self.n_y - 1):
                        A_matrix[i, self.unroll_index([g, x_i, y_i+1])] += -mat.D / self.delta_y**2'''

                    for g_p in range(self.G): # group to group scattering and fission terms
                        A_matrix[i,self.unroll_index([g_p, x_i, y_i])] += -(mat.sigma_sgg[g_p, g] + mat.chi[g] * mat.nu_sigma_f[g_p])
                    b_vector[i] = mat.Q[g]
        #for row in range(len(A_matrix)):
        #    print("row: ", A_matrix[row])
        return A_matrix, b_vector

    # use finite difference method to construct the A matrix representing the diffusion equation in the form Ax=b, assume 0 flux at boundary conditions for simplicity
    def diffusion_construct_sparse_FD_A_matrix(self, A_mat_size):
        #A_matrix = np.zeros((A_mat_size, A_mat_size))
        row = []
        col = []
        data = []
        b_vector = np.zeros(A_mat_size)
        for g in range(self.G):
            for x_i in range(self.n_x):
                for y_i in range(self.n_y):
                    mat = self.materials[self.material_matrix[x_i,y_i]]
                    i = self.unroll_index([g, x_i, y_i])
                    
                    # combination of double derivative term on current element and total cross section term
                    row.append(i)
                    col.append(i)
                    data.append(2 * mat.D[g] / self.delta_x**2 + 2 * mat.D[g] / self.delta_y**2 + mat.sigma_t[g])
                    
                    # double derivative term in x and y direction with periodic boundary conditions
                    '''row.append(i)
                    col.append(self.unroll_index([g, (x_i-1)%self.n_x, y_i]))
                    data.append(-mat.D[g] / self.delta_x**2)

                    row.append(i)
                    col.append(self.unroll_index([g, (x_i+1)%self.n_x, y_i]))
                    data.append(-mat.D[g] / self.delta_x**2)

                    # double derivative term in y direction
                    row.append(i)
                    col.append(self.unroll_index([g, x_i, (y_i-1)%self.n_y]))
                    data.append(-mat.D[g] / self.delta_y**2)
                    
                    row.append(i)
                    col.append(self.unroll_index([g, x_i, (y_i+1)%self.n_y]))
                    data.append(-mat.D[g] / self.delta_y**2)'''

                    # double derivative term in x direction
                    if(x_i > 0):
                        row.append(i)
                        col.append(self.unroll_index([g, x_i-1, y_i]))
                        data.append(-mat.D[g] / self.delta_x**2)
                    else:
                        row.append(i)
                        col.append(self.unroll_index([g, x_i, y_i]))
                        data.append(mat.D[g] / self.delta_x**2)
                    if(x_i < self.n_x - 1):
                        row.append(i)
                        col.append(self.unroll_index([g, x_i+1, y_i]))
                        data.append(-mat.D[g] / self.delta_x**2)
                    else:
                        row.append(i)
                        col.append(self.unroll_index([g, x_i, y_i]))
                        data.append(mat.D[g] / self.delta_x**2)

                    # double derivative term in y direction
                    if(y_i > 0):
                        row.append(i)
                        col.append(self.unroll_index([g, x_i, y_i-1]))
                        data.append(-mat.D[g] / self.delta_y**2)
                    else:
                        row.append(i)
                        col.append(self.unroll_index([g, x_i, y_i]))
                        data.append(mat.D[g] / self.delta_y**2)
                    if(y_i < self.n_y - 1):
                        row.append(i)
                        col.append(self.unroll_index([g, x_i, y_i+1]))
                        data.append(-mat.D[g] / self.delta_y**2)
                    else:
                        row.append(i)
                        col.append(self.unroll_index([g, x_i, y_i]))
                        data.append(mat.D[g] / self.delta_y**2)

                    for g_p in range(self.G): # group to group scattering and fission terms
                        row.append(i)
                        col.append(self.unroll_index([g_p, x_i, y_i]))
                        data.append(-(mat.sigma_sgg[g_p, g] + mat.chi[g] * mat.nu_sigma_f[g_p]))
                    b_vector[i] = mat.Q[g]
        #for row in range(len(A_matrix)):
        #    print("row: ", A_matrix[row])
        return sp.csr_matrix((data, (row, col)), shape=(A_mat_size, A_mat_size)), b_vector

    # use finite difference method to construct the A matrix representing the 1D diffusion equation in the form Ax=b, assume 0 flux at boundary conditions for simplicity
    def diffusion_construct_FD_A_matrix_1D(self, A_mat_size):
        fd_order = 2
        A_matrix = np.zeros((self.n_x, self.n_x))
        b_vector = np.zeros(self.n_x)
        for g in range(self.G):
            for x_i in range(self.n_x):
                mat = self.materials[self.material_matrix[x_i,0]]
                i = self.unroll_index([g, x_i, 0])
                
                # combination of double derivative term on current element and total cross section term
                A_matrix[i,i] += 2 * mat.D / self.delta_x**2 + mat.sigma_t[g]
                
                # double derivative term in x direction
                if(x_i > 0):
                    A_matrix[i, self.unroll_index([g, x_i-1, 0])] += -mat.D / self.delta_x**2
                if(x_i < self.n_x - 1):
                    A_matrix[i, self.unroll_index([g, x_i+1, 0])] += -mat.D / self.delta_x**2

                for g_p in range(self.G): # group to group scattering and fission terms
                    A_matrix[i,self.unroll_index([g_p, x_i, 0])] += -(mat.sigma_sgg[g_p, g] + mat.chi[g] * mat.nu_sigma_f[g_p])
                b_vector[i] = mat.Q[g]
        #for row in range(len(A_matrix)):
        #    print("row: ", A_matrix[row])
        return A_matrix, b_vector

    def sp3_construct_A_matrix(self, A_mat_size):
        fd_order = 2
        beta = 0.5
        phi_2_offset = self.G * self.n_x * self.n_y
        A_matrix = np.zeros((A_mat_size, A_mat_size))
        b_vector = np.zeros((A_mat_size))
        for g in range(self.G):
            for x_i in range(self.n_x):
                for y_i in range(self.n_y):
                    i = self.unroll_index([g, x_i, y_i])
                    mat = self.materials[self.material_matrix[x_i,y_i]]
                    D = mat.D[g]
                    D2 = mat.D2[g]

                    # set up some coefficients used by the neutron current variables in the sp3 equation
                    if(x_i == 0 or x_i == self.n_x - 1 or y_i == 0 or y_i == self.n_y - 1):
                        a1 = (1 + 4 * D/self.delta_x)
                        a2 = (-3/4) * D/D2
                        a3 = 2 * D/self.delta_x
                        a4 = (-3/4) * 2 * D / self.delta_x
                        a5 =  4 * D/self.delta_x * 2 * self.get_I_1_value([x_i,y_i])

                        b2 = (1 + (80/21) * D2/self.delta_x)
                        b1 = (-1/7) * D2/D
                        b4 = 2 * D2/self.delta_x
                        b3 = (-2/7) * D2 / self.delta_x
                        b5 =  (6/5) * (80/21) * D2/self.delta_x * self.get_I_3_value([x_i,y_i])

                        c_denom = (a1 - a2 * b1 / b2)
                        c1 = (a5 - a2 * b5 / b2) / c_denom
                        c2 = (a2 * b3 / b2 - a3) / c_denom
                        c3 = (a2 * b4 / b2 - a4) / c_denom

                        d_denom = (a2 - a1 * b2 / b1)
                        d1 = (a5 - a1 * b5 / b1) / d_denom
                        d2 = (a1 * b3 / b1 - a3) / d_denom
                        d3 = (a1 * b4 / b1 - a4) / d_denom

                    # fill in neutron current terms
                    if(x_i == 0): # left BC, normal vector = (-1,0)
                        b_vector[i] += c1 *  self.delta_y
                        A_matrix[i,i] -= c2 *  self.delta_y
                        A_matrix[i,i+phi_2_offset] -= (2*c2 + c3) *  self.delta_y

                        b_vector[i + phi_2_offset] += d1 *  self.delta_y
                        A_matrix[i+phi_2_offset,i] -= d2 *  self.delta_y
                        A_matrix[i+phi_2_offset,i+phi_2_offset] -= (2*d2 + d3) *  self.delta_y
                    else:
                        # Phi_0 equations
                        x_minus_term_0 = self.get_av_D("x",x_i-1,y_i,g) *  self.delta_y
                        A_matrix[i,self.unroll_index([g, x_i-1, y_i])] =  -x_minus_term_0 # phi_0, (i-1,j) term
                        A_matrix[i,self.unroll_index([g, x_i-1, y_i]) + phi_2_offset] =  -2 * x_minus_term_0 # phi_0, (i-1,j) term
                        A_matrix[i,i] +=  x_minus_term_0 # phi_0, (i,j) term
                        A_matrix[i,i+phi_2_offset] +=  2 * x_minus_term_0 # phi_0, (i,j) term

                        # Phi_2 equations
                        x_minus_term_2 = self.get_av_D2("x",x_i-1,y_i,g) *  self.delta_y
                        A_matrix[i+phi_2_offset,self.unroll_index([g, x_i-1, y_i]) + phi_2_offset] =  -x_minus_term_2 # phi_2, (i-1,j) term
                        A_matrix[i+phi_2_offset,i+phi_2_offset] +=  x_minus_term_2 # phi_2, (i,j) term
                    if(x_i == self.n_x - 1): # right BC, normal vector = (1,0)
                        b_vector[i] += c1 *  self.delta_y
                        A_matrix[i,i] -= c2 *  self.delta_y
                        A_matrix[i,i+phi_2_offset] -= (2*c2 + c3) *  self.delta_y

                        b_vector[i + phi_2_offset] += d1 *  self.delta_y
                        A_matrix[i+phi_2_offset,i] -= d2 *  self.delta_y
                        A_matrix[i+phi_2_offset,i+phi_2_offset] -= (2*d2 + d3) *  self.delta_y
                    else:
                        # Phi_0 equations
                        x_plus_term_0 = self.get_av_D("x",x_i,y_i,g) *  self.delta_y
                        A_matrix[i,self.unroll_index([g, x_i+1, y_i])] =  -x_plus_term_0 # phi_0, (i+1,j) term
                        A_matrix[i,self.unroll_index([g, x_i+1, y_i]) + phi_2_offset] =  -2 * x_plus_term_0 # phi_0, (i+1,j) term
                        A_matrix[i,i] +=  x_plus_term_0 # phi_0, (i,j) term
                        A_matrix[i,i+phi_2_offset] +=  2 * x_plus_term_0 # phi_0, (i,j) term

                        # Phi_2 equations
                        x_plus_term_2 = self.get_av_D2("x",x_i,y_i,g) *  self.delta_y
                        A_matrix[i+phi_2_offset,self.unroll_index([g, x_i+1, y_i]) + phi_2_offset] =  -x_plus_term_2 # phi_2, (i+1,j) term
                        A_matrix[i+phi_2_offset,i+phi_2_offset] +=  x_plus_term_2 # phi_2, (i,j) term
                    if(y_i == 0): # bottom BC, normal vector = (0,-1)
                        b_vector[i] += c1 *  self.delta_x
                        A_matrix[i,i] -= c2 *  self.delta_x
                        A_matrix[i,i+phi_2_offset] -= (2*c2 + c3) *  self.delta_x

                        b_vector[i + phi_2_offset] += d1 *  self.delta_x
                        A_matrix[i+phi_2_offset,i] -= d2 *  self.delta_x
                        A_matrix[i+phi_2_offset,i+phi_2_offset] -= (2*d2 + d3) *  self.delta_x
                    else:
                        # Phi_0 equations
                        y_minus_term_0 = self.get_av_D("y",y_i-1,x_i,g) *  self.delta_x
                        A_matrix[i,self.unroll_index([g, x_i, y_i-1])] =  -y_minus_term_0 # phi_0, (i,j-1) term
                        A_matrix[i,self.unroll_index([g, x_i, y_i-1]) + phi_2_offset] =  -2 * y_minus_term_0 # phi_0, (i,j-1) term
                        A_matrix[i,i] +=  y_minus_term_0 # phi_0, (i,j) term
                        A_matrix[i,i+phi_2_offset] +=  2 * y_minus_term_0 # phi_0, (i,j) term

                        # Phi_2 equations
                        y_minus_term_2 = self.get_av_D2("y",y_i-1,x_i,g) *  self.delta_x
                        A_matrix[i+phi_2_offset,self.unroll_index([g,x_i, y_i-1]) + phi_2_offset] =  -y_minus_term_2 # phi_2, (i,j-1) term
                        A_matrix[i+phi_2_offset,i+phi_2_offset] +=  y_minus_term_2 # phi_2, (i,j) term
                    if(y_i == self.n_y - 1): # top BC, normal vector = (0,1)
                        b_vector[i] += c1 *  self.delta_x
                        A_matrix[i,i] -= c2 *  self.delta_x
                        A_matrix[i,i+phi_2_offset] -= (2*c2 + c3) *  self.delta_x

                        b_vector[i + phi_2_offset] += d1 *  self.delta_x
                        A_matrix[i+phi_2_offset,i] -= d2 *  self.delta_x
                        A_matrix[i+phi_2_offset,i+phi_2_offset] -= (2*d2 + d3) *  self.delta_x
                    else:
                        # Phi_0 equations
                        y_plus_term_0 = self.get_av_D("y",y_i,x_i,g) *  self.delta_x
                        A_matrix[i,self.unroll_index([g, x_i, y_i+1])] =  -y_plus_term_0 # phi_0, (i,j+1) term
                        A_matrix[i,self.unroll_index([g, x_i, y_i+1])+phi_2_offset] =  -2 * y_plus_term_0 # phi_0, (i,j+1) term
                        A_matrix[i,i] +=  y_plus_term_0 # phi_0, (i,j) term
                        A_matrix[i,i+phi_2_offset] +=  2*y_plus_term_0 # phi_0, (i,j) term

                        # Phi_2 equations
                        y_plus_term_2 = self.get_av_D2("y",y_i,x_i,g) *  self.delta_x
                        A_matrix[i+phi_2_offset,self.unroll_index([g, x_i, y_i+1]) + phi_2_offset] =  -y_plus_term_2 # phi_2, (i,j+1) term
                        A_matrix[i+phi_2_offset,i+phi_2_offset] +=  y_plus_term_2 # phi_2, (i,j) term
                    A_matrix[i,i] += (mat.sigma_t[g]) *  self.delta_x *  self.delta_y
                    #A_matrix[i,i + phi_2_offset] += (-2) * (mat.sigma_t[g]) *  self.delta_x *  self.delta_y
                    for g_p in range(self.G):
                        A_matrix[i,self.unroll_index([g_p, x_i, y_i])] += (- mat.chi[g] * mat.nu_sigma_f[g_p] - mat.sigma_sgg[g_p,g]) *  self.delta_x *  self.delta_y
                    b_vector[i] += mat.Q[g] *  self.delta_x *  self.delta_y

                    A_matrix[i+phi_2_offset,i] += (-2/5) * (mat.sigma_t[g]) *  self.delta_x *  self.delta_y
                    A_matrix[i+phi_2_offset,i + phi_2_offset] += (mat.sigma_t[g] - mat.sigma_s2[g]) *  self.delta_x *  self.delta_y
                    for g_p in range(self.G):
                        A_matrix[i+phi_2_offset,self.unroll_index([g_p, x_i, y_i])] = (2/5) * (mat.chi[g] * mat.nu_sigma_f[g_p] + mat.sigma_sgg[g_p,g]) *  self.delta_x *  self.delta_y
                    b_vector[i+phi_2_offset] += (-2/5) * mat.Q[g] *  self.delta_x *  self.delta_y
        return A_matrix, b_vector



    # convert 3D (g,x,y) index to 1D index
    def unroll_index(self, index_vec):
        return index_vec[0]*self.n_x*self.n_y + index_vec[1]*self.n_y + index_vec[2]

    # convert 1D index to 3D (g,x,y) index
    def roll_index(self, index):
        g = math.floor(index/(self.n_x * self.n_y))
        x = math.floor((index % (self.n_x * self.n_y))/(self.n_y))
        y = index % self.n_y
        return np.array([g,x,y])



