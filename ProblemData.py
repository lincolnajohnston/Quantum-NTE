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
        self.initialize_materials()
        self.initialize_geometry()

    def read_input(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        self.dim = int(lines[0].strip())
        self.n = np.zeros(self.dim, dtype=int)
        self.h = np.zeros(self.dim)
        for d in range(self.dim):
            self.n[d] = int(lines[1+d].strip())
            self.h[d] = float(lines[1 + self.dim + d].strip())
        self.geometry_name = lines[1 + 2 * self.dim].strip()
        self.sim_method = lines[2 + 2 * self.dim].strip()
        self.G = int(lines[3 + 2 * self.dim].strip())
        self.xs_folder = lines[4 + 2 * self.dim].strip()
        self.mat_name_dict = {}
    
    def initialize_BC(self):
        if self.sim_method == "sp3":
            top_I_3 = 0
            bottom_I_3 = 0
            right_I_3 = 0
            left_I_3 = 0

            # TODO: update this for the new arbitrary number of spatial dimensions
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
            self.beta = 0.5 # related to albedo constant, beta=0.5 corresponds to alpha=0 (vacuum BC) and beta=0 correpsonds to alpha=1 (reflective BC)

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
    def get_av_D(self, direction_index, lower_index, set_indices, g):
        temp_indices = np.array(set_indices)

        temp_indices[direction_index] = lower_index
        D_lower = self.materials[self.material_matrix[tuple(temp_indices)]].D[g]

        temp_indices[direction_index] = lower_index + 1
        D_upper = self.materials[self.material_matrix[tuple(temp_indices)]].D[g]

        delta = self.h[direction_index]
        return 2 * (D_lower/delta) * (D_upper/delta) / (D_lower/delta + D_upper/delta)


    def get_av_D2(self, direction_index, lower_index, spatial_indices, g):
        spatial_indices[direction_index] = lower_index
        D_lower = self.materials[self.material_matrix[tuple(spatial_indices)]].D2[g]

        spatial_indices[direction_index] = lower_index + 1
        D_upper = self.materials[self.material_matrix[tuple(spatial_indices)]].D2[g]
        
        delta = self.h[direction_index]
        return 2 * (D_lower/delta) * (D_upper/delta) / (D_lower/delta + D_upper/delta)


    def get_edge_D(self, spatial_indices, g, delta):
        D = self.materials[self.material_matrix[tuple(spatial_indices)]].D[g]
        return 2 * (self.beta/2) * (D/delta) / (self.beta/2 + (D/delta)) # TODO: allow for different albedos on different boundaries
    

    def initialize_geometry(self):
        self.material_matrix = np.zeros(self.n, dtype=int)
        #x_range = self.n_x * self.delta_x
        #y_range = self.n_y * self.delta_y
        ranges = self.n * self.h

        # custom geometry
        if (self.geometry_name[:7] == "custom:"):
            geometry_filename = self.geometry_name[7:]
            i = 0
            with open(geometry_filename, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    for word in line.split():
                        indices = self.roll_index(i)[1:]
                        self.material_matrix[tuple(indices)] = self.mat_name_dict[word]
                        i+=1
                assert(i == self.G * math.prod(self.n))
        else:
            # pre-baked geometries       
            for index in range(math.prod(self.n)):
                indices = self.roll_index(index)[1:]
                
                if(self.geometry_name == "homogeneous_fuel"):
                    self.material_matrix[tuple(indices)] = self.mat_name_dict["fuel"]
                elif (self.geometry_name == "single_pin_cell_2d"):
                    fuel_radius = min(ranges)/4
                    coordinates = (indices + 0.5) * self.h - ranges/2

                    if (np.linalg.norm(coordinates) < fuel_radius):
                        # use fuel XSs
                        self.material_matrix[tuple(indices)] = self.mat_name_dict["fuel"]
                        
                    else:
                        # use moderator XSs
                        self.material_matrix[tuple(indices)] = self.mat_name_dict["water"]
                elif (self.geometry_name == "single_square_pin"):
                    fuel_radius = min(ranges)/8 # radius of central fuel section
                    coordinates = (indices + 0.5) * self.h - ranges/2 # position of center of finite volume

                    if (np.all(abs(coordinates) <= fuel_radius)):
                        # use fuel XSs
                        self.material_matrix[tuple(indices)] = self.mat_name_dict["fuel"]
                        
                    else:
                        # use moderator XSs
                        self.material_matrix[tuple(indices)] = self.mat_name_dict["water"]
        return

    def initialize_materials(self):
        self.materials = {}
        mat_index = 0
        for _, dirs, _ in os.walk(self.xs_folder):
            for mat in dirs:
                xs_file = self.xs_folder + "/" + mat + "/xs.txt"
                self.mat_name_dict[mat] = mat_index
                self.materials[mat_index] = Material(mat, xs_file, self.G)
                mat_index += 1
    
    def diffusion_construct_A_matrix(self, A_mat_size):
        fd_order = 2
        A_matrix = np.zeros((A_mat_size, A_mat_size))
        b_vector = np.zeros(A_mat_size)
        delta_V = math.prod(self.h)
        for i in range(self.G * math.prod(self.n)):
            indices = self.roll_index(i)
            g = indices[0] # current energy group
            mat = self.materials[self.material_matrix[tuple(indices[1:])]]
            for d in range(self.dim): # apply terms for neutron current in each spatial dimension
                x_i = indices[d+1] # index of position in the spatial dimension for which the current term is being created in this iteration
                if(x_i == 0): # left BC, normal vector = (-1,0)
                    D_left = self.get_edge_D(indices[1:], g, self.h[d])
                    J_left = -D_left
                    A_matrix[i,i] += -J_left * delta_V / self.h[d]
                else: # internal edge on left side of finite volume
                    D_left = self.get_av_D(d,x_i-1,indices[1:], g)
                    A_matrix[i,i] += D_left * delta_V / self.h[d] # (i) term

                    left_indices = np.array(indices)
                    left_indices[d+1] -= 1
                    A_matrix[i,self.unroll_index(left_indices)] +=  -D_left * delta_V / self.h[d] # (i-1) term
                if(x_i == self.n[d] - 1): # right BC, normal vector = (1,0)
                    D_right = self.get_edge_D(indices[1:], g, self.h[d])
                    J_right = D_right
                    A_matrix[i,i] += J_right * delta_V / self.h[d]
                else: # internal edge on right side of finite volume
                    D_right = self.get_av_D(d,x_i,indices[1:], g)
                    A_matrix[i,i] += D_right * delta_V / self.h[d] # (i) term

                    right_indices = np.array(indices)
                    right_indices[d+1] += 1
                    A_matrix[i,self.unroll_index(right_indices)] +=  -D_right * delta_V / self.h[d] # (i+1) term
            for g_p in range(self.G): # group to group scattering and fission terms
                group_indices = np.array(indices)
                group_indices[0] = g_p
                A_matrix[i,self.unroll_index(group_indices)] -= (mat.sigma_sgg[g_p, g]) * delta_V
                A_matrix[i,self.unroll_index(group_indices)] -= (mat.chi[g] * mat.nu_sigma_f[g_p]) * delta_V
            A_matrix[i,i] += (mat.sigma_t[g]) * delta_V
            b_vector[i] = mat.Q[g] * delta_V
        return A_matrix, b_vector
    
    def diffusion_construct_diffusion_operator(self, A_mat_size):
        L_matrix = np.zeros((A_mat_size, A_mat_size))
        delta_V = math.prod(self.h)
        for i in range(self.G * math.prod(self.n)):
            indices = self.roll_index(i)
            g = indices[0] # current energy group
            mat = self.materials[self.material_matrix[tuple(indices[1:])]]
            for d in range(self.dim): # apply terms for neutron current in each spatial dimension
                x_i = indices[d+1] # index of position in the spatial dimension for which the current term is being created in this iteration
                if(x_i == 0): # left BC, normal vector = (-1,0)
                    D_left = self.get_edge_D(indices[1:], g, self.h[d])
                    J_left = -D_left
                    L_matrix[i,i] += -J_left * delta_V / self.h[d]
                else: # internal edge on left side of finite volume
                    D_left = self.get_av_D(d,x_i-1,indices[1:], g)
                    L_matrix[i,i] += D_left * delta_V / self.h[d] # (i) term

                    left_indices = np.array(indices)
                    left_indices[d+1] -= 1
                    L_matrix[i,self.unroll_index(left_indices)] +=  -D_left * delta_V / self.h[d] # (i-1) term
                if(x_i == self.n[d] - 1): # right BC, normal vector = (1,0)
                    D_right = self.get_edge_D(indices[1:], g, self.h[d])
                    J_right = D_right
                    L_matrix[i,i] += J_right * delta_V / self.h[d]
                else: # internal edge on right side of finite volume
                    D_right = self.get_av_D(d,x_i,indices[1:], g)
                    L_matrix[i,i] += D_right * delta_V / self.h[d] # (i) term

                    right_indices = np.array(indices)
                    right_indices[d+1] += 1
                    L_matrix[i,self.unroll_index(right_indices)] +=  -D_right * delta_V / self.h[d] # (i+1) term
        return L_matrix
    
    def diffusion_construct_L_F_matrices(self, A_mat_size):
        fd_order = 2
        L_matrix = np.zeros((A_mat_size, A_mat_size))
        F_matrix = np.zeros((A_mat_size, A_mat_size))
        delta_V = math.prod(self.h)
        for i in range(self.G * math.prod(self.n)):
            indices = self.roll_index(i)
            g = indices[0] # current energy group
            mat = self.materials[self.material_matrix[tuple(indices[1:])]]
            for d in range(self.dim): # apply terms for neutron current in each spatial dimension
                x_i = indices[d+1] # index of position in the spatial dimension for which the current term is being created in this iteration
                if(x_i == 0): # left BC, normal vector = (-1,0)
                    D_left = self.get_edge_D(indices[1:], g, self.h[d])
                    J_left = -D_left
                    L_matrix[i,i] += -J_left * delta_V / self.h[d]
                else: # internal edge on left side of finite volume
                    D_left = self.get_av_D(d,x_i-1,indices[1:], g)
                    L_matrix[i,i] += D_left * delta_V / self.h[d] # (i) term

                    left_indices = np.array(indices)
                    left_indices[d+1] -= 1
                    L_matrix[i,self.unroll_index(left_indices)] +=  -D_left * delta_V / self.h[d] # (i-1) term
                if(x_i == self.n[d] - 1): # right BC, normal vector = (1,0)
                    D_right = self.get_edge_D(indices[1:], g, self.h[d])
                    J_right = D_right
                    L_matrix[i,i] += J_right * delta_V / self.h[d]
                else: # internal edge on right side of finite volume
                    D_right = self.get_av_D(d,x_i,indices[1:], g)
                    L_matrix[i,i] += D_right * delta_V / self.h[d] # (i) term

                    right_indices = np.array(indices)
                    right_indices[d+1] += 1
                    L_matrix[i,self.unroll_index(right_indices)] +=  -D_right * delta_V / self.h[d] # (i+1) term
            for g_p in range(self.G): # group to group scattering and fission terms
                group_indices = np.array(indices)
                group_indices[0] = g_p
                L_matrix[i,self.unroll_index(group_indices)] += -(mat.sigma_sgg[g_p, g]) * delta_V
                F_matrix[i,self.unroll_index(group_indices)] += (mat.chi[g] * mat.nu_sigma_f[g_p]) * delta_V
            L_matrix[i,i] += (mat.sigma_t[g]) * delta_V
        return L_matrix, F_matrix
    
    # use finite difference discretization with spatially-varying diffusion coefficient
    # 0 Dirichlet BCs
    def diffusion_FD_construct_L_F_matrices(self):
        A_mat_size = math.prod(self.n) * self.G
        L_matrix = np.zeros((A_mat_size, A_mat_size))
        F_matrix = np.zeros((A_mat_size, A_mat_size))
        for i in range(A_mat_size):
            indices = self.roll_index(i)
            g = indices[0] # current energy group
            mat = self.materials[self.material_matrix[tuple(indices[1:])]]
            for d in range(self.dim): # apply terms for neutron current in each spatial dimension
                x_i = indices[d+1] # index of position in the spatial dimension for which the current term is being created in this iteration
                L_matrix[i,i] +=  2 * mat.D[g] / (self.h[d]**2) # diffusion term

                left_indices = np.array(indices)
                left_indices[d+1] -= 1
                right_indices = np.array(indices)
                right_indices[d+1] += 1

                # find the material at the neighboring points, set the material at the ghost points 
                # on the boundaries to be the same as the closest internal points
                if(x_i == 0): # else left BC, point to the left is 0
                    mat_left = mat
                else:
                    mat_left = self.materials[self.material_matrix[tuple(left_indices[1:])]]

                if(x_i == self.n[d] - 1):
                    mat_right = mat
                else:
                    mat_right = self.materials[self.material_matrix[tuple(right_indices[1:])]]

                # set the off-diagonal values from the diffusion term
                if(x_i != 0): # else left BC, point to the left is 0
                    L_matrix[i,self.unroll_index(left_indices)] -=  (mat.D[g] - mat_right.D[g]/4 + mat_left.D[g]/4) / (self.h[d]**2)
                
                if(x_i != self.n[d] - 1): # else right BC, point to the right is 0
                    L_matrix[i,self.unroll_index(right_indices)] -=  (mat.D[g] + mat_right.D[g]/4 - mat_left.D[g]/4) / (self.h[d]**2)

            # add in group to group fission and scattering terms
            for g_p in range(self.G): # group to group scattering and fission terms
                group_indices = np.array(indices)
                group_indices[0] = g_p
                L_matrix[i,self.unroll_index(group_indices)] += -(mat.sigma_sgg[g_p, g])
                F_matrix[i,self.unroll_index(group_indices)] += (mat.chi[g] * mat.nu_sigma_f[g_p])
            L_matrix[i,i] += (mat.sigma_t[g])
        return L_matrix, F_matrix

    
    def sp3_construct_A_matrix(self, A_mat_size):
        fd_order = 2
        beta = 0.5 # hardcoded vacuum boundary condition
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



    # convert multidimensional (g,x1,x2...,xD) index to 1D index
    def unroll_index(self, index_vec):
        t1 = [index_vec[d]*math.prod(self.n[d:]) for d in range(len(index_vec)-1)]
        return sum([index_vec[d]*math.prod(self.n[d:]) for d in range(len(index_vec)-1)]) + index_vec[-1]

    # convert 1D index to multidimensional (g,x1,x2...,xD) index
    def roll_index(self, index):
        indices = np.zeros(1 + self.dim, dtype=int) # 1 index for energy group, self.dim dimensions for spatial dimensions
        for d in range(self.dim + 1):
            indices[d] = index
            if d < self.dim:
                indices[d] = math.floor(indices[d] / math.prod(self.n[d:]))
            if d > 0:
                indices[d] = indices[d] % self.n[d-1]
        return indices



