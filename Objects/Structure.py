# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:02:25 2024

@author: ibouckaert
"""

# Standard imports 

import importlib
import os
import time
import warnings
from copy import deepcopy

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc


def reload_modules():
    importlib.reload(bl)
    importlib.reload(cf)
    importlib.reload(mat)
    importlib.reload(tfe)


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"


warnings.formatwarning = custom_warning_format

from Objects import Block as bl
from Objects import ContactFace as cf
from Objects import Material as mat
from Objects import Timoshenko_FE as tfe

reload_modules()


class Structure_2D:

    def __init__(self):

        self.list_blocks = []
        self.list_fes = []
        self.list_nodes = []

    def add_fe(self, N1, N2, E, nu, h, b=1, lin_geom=True, rho=0.):

        self.list_fes.append(tfe.Timoshenko_FE_2D(N1, N2, E, nu, b, h, lin_geom=lin_geom, rho=rho))

    def add_block(self, vertices, rho, b=1, material=None, ref_point=None):

        self.list_blocks.append(bl.Block_2D(vertices, rho, b=b, material=material, ref_point=ref_point))

    def add_beam(self, N1, N2, n_blocks, h, rho, b=1, material=None, end_1=True, end_2=True):

        lx = N2[0] - N1[0]
        ly = N2[1] - N1[1]
        L = np.sqrt(lx ** 2 + ly ** 2)
        L_b = L / (n_blocks - 1)

        long = np.array([lx, ly]) / L
        tran = np.array([-ly, lx]) / L

        # Loop to create the blocks
        ref_point = N1.copy()

        for i in np.arange(n_blocks):

            # Initialize array of vertices
            vertices = np.array([ref_point, ref_point, ref_point, ref_point])

            if i == 0:  # First block is half block
                vertices[0] += L_b / 2 * long - h / 2 * tran
                vertices[1] += L_b / 2 * long + h / 2 * tran
                vertices[2] += h / 2 * tran
                vertices[3] += -h / 2 * tran

                if end_1:
                    ref = ref_point
                else:
                    ref = None

            elif i == n_blocks - 1:  # Last block is also a half_block
                vertices[0] += -h / 2 * tran
                vertices[1] += h / 2 * tran
                vertices[2] += h / 2 * tran - L_b / 2 * long
                vertices[3] += -h / 2 * tran - L_b / 2 * long

                if end_2:
                    ref = ref_point
                else:
                    ref = None

            else:
                vertices[0] += -h / 2 * tran + L_b / 2 * long
                vertices[1] += h / 2 * tran + L_b / 2 * long
                vertices[2] += h / 2 * tran - L_b / 2 * long
                vertices[3] += -h / 2 * tran - L_b / 2 * long
                ref = None

            self.add_block(vertices, rho, b=b, material=material, ref_point=ref)

            ref_point += L_b * long

    def add_tapered_beam(self, N1, N2, n_blocks, h1, h2, rho, b=1, material=None, contact=None, end_1=True, end_2=True):

        lx = N2[0] - N1[0]
        ly = N2[1] - N1[1]
        L = np.sqrt(lx ** 2 + ly ** 2)
        L_b = L / (n_blocks - 1)

        heights = np.linspace(h1, h2, n_blocks)
        d_h = (heights[1] - heights[0]) / 2

        long = np.array([lx, ly]) / L
        tran = np.array([-ly, lx]) / L

        # Loop to create the blocks
        ref_point = N1.copy()

        for i in np.arange(n_blocks):

            # Initialize array of vertices
            vertices = np.array([ref_point, ref_point, ref_point, ref_point])

            if i == 0:  # First block is half block
                h1 = heights[i]
                h2 = heights[i] + d_h
                vertices[0] += L_b / 2 * long - h2 / 2 * tran
                vertices[1] += L_b / 2 * long + h2 / 2 * tran
                vertices[2] += h1 / 2 * tran
                vertices[3] += -h1 / 2 * tran

                if end_1:
                    ref = ref_point
                else:
                    ref = None

            elif i == n_blocks - 1:  # Last block is also a half_block
                h2 = heights[i]
                h1 = heights[i] - d_h
                vertices[0] += -h2 / 2 * tran
                vertices[1] += h2 / 2 * tran
                vertices[2] += h1 / 2 * tran - L_b / 2 * long
                vertices[3] += -h1 / 2 * tran - L_b / 2 * long

                if end_2:
                    ref = ref_point
                else:
                    ref = None

            else:
                h1 = heights[i] - d_h
                h2 = heights[i] + d_h
                vertices[0] += -h2 / 2 * tran + L_b / 2 * long
                vertices[1] += h2 / 2 * tran + L_b / 2 * long
                vertices[2] += h1 / 2 * tran - L_b / 2 * long
                vertices[3] += -h1 / 2 * tran - L_b / 2 * long
                ref = None

            self.add_block(vertices, rho, b=b, material=material, ref_point=ref)

            ref_point += L_b * long

    def add_arch(self, c, a1, a2, R, n_blocks, h, rho, b=1, material=None, contact=None):

        d_a = (a2 - a1) / (n_blocks)
        angle = a1

        R_int = R - h / 2
        R_out = R + h / 2

        for i in np.arange(n_blocks):
            # Initialize array of vertices
            vertices = np.array([c, c, c, c])

            unit_dir_1 = np.array([np.cos(angle), np.sin(angle)])
            unit_dir_2 = np.array([np.cos(angle + d_a), np.sin(angle + d_a)])
            vertices[0] += R_int * unit_dir_1
            vertices[1] += R_out * unit_dir_1
            vertices[2] += R_out * unit_dir_2
            vertices[3] += R_int * unit_dir_2

            # print(vertices)
            self.add_block(vertices, rho, b=b, material=material)

            angle += d_a

    def add_wall(self, c1, l_block, h_block, pattern, rho, b=1, material=None, orientation=None):

        if orientation is not None:
            long = orientation
            tran = np.array([-orientation[1], orientation[0]])
        else:
            long = np.array([1, 0], dtype=float)
            tran = np.array([0, 1], dtype=float)

        for j, line in enumerate(pattern):

            ref_point = .5 * abs(line[0]) * l_block * long + (j + 0.5) * h_block * tran

            for i, brick in enumerate(line):

                if brick > 0:
                    vertices = np.array([ref_point, ref_point, ref_point, ref_point])
                    vertices[0] += brick * l_block / 2 * long - h_block / 2 * tran
                    vertices[1] += brick * l_block / 2 * long + h_block / 2 * tran
                    vertices[2] += -brick * l_block / 2 * long + h_block / 2 * tran
                    vertices[3] += -brick * l_block / 2 * long - h_block / 2 * tran

                    self.add_block(vertices, rho, b=b, material=material)

                if not i == len(line) - 1:
                    ref_point += .5 * l_block * long * (abs(brick) + abs(line[i + 1]))

    def add_voronoi_surface(self, surface, list_of_points, rho, b=1, material=None):

        # Surface is a list of points defining the surface to be subdivided into 
        # Voronoi cells. 

        def point_in_surface(point, surface):
            # Check if a point lies on the surface
            # Surface is a list of points delimiting the surface
            # Point is a 2D numpy array

            n = len(surface)

            for i in range(n):
                A = surface[i]
                B = surface[(i + 1) % n]
                C = point

                if np.cross(B - A, C - A) < 0:
                    return False

            return True

        for point in list_of_points:
            # Check if all points lie on the surface
            if not point_in_surface(point, surface):
                warnings.warn('Not all points lie on the surface')
                return

        # Create Voronoi cells  
        vor = sc.spatial.Voronoi(list_of_points)

        # Create block for each Voronoi region
        # If region is finite, it's easy
        # If region is infinite, delimit it with the edge of the surface
        for region in vor.regions[1:]:

            if not -1 in region:
                vertices = np.array([vor.vertices[i] for i in region])
                self.add_block(vertices, rho, b=b, material=material)

            else:
                vertices = []
                for i in region:
                    if not i == -1:
                        vertices.append(vor.vertices[i])

                # Find the edges of the surface that intersect the infinite cell
                for i in range(len(vertices)):
                    A = vertices[i]
                    B = vertices[(i + 1) % len(vertices)]

                    for j in range(len(surface)):
                        C = surface[j]
                        D = surface[(j + 1) % len(surface)]

                        if np.cross(B - A, C - A) * np.cross(B - A, D - A) < 0:
                            # Intersection between AB and CD
                            if np.cross(D - C, A - C) * np.cross(D - C, B - C) < 0:
                                # Intersection between CD and AB
                                vertices.insert(i + 1, C + np.cross(D - C, A - C) / np.cross(D - C, B - C) * (B - A))
                                vertices.insert(i + 2, D)
                                break

                self.add_block(np.array(vertices), rho, b=b, material=material)

    def make_nodes(self):

        def node_exists(node):

            for i, n in enumerate(self.list_nodes):
                if np.all(np.isclose(n, node, rtol=1e-8)): return i, True

            return -1, False

        self.list_nodes = []

        for i, bl in enumerate(self.list_blocks):

            index, exists = node_exists(bl.ref_point)

            if exists:
                bl.make_connect(index)
            else:
                self.list_nodes.append(bl.ref_point.copy())
                bl.make_connect(len(self.list_nodes) - 1)

        for i, fe in enumerate(self.list_fes):

            for j, node in enumerate(fe.nodes):

                index, exists = node_exists(node)

                if exists:
                    fe.make_connect(index, j)
                else:
                    self.list_nodes.append(node)
                    fe.make_connect(len(self.list_nodes) - 1, j)

        self.nb_dofs = 3 * len(self.list_nodes)
        self.U = np.zeros(self.nb_dofs)
        self.P = np.zeros(self.nb_dofs)
        self.P_fixed = np.zeros(self.nb_dofs)
        self.dof_fix = np.array([], dtype=int)
        self.dof_free = np.arange(self.nb_dofs, dtype=int)
        self.nb_dof_fix = 0
        self.nb_dof_free = self.nb_dofs

    def make_cfs(self, lin_geom, nb_cps=2, offset=-1, contact=None, surface=None):

        interfaces = self.detect_interfaces()
        self.list_cfs = []

        for contactface in interfaces:
            self.list_cfs.append(
                cf.CF_2D(contactface, nb_cps, lin_geom, offset=offset, contact=contact, surface=surface))

    def get_P_r(self):

        if not hasattr(self, 'nb_dofs'): warnings.warn('The DoFs of the structure were not defined')

        self.P_r = np.zeros(self.nb_dofs)

        for CF in self.list_cfs:
            qf_glob = np.zeros(6)
            qf_glob[:3] = self.U[CF.bl_A.dofs]
            qf_glob[3:] = self.U[CF.bl_B.dofs]

            pf_glob = CF.get_pf_glob(qf_glob)

            self.P_r[CF.bl_A.dofs] += pf_glob[:3]
            self.P_r[CF.bl_B.dofs] += pf_glob[3:]

        for FE in self.list_fes:
            q_glob = self.U[FE.dofs]
            p_glob = FE.get_p_glob(q_glob)
            self.P_r[FE.dofs] += p_glob

    def fixNode(self, node_ids, dofs):

        def fix_onenode(Str, index):
            Str.dof_fix = np.append(Str.dof_fix, index)
            Str.dof_free = Str.dof_free[Str.dof_free != index]
            Str.nb_dof_fix = len(Str.dof_fix)
            Str.nb_dof_free = len(Str.dof_free)

        if isinstance(node_ids, int):  # Loading one single node
            if isinstance(dofs, int):
                fix_onenode(self, 3 * node_ids + dofs)
            elif isinstance(dofs, list):
                for i in dofs:
                    fix_onenode(self, 3 * node_ids + i)
            else:
                warnings.warn('DoFs to be fixed is not an int or a list')
        elif isinstance(node_ids, list):
            for j in node_ids:
                if isinstance(dofs, int):
                    fix_onenode(self, 3 * j + dofs)
                elif isinstance(dofs, list):
                    for i in dofs:
                        fix_onenode(self, 3 * j + i)
                else:
                    warnings.warn('DoFs to be fixed is not an int or a list')

        elif isinstance(node_ids, np.ndarray) and node_ids.size == 2:  # With coordinates of Node

            node_fixed = -1
            for index, node in enumerate(self.list_nodes):
                if np.allclose(node, node_ids, rtol=1e-9):
                    node_fixed = index
                    break

            if node_fixed < 0:
                warnings.warn('Input node to be fixed does not exist')
            else:
                if isinstance(dofs, int):
                    fix_onenode(self, 3 * node_fixed + dofs)
                elif isinstance(dofs, list):
                    for i in dofs:
                        fix_onenode(self, 3 * node_fixed + i)

        else:
            warnings.warn('Nodes to be loaded must be int, list of ints or numpy array')

    def loadNode(self, node_ids, dofs, force, fixed=False):

        def load_onenode(Str, node_id, dof, force, fixed=False):

            index = 3 * node_id + dof
            if fixed:
                Str.P_fixed[index] = force
            else:
                Str.P[index] = force

        if isinstance(node_ids, int):
            if isinstance(dofs, int):
                load_onenode(self, node_ids, dofs, force, fixed)
            elif isinstance(dofs, list):
                for i in dofs:
                    load_onenode(self, node_ids, i, force, fixed)
            else:
                warnings.warn('DoFs to be loaded is not an int or a list')
        elif isinstance(node_ids, list):
            for j in node_ids:
                if isinstance(dofs, int):
                    load_onenode(self, j, dofs, force, fixed)
                elif isinstance(dofs, list):
                    for i in dofs:
                        load_onenode(self, j, i, force, fixed)
                else:
                    warnings.warn('DoFs to be loaded is not an int or a list')

        elif isinstance(node_ids, np.ndarray) and node_ids.size == 2:  # With coordinates of Node

            node_loaded = -1
            for index, node in enumerate(self.list_nodes):
                if np.allclose(node, node_ids, rtol=1e-9):
                    node_loaded = index
                    break

            if node_loaded < 0:
                warnings.warn('Input node to be loaded does not exist')

            else:
                if isinstance(dofs, int):
                    load_onenode(self, node_loaded, dofs, force, fixed)
                elif isinstance(dofs, list):
                    for i in dofs:
                        load_onenode(self, node_loaded, i, force, fixed)

        else:
            warnings.warn('Nodes to be loaded must be int, list of ints or numpy array')

    def get_node_id(self, node):

        if node.size != 2: warnings.warn('Input node should be an array of size 2')  # With coordinates of Node

        for index, n in enumerate(self.list_nodes):
            if np.allclose(n, node, rtol=1e-9):
                return index

        warnings.warn('Input node to be loaded does not exist')

    def get_M_str(self, no_inertia=False):

        if not hasattr(self, 'nb_dofs'): warnings.warn('The DoFs of the structure were not defined')

        self.M = np.zeros((self.nb_dofs, self.nb_dofs))

        for block in self.list_blocks:
            self.M[np.ix_(block.dofs, block.dofs)] += block.get_mass(no_inertia=no_inertia)

        for FE in self.list_fes:
            mass_fe = FE.get_mass(no_inertia=no_inertia)
            self.M[np.ix_(FE.dofs[:3], FE.dofs[:3])] += mass_fe[:3, :3]
            self.M[np.ix_(FE.dofs[3:], FE.dofs[3:])] += mass_fe[3:, 3:]

    def get_K_str(self):

        if not hasattr(self, 'nb_dofs'): warnings.warn('The DoFs of the structure were not defined')

        self.K = np.zeros((self.nb_dofs, self.nb_dofs))

        for CF in self.list_cfs:
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob = CF.get_kf_glob()

            self.K[np.ix_(dof1, dof1)] += kf_glob[:3, :3]
            self.K[np.ix_(dof1, dof2)] += kf_glob[:3, 3:]
            self.K[np.ix_(dof2, dof1)] += kf_glob[3:, :3]
            self.K[np.ix_(dof2, dof2)] += kf_glob[3:, 3:]

        for FE in self.list_fes:
            k_glob = FE.get_k_glob()

            self.K[np.ix_(FE.dofs[:3], FE.dofs[:3])] += k_glob[:3, :3]
            self.K[np.ix_(FE.dofs[:3], FE.dofs[3:])] += k_glob[:3, 3:]
            self.K[np.ix_(FE.dofs[3:], FE.dofs[:3])] += k_glob[3:, :3]
            self.K[np.ix_(FE.dofs[3:], FE.dofs[3:])] += k_glob[3:, 3:]

    def get_K_str0(self):

        if not hasattr(self, 'nb_dofs'): warnings.warn('The DoFs of the structure were not defined')

        self.K0 = np.zeros((self.nb_dofs, self.nb_dofs))

        for CF in self.list_cfs:
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob = CF.get_kf_glob0()

            self.K0[np.ix_(dof1, dof1)] += kf_glob[:3, :3]
            self.K0[np.ix_(dof1, dof2)] += kf_glob[:3, 3:]
            self.K0[np.ix_(dof2, dof1)] += kf_glob[3:, :3]
            self.K0[np.ix_(dof2, dof2)] += kf_glob[3:, 3:]

        for FE in self.list_fes:
            k_glob = FE.get_k_glob0()

            self.K0[np.ix_(FE.dofs[:3], FE.dofs[:3])] += k_glob[:3, :3]
            self.K0[np.ix_(FE.dofs[:3], FE.dofs[3:])] += k_glob[:3, 3:]
            self.K0[np.ix_(FE.dofs[3:], FE.dofs[:3])] += k_glob[3:, :3]
            self.K0[np.ix_(FE.dofs[3:], FE.dofs[3:])] += k_glob[3:, 3:]

    def solve_linear(self):

        self.get_P_r()
        self.get_K_str0()

        K_ff = self.K0[np.ix_(self.dof_free, self.dof_free)]
        K_fr = self.K0[np.ix_(self.dof_free, self.dof_fix)]
        K_rf = self.K0[np.ix_(self.dof_fix, self.dof_free)]
        K_rr = self.K0[np.ix_(self.dof_fix, self.dof_fix)]

        self.U[self.dof_free] = np.linalg.solve(K_ff, self.P[self.dof_free] + self.P_fixed[self.dof_free] - K_fr @ self.U[self.dof_fix])
        # self.P[self.dof_fix] = K_rf @ self.U[self.dof_free] + K_rr @ self.U[self.dof_fix]
        self.get_P_r()

    def commit(self):

        for CF in self.list_cfs:
            CF.commit()

    def revert_commit(self):

        for CF in self.list_cfs:
            CF.revert_commit()

    def solve_forcecontrol(self, steps, tol=1, stiff='tan', max_iter=15, filename='Results_ForceControl', dir_name=''):

        time_start = time.time()

        if isinstance(steps, list):
            nb_steps = len(steps)
            lam = steps

        elif isinstance(steps, int):
            nb_steps = steps
            lam = np.linspace(0, 1, nb_steps + 1)

        else:
            warnings.warn('Steps of the simulation should be either a list or a number of steps (int)')

        # Displacements, forces and stiffness
        U_conv = np.zeros((self.nb_dofs, nb_steps + 1), dtype=float)
        P_r_conv = np.zeros((self.nb_dofs, nb_steps + 1), dtype=float)
        K_conv = np.zeros((self.nb_dofs, self.nb_dofs, nb_steps + 1), dtype=float)

        # Parameters of the simulation
        iter_counter = np.zeros(nb_steps)
        res_counter = np.zeros(nb_steps)

        # self.revert_commit()

        self.get_P_r()

        U_conv[:, 0] = deepcopy(self.U)
        P_r_conv[:, 0] = deepcopy(self.P_r)
        K_conv[:, :, 0] = deepcopy(self.K)

        non_conv = False

        for i in range(nb_steps):

            converged = False
            iteration = 0

            P_target = lam[i] * self.P + self.P_fixed
            # print(np.around(P_target,2))   

            while not converged:

                self.get_P_r()
                self.get_K_str()
                # print(np.around(self.P_r,2))
                R = P_target[self.dof_free] - self.P_r[self.dof_free]

                res = np.linalg.norm(R)

                if res < tol:
                    converged = True

                else:
                    self.revert_commit()

                    try:
                        self.U[self.dof_free] += np.linalg.solve(self.K[np.ix_(self.dof_free, self.dof_free)], R)

                    except np.linalg.LinAlgError:
                        warnings.warn('Should use either the tangent, secant or initial stiffness')

                    iteration += 1

                    if iteration > max_iter:
                        non_conv = True
                        break

            if non_conv:
                print(f'Method did not converge at step {i + 1}')
                break

            else:
                self.commit()
                res_counter[i] = res
                iter_counter[i] = iteration
                last_conv = i + 1

                U_conv[:, i + 1] = self.U.copy()
                P_r_conv[:, i + 1] = self.P_r.copy()
                K_conv[:, :, i + 1] = self.K.copy()

                print(f'Step {i + 1} converged after {iteration} steps')

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f'Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file')

        filename = filename + '.h5'
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, 'w') as hf:

            hf.create_dataset('U_conv', data=U_conv)
            hf.create_dataset('P_r_conv', data=P_r_conv)
            hf.create_dataset('K_conv', data=K_conv)
            hf.create_dataset('Residuals', data=res_counter)
            hf.create_dataset('Iterations', data=iter_counter)
            hf.create_dataset('Last_conv', data=last_conv)
            hf.create_dataset('Load_Multiplier', data=lam)

            hf.attrs['Descr'] = f'Results of the force_control simulation'
            hf.attrs['Tolerance'] = tol
            hf.attrs['Lambda'] = lam
            hf.attrs['Simulation_Time'] = total_time

    def solve_dispcontrol(self, steps, disp, node, dof, tol=1, stiff='tan', max_iter=100,
                          filename='Results_DispControl', dir_name=''):

        time_start = time.time()

        if isinstance(steps, list):
            nb_steps = len(steps) - 1
            lam = [step / max(steps, key=abs) for step in steps]
            d_c = steps

        elif isinstance(steps, int):
            nb_steps = steps
            lam = np.linspace(0, 1, nb_steps + 1)
            print(lam)
            d_c = lam * disp

        else:
            warnings.warn('Steps of the simulation should be either a list or a number of steps (int)')
        # Displacements, forces and stiffness
        U_conv = np.zeros((self.nb_dofs, nb_steps + 1), dtype=float)
        P_r_conv = np.zeros((self.nb_dofs, nb_steps + 1), dtype=float)
        K_conv = np.zeros((self.nb_dofs, self.nb_dofs, nb_steps + 1), dtype=float)

        try:
            U_conv[:, 0] = deepcopy(self.U)
            P_r_conv[:, 0] = deepcopy(self.P_r)
            K_conv[:, :, 0] = deepcopy(self.K)
        except:
            self.get_P_r()
            self.get_K_str()
            U_conv[:, 0] = deepcopy(self.U)
            P_r_conv[:, 0] = deepcopy(self.P_r)
            K_conv[:, :, 0] = deepcopy(self.K)

        # Parameters of the simulation
        iter_counter = np.zeros(nb_steps)
        res_counter = np.zeros(nb_steps)

        control_dof = 3 * node + dof

        non_conv = False

        for i in range(1, nb_steps + 1):

            converged = False
            iteration = 0

            while not converged:

                if iteration == 0:
                    P_target = lam[i] * self.P

                self.get_P_r()
                self.get_K_str()

                R = P_target[self.dof_free] - self.P_r[self.dof_free] + self.P_fixed[self.dof_free]
                res = np.linalg.norm(R)

                if res < tol:
                    converged = True

                else:
                    self.revert_commit()

                    dU_R = np.zeros(self.nb_dofs)
                    dU_ref = np.zeros(self.nb_dofs)

                    try:
                        dU_R[self.dof_free] = np.linalg.solve(self.K[np.ix_(self.dof_free, self.dof_free)], R)
                        dU_ref[self.dof_free] = np.linalg.solve(self.K[np.ix_(self.dof_free, self.dof_free)],
                                                                self.P[self.dof_free])

                    except np.linalg.LinAlgError:
                        warnings.warn('Should use either the tangent, secant or initial stiffness')

                    if iteration == 0:
                        dU_c = d_c[i] - d_c[i - 1]
                        d_lam = (dU_c - dU_R[control_dof]) / dU_ref[control_dof]
                    else:
                        d_lam = - dU_R[control_dof] / dU_ref[control_dof]

                    dU = dU_R + d_lam * dU_ref
                    P_target += d_lam * self.P

                    self.U += dU

                    iteration += 1

                if iteration > max_iter:
                    non_conv = True
                    print(f'Method did not converge at step {i + 1}')
                    break

            if non_conv:
                break

            else:
                self.commit()

                res_counter[i - 1] = res
                iter_counter[i - 1] = iteration
                last_conv = i

                U_conv[:, i] = deepcopy(self.U)
                P_r_conv[:, i] = deepcopy(self.P_r)
                K_conv[:, :, i] = deepcopy(self.K)

                print(f'Step {i} converged after {iteration} steps')

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f'Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file')

        filename = filename + '.h5'
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, 'w') as hf:

            hf.create_dataset('U_conv', data=U_conv)
            hf.create_dataset('P_r_conv', data=P_r_conv)
            hf.create_dataset('K_conv', data=K_conv)
            hf.create_dataset('Residuals', data=res_counter)
            hf.create_dataset('Iterations', data=iter_counter)
            hf.create_dataset('Last_conv', data=last_conv)
            hf.create_dataset('Control_Disp', data=d_c)
            hf.create_dataset('Lambda', data=lam)

            hf.attrs['Descr'] = f'Results of the force_control simulation'
            hf.attrs['Tolerance'] = tol
            hf.attrs['Simulation_Time'] = total_time

    def set_damping_properties(self, xsi=0., damp_type='RAYLEIGH'):

        if isinstance(xsi, float):
            self.xsi = [xsi, xsi]

        elif isinstance(xsi, list) and len(xsi) == 2:
            self.xsi = xsi

        self.damp_type = damp_type

    def get_C_str(self):

        if self.damp_type == 'RAYLEIGH':

            self.get_P_r()
            self.get_K_str()
            self.get_M_str()

            try:
                self.solve_modal(modes=2, save=False)
            except:
                self.solve_modal(save=False)

            if isinstance(self.xsi, float):
                self.xsi = [self.xsi, self.xsi]

            if isinstance(self.xsi, list) and len(self.xsi) == 2:

                A = np.array([[1 / self.eig_vals[0], self.eig_vals[0]],
                              [1 / self.eig_vals[1], self.eig_vals[1]]])

                a = 2 * np.linalg.solve(A, np.array(self.xsi))

            else:
                warnings.warn('Xsi is not a list of two damping ratios for Rayleigh damping')

            self.C = a[0] * self.M + a[1] * self.K

        elif self.damp_type == 'STIFF':

            try:
                self.solve_modal(modes=1, save=False)
            except:
                self.solve_modal(save=False)
            self.get_K_str()
            # print(self.xsi)

            self.C = 2 * self.xsi[0] * self.K / self.eig_vals[0]

        elif self.damp_type == 'MASS':

            try:
                self.solve_modal(modes=1, save=False)
            except:
                self.solve_modal(save=False)
            self.get_M_str()
            self.C = 2 * self.xsi[0] * self.M * self.eig_vals[0]

    def ask_method(self, Meth=None):

        if Meth is None:

            Meth = input('Which method do you want to use ? CDM, CAA, LA, NWK, WIL, HHT, WBZ or GEN - Default is CDM')

            if Meth == 'CDM' or Meth == '':
                return Meth, None
            elif Meth == 'CAA' or Meth == 'NWK':
                return 'NWK', {'g': 1 / 2, 'b': 1 / 4}  # If not specified run CAA by default
            elif Meth == 'LA':
                return 'NWK', {'g': 1 / 2, 'b': 1 / 6}
            elif Meth == 'NWK':

                g = input('Which value for Gamma ? - Default is 1/2')
                b = input('Which value for Beta ? - Default is 1/4')

                if g == '':
                    g = 1 / 2
                else:
                    g = float(g)
                if b == '':
                    b = 1 / 4
                else:
                    b = float(b)

                return 'NWK', {'g': g, 'b': b}

            elif Meth == 'WIL':
                t = input('Which value for Gamma ? - Default is 1.5')
                if t == '':
                    t = 1.5
                else:
                    t = float(t)
                if t < 1:
                    warnings.warn('Theta should be larger or equal to one for Wilson\'s theta method')
                elif t < 1.37:
                    warnings.warn(
                        'Theta should be larger or equal to one for unconditional stability in Wilson\'s theta method')
                return 'WIL', {'t': t}

            elif Meth == 'HHT':

                a = input('Which value for Alpha ? - Default is 1/4')
                g = input('Which value for Gamma ? - Default is (1+2a)/2')
                b = input('Which value for Beta ? - Default is (1+a)^2/4')

                if a == '':
                    a = 1 / 4
                else:
                    a = float(a)
                if a < 0 or a > 1 / 3: warnings.warn(
                    'Alpha should be between 0 and 1/3 for unconditional stability in HHT Method')
                if g == '':
                    g = (1 + 2 * a) / 2
                else:
                    g = float(g)
                if b == '':
                    b = (1 + a) ** 2 / 4
                else:
                    b = float(b)

                return 'GEN', {'am': 0, 'af': a, 'g': g, 'b': b}

            elif Meth == 'WBZ':

                a = input('Which value for Alpha ? - Default is 1/2')
                g = input('Which value for Gamma ? - Default is (1-2a)/2')
                b = input('Which value for Beta ? - Default is 1/4')

                if a == '':
                    a = 1 / 2
                else:
                    a = float(a)
                if a > 1 / 2: warnings.warn(
                    'Alpha should be smaller thann 1/2 for unconditional stability in WBZ Method')
                if g == '':
                    g = (1 - 2 * a) / 2
                else:
                    g = float(g)
                if g < (1 - 2 * a) / 2: warnings.warn(
                    'Gamma should be larger than (1-2a)/2 for unconditional stability in WBZ Method')
                if b == '':
                    b = 1 / 4
                else:
                    b = float(b)
                if b < g / 2: warnings.warn('Beta should be larger than g/2 for unconditional stability in WBZ Method')

                return 'GEN', {'am': a, 'af': 0, 'g': g, 'b': b}

            elif Meth == 'GEN':

                m = input('Which value for Alpha ? - Default is 1')

                if m == '':
                    m = 1
                else:
                    m = float(m)
                if m < 0 or m > 1: warnings.warn('Mu should be between 0 and 1 for Generalized-alpha Method')

                return 'GEN', {'am': (2 * m - 1) / (m + 1), 'af': m / (m + 1), 'g': (3 * m - 1) / (2 * (m + 1)),
                               'b': (m / (m + 1)) ** 2}

        elif isinstance(Meth, str):

            if Meth == 'CDM':
                return Meth, {}
            elif Meth == 'CAA' or Meth == 'NWK':
                return 'NWK', {'g': 1 / 2, 'b': 1 / 4}  # If not specified run CAA by default
            elif Meth == 'LA':
                return 'GEN', {'am': 0, 'af': 0, 'g': 1 / 2, 'b': 1 / 6}
            elif Meth == 'NWK':
                return 'GEN', {'am': 0, 'af': 0, 'g': 1 / 2, 'b': 1 / 4}
            elif Meth == 'WIL':
                return 'WIL', {'t': 1.5}
            elif Meth == 'HHT':
                return 'GEN', {'am': 0, 'af': 0, 'g': 1 / 2, 'b': 1 / 4}
            elif Meth == 'WBZ':
                return 'GEN', {'am': 0, 'af': 0, 'g': 1 / 2, 'b': 1 / 4}
            elif Meth == 'GEN':
                m = 1
                return 'GEN', {'am': (2 * m - 1) / (m + 1), 'af': m / (m + 1), 'g': (3 * m - 1) / (2 * (m + 1)),
                               'b': (m / (m + 1)) ** 2}

        elif isinstance(Meth, list):

            if Meth[0] == 'NWK':

                if len(Meth) != 3: warnings.warn('Requiring 2 parameters for Newmark method')

                g = Meth[1]
                b = Meth[2]

                return 'GEN', {'am': 0, 'af': 0, 'g': g, 'b': b}

            elif Meth[0] == 'WIL':

                if len(Meth) != 2: warnings.warn('Requiring 1 parameters for Wilson\'s theta method')

                t = Meth[1]
                if t < 1:
                    warnings.warn('Theta should be larger or equal to one for Wilson\'s theta method')
                elif t < 1.37:
                    warnings.warn(
                        'Theta should be larger or equal to one for unconditional stability in Wilson\'s theta method')
                return 'WIL', {'t': t}

            elif Meth[0] == 'HHT':

                if len(Meth) == 2:
                    a = Meth[1]
                    g = (1 + 2 * a) / 2
                    b = (1 + a) ** 2 / 4

                elif len(Meth) == 4:

                    a = Meth[1]
                    g = Meth[2]
                    b = Meth[3]

                else:
                    warnings.warn('Requiring 3 parameters for HHT method')

                if a < 0 or a > 1 / 3: warnings.warn(
                    'Alpha should be between 0 and 1/3 for unconditional stability in HHT Method')

                return 'GEN', {'am': 0, 'af': a, 'g': g, 'b': b}

            elif Meth[0] == 'WBZ':

                if len(Meth) != 4: warnings.warn('Requiring 3 parameters for WBZ method')

                a = Meth[1]
                g = Meth[2]
                b = Meth[3]

                if a > 1 / 2: warnings.warn(
                    'Alpha should be smaller thann 1/2 for unconditional stability in WBZ Method')
                if g < (1 - 2 * a) / 2: warnings.warn(
                    'Gamma should be larger than (1-2a)/2 for unconditional stability in WBZ Method')
                if b < g / 2: warnings.warn('Beta should be larger than g/2 for unconditional stability in WBZ Method')

                return 'GEN', {'am': a, 'af': 0, 'g': g, 'b': b}

            elif Meth[0] == 'GEN':

                if len(Meth) != 2: warnings.warn('Requiring 1 parameters for Generalized Alpha method')

                m = Meth[1]

                if m < 0 or m > 1: warnings.warn('Mu should be between 0 and 1 for Generalized-alpha Method')

                return 'GEN', {'am': (2 * m - 1) / (m + 1), 'af': m / (m + 1), 'g': (3 * m - 1) / (2 * (m + 1)),
                               'b': (m / (m + 1)) ** 2}

        return None, None

    def solve_dyn_linear(self, T, dt, U0=None, V0=None, lmbda=None, Meth=None, filename='', dir_name=''):

        time_start = time.time()

        self.get_K_str0()
        self.get_M_str()
        self.get_C_str()

        if U0 is None: U0 = np.zeros(self.nb_dofs)
        if V0 is None: V0 = np.zeros(self.nb_dofs)

        Time = np.arange(0, T, dt, dtype=float)
        Time = np.append(Time, T)
        nb_steps = len(Time)

        loading = np.zeros(nb_steps)

        if callable(lmbda):
            for i, t in enumerate(Time):
                loading[i] = lmbda(t)
        elif isinstance(lmbda, list):
            pass

        U_conv = np.zeros((self.nb_dofs, nb_steps))
        V_conv = np.zeros((self.nb_dofs, nb_steps))
        A_conv = np.zeros((self.nb_dofs, nb_steps))
        P_conv = np.zeros((self.nb_dofs, nb_steps))

        U_conv[:, 0] = U0.copy()
        V_conv[:, 0] = V0.copy()
        A_conv[:, 0] = np.linalg.solve(self.M, loading[0] * self.P - self.C @ V_conv[:, 0] - self.K0 @ U_conv[:, 0])

        Meth, P = self.ask_method(Meth)

        if Meth == 'CDM':

            U_conv[:, -1] = U_conv[:, 0] - dt * V_conv[:, 0] + dt ** 2 * A_conv[:, 0] / 2

            K_h = self.M / dt ** 2 + self.C / (2 * dt)
            a = self.M / dt ** 2 - self.C / (2 * dt)
            b = self.K0 - 2 * self.M / dt ** 2

            for i in np.arange(1, nb_steps):
                P_h = loading[i - 1] * self.P + self.P_fixed - a @ U_conv[:, i - 2] - b @ U_conv[:, i - 1]
                U_conv[self.dof_free, i] = (np.linalg.solve(K_h, P_h))[self.dof_free]
                V_conv[self.dof_free, i] = (U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]) / (2 * dt)
                A_conv[self.dof_free, i] = (U_conv[self.dof_free, i] - 2 * U_conv[self.dof_free, i - 1] + U_conv[
                    self.dof_free, i - 2]) / (dt ** 2)
                P_conv[:, i] = P_h.copy()

        elif Meth == 'NWK':

            A1 = self.M / (P['b'] * dt ** 2) + P['g'] * self.C / (P['b'] * dt)
            A2 = self.M / (P['b'] * dt) + (P['g'] / P['b'] - 1) * self.C
            A3 = (1 / (2 * P['b']) - 1) * self.M + dt * (P['g'] / (2 * P['b']) - 1) * self.C

            K_h = self.K0 + A1

            for i in np.arange(1, nb_steps):
                P_h = loading[i] * self.P + A1 @ U_conv[:, i - 1] + A2 @ V_conv[:, i - 1] + A3 @ A_conv[:, i - 1]
                U_conv[self.dof_free, i] = np.linalg.solve(K_h, P_h)[self.dof_free]
                V_conv[self.dof_free, i] = (P['g'] / (P['b'] * dt)) * (
                            U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]) + (1 - P['g'] / P['b']) * V_conv[
                                               self.dof_free, i - 1] + dt * (1 - P['g'] / (2 * P['b'])) * A_conv[
                                               self.dof_free, i - 1]
                A_conv[self.dof_free, i] = (1 / (P['b'] * dt ** 2)) * (
                            U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]) - V_conv[self.dof_free, i - 1] / (
                                                       P['b'] * dt) - (1 / (2 * P['b']) - 1) * A_conv[
                                               self.dof_free, i - 1]
                P_conv[:, i] = self.K0

        elif Meth == 'WIL':

            A1 = 6 / (P['t'] * dt) * self.M + 3 * self.C
            A2 = 3 * self.M + P['t'] * dt / 2 * self.C

            K_h = self.K0 + 6 / (P['t'] * dt) ** 2 * self.M + 3 / (P['t'] * dt) * self.C

            loading = np.append(loading, loading[-1])

            for i in np.arange(1, nb_steps):
                dp_h = ((P['t'] - 1) * (loading[i + 1] - loading[i]) + loading[i] - loading[i - 1]) * self.P

                dp_h += A1 @ V_conv[:, i - 1] + A2 @ A_conv[:, i - 1]

                d_Uh = (np.linalg.solve(K_h, dp_h))

                d_A = (6 / (P['t'] * dt) ** 2 * d_Uh - 6 / (P['t'] * dt) * V_conv[:, i - 1] - 3 * A_conv[:, i - 1]) / (
                P['t'])

                d_V = dt * A_conv[:, i - 1] + dt / 2 * d_A
                d_U = dt * V_conv[:, i - 1] + (dt ** 2) / 2 * A_conv[:, i - 1] + (dt ** 2) / 6 * d_A

                U_conv[self.dof_free, i] = (U_conv[:, i - 1] + d_U)[self.dof_free]
                V_conv[self.dof_free, i] = (V_conv[:, i - 1] + d_V)[self.dof_free]
                A_conv[self.dof_free, i] = (A_conv[:, i - 1] + d_A)[self.dof_free]

        elif Meth == 'GEN':

            A1 = (1 - P['am']) / (P['b'] * dt ** 2) * self.M + P['g'] * (1 + P['af']) / (P['b'] * dt) * self.C
            A2 = (1 - P['am']) / (P['b'] * dt) * self.M + (P['g'] + P['g'] * P['af'] - P['b']) / P['b'] * self.C
            A3 = (1 - P['am'] - 2 * P['b']) / (2 * P['b']) * self.M + dt * (1 + P['af']) * (P['g'] - 2 * P['b']) / (
                        2 * P['b']) * self.C

            K_h = self.K0 * (1 + P['af']) + A1

            for i in np.arange(1, nb_steps):
                P_h = loading[i] * self.P + A1 @ U_conv[:, i - 1] + A2 @ V_conv[:, i - 1] + A3 @ A_conv[:, i - 1] + P[
                    'af'] * self.K0 @ U_conv[:, i - 1]

                U_conv[:, i][self.dof_free] = (np.linalg.solve(K_h, P_h))[self.dof_free]
                V_conv[:, i][self.dof_free] = (
                            P['g'] / (P['b'] * dt) * (U_conv[:, i] - U_conv[:, i - 1]) + (1 - P['g'] / P['b']) * V_conv[
                                                                                                                 :,
                                                                                                                 i - 1] + dt * (
                                        1 - P['g'] / (2 * P['b'])) * A_conv[:, i - 1])[self.dof_free]
                A_conv[:, i][self.dof_free] = (
                            1 / (P['b'] * dt ** 2) * (U_conv[:, i] - U_conv[:, i - 1]) - 1 / (dt * P['b']) * V_conv[:,
                                                                                                             i - 1] - (
                                        1 / (2 * P['b']) - 1) * A_conv[:, i - 1])[self.dof_free]
                P_conv[:, i] = P_h.copy()


        elif Meth is None:
            print('Method does not exist')

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f'Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file')

        Params = []
        for key, value in P.items():
            Params.append(f'{key}={np.around(value, 2)}')

        Params = "_".join(Params)

        filename = filename + '_' + Meth + '_' + Params + '.h5'
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, 'w') as hf:

            hf.create_dataset('U_conv', data=U_conv)
            hf.create_dataset('V_conv', data=V_conv)
            hf.create_dataset('A_conv', data=A_conv)
            hf.create_dataset('P_ref', data=self.P)
            hf.create_dataset('P_conv', data=P_conv)
            hf.create_dataset('Load_Multiplier', data=loading)
            hf.create_dataset('Time', data=Time)
            hf.create_dataset('Last_conv', data=nb_steps - 1)

            hf.attrs['Descr'] = 'Results of the' + Meth + 'simulation'
            hf.attrs['Method'] = Meth

    def solve_dyn_nonlinear(self, T, dt, U0=None, V0=None, lmbda=None, Meth=None, filename='', dir_name=''):

        time_start = time.time()

        if U0 is None: U0 = np.zeros(self.nb_dofs)
        if V0 is None: V0 = np.zeros(self.nb_dofs)

        Time = np.arange(0, T, dt, dtype=float)
        Time = np.append(Time, T)
        nb_steps = len(Time)

        loading = np.zeros(nb_steps)

        if callable(lmbda):
            for i, t in enumerate(Time):
                loading[i] = lmbda(t)
        elif isinstance(lmbda, list):
            pass

        self.get_P_r()
        self.get_K_str()
        self.get_M_str()
        self.get_C_str()

        U_conv = np.zeros((self.nb_dofs, nb_steps))
        V_conv = np.zeros((self.nb_dofs, nb_steps))
        A_conv = np.zeros((self.nb_dofs, nb_steps))
        F_conv = np.zeros((self.nb_dofs, nb_steps))

        U_conv[:, 0] = U0.copy()
        V_conv[:, 0] = V0.copy()

        last_sec = 0

        Meth, P = self.ask_method(Meth)

        if Meth == 'CDM':

            self.U = U_conv[:, 0].copy()
            self.get_P_r()
            F_conv[:, 0] = self.P_r.copy()

            A_conv[:, 0] = np.linalg.solve(self.M,
                                           loading[0] * self.P + self.P_fixed - self.C @ V_conv[:, 0] - F_conv[:, 0])

            U_conv[:, -1] = U_conv[:, 0] - dt * V_conv[:, 0] + dt ** 2 / 2 * A_conv[:, 0]

            K_h = 1 / (dt ** 2) * self.M + 1 / (2 * dt) * self.C
            A = 1 / (dt ** 2) * self.M - 1 / (2 * dt) * self.C
            B = 2 / (dt ** 2) * self.M

            for i in np.arange(1, nb_steps):

                self.U = U_conv[:, i - 1].copy()
                try:
                    self.get_P_r()
                except Exception as e:
                    print(e)
                    break

                F_conv[:, i - 1] = self.P_r.copy()

                P_h = loading[i - 1] * self.P + self.P_fixed - A @ U_conv[:, i - 2] + B @ U_conv[:, i - 1] - F_conv[:,
                                                                                                             i - 1]

                U_conv[self.dof_free, i] = np.linalg.solve(K_h, P_h)[self.dof_free]

                V_conv[self.dof_free, i] = ((U_conv[:, i] - U_conv[:, i - 2]) / (2 * dt))[self.dof_free]
                A_conv[self.dof_free, i] = ((U_conv[:, i] - 2 * U_conv[:, i - 1] + U_conv[:, i - 2]) / (dt ** 2))[
                    self.dof_free]

                if int(i * dt) >= last_sec:
                    print(f'reached {last_sec} seconds out of {int(Time[-1])} seconds')
                    last_sec += 1
                # print(U_conv[self.dof_free,i])

                last_conv = i

        elif Meth == 'NWK':

            pass

        elif Meth == 'WIL':

            pass

        elif Meth == 'GEN':

            self.U = U_conv[:, 0].copy()
            self.get_P_r()
            F_conv[:, 0] = self.P_r.copy()

            A_conv[:, 0] = np.linalg.solve(self.M,
                                           loading[0] * self.P + self.P_fixed - self.C @ V_conv[:, 0] - F_conv[:, 0])

            A1 = (1 - P['am']) * self.M / (P['b'] * dt ** 2) + P['g'] * (1 + P['af']) * self.C / (P['b'] * dt)
            A2 = (1 - P['am']) * self.M / (P['b'] * dt) + (P['g'] + P['g'] * P['af'] - P['b']) * self.C / (P['b'])
            A3 = ((1 - P['am']) / (2 * P['b']) - 1) * self.M + dt * (1 + P['af']) * (
                        P['g'] / (P['b'] * dt) - 1) * self.C

            no_conv = 0

            for i in np.arange(1, nb_steps):

                self.U = U_conv[:, i - 1].copy()

                try:
                    self.get_P_r()
                except Exception as e:
                    print(e)
                    break

                F_conv[:, i] = self.P_r.copy()

                R = np.zeros(self.nb_dofs)
                P_h = loading[i] * self.P + self.P_fixed + A1 @ self.U + A2 @ V_conv[:, i - 1] + A3 @ A_conv[:, i - 1] + \
                      P['af'] * F_conv[:, i]

                counter = 0
                conv = False

                while not conv:

                    if counter > 100:
                        no_conv = i
                        break

                    R[self.dof_free] = (P_h - A1 @ self.U - (1 + P['af']) * F_conv[:, i])[self.dof_free]

                    if np.linalg.norm(R) < 1:
                        U_conv[:, i] = self.U.copy()
                        conv = True
                        last_conv = i

                    self.get_K_str()
                    Kt_p = self.K * (1 + P['af']) + A1

                    self.U[self.dof_free] = (self.U + np.linalg.solve(Kt_p, R))[self.dof_free]

                    try:
                        self.get_P_r()
                    except Exception as e:
                        no_conv = i
                        print(f'Error occured at step {i}')
                        print(e)
                        break
                    F_conv[:, i] = self.P_r.copy()

                    counter += 1

                if no_conv > 0:
                    print(f'Step {no_conv} did not converge')
                    break

                V_conv[self.dof_free, i] = ((P['g'] / (P['b'] * dt)) * (U_conv[:, i] - U_conv[:, i - 1]) + (
                            1 - P['g'] / P['b']) * V_conv[:, i - 1] + dt * (1 - P['g'] / (2 * P['b'])) * A_conv[:,
                                                                                                         i - 1])[
                    self.dof_free]
                A_conv[self.dof_free, i] = (
                            1 / (P['b'] * dt ** 2) * (U_conv[:, i] - U_conv[:, i - 1]) - (1 / (dt * P['b'])) * V_conv[:,
                                                                                                               i - 1] - (
                                        1 / (2 * P['b']) - 1) * A_conv[:, i - 1])[self.dof_free]

                if int(i * dt) >= last_sec:
                    print(f'reached {last_sec} seconds out of {int(Time[-1])} seconds')
                    last_sec += 1

        elif Meth is None:
            print('Method does not exist')

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f'Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file')

        Params = []
        for key, value in P.items():
            Params.append(f'{key}={np.around(value, 2)}')

        Params = "_".join(Params)

        filename = filename + '_' + Meth + '_' + Params + '.h5'
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, 'w') as hf:

            hf.create_dataset('U_conv', data=U_conv)
            hf.create_dataset('V_conv', data=V_conv)
            hf.create_dataset('A_conv', data=A_conv)
            hf.create_dataset('P_ref', data=self.P)
            hf.create_dataset('Load_Multiplier', data=loading)
            hf.create_dataset('Time', data=Time)
            hf.create_dataset('Last_conv', data=last_conv)

            hf.attrs['Descr'] = f'Results of the' + Meth + 'simulation'
            hf.attrs['Method'] = Meth

    def save_structure(self, filename):

        import pickle

        with open(filename + '.pkl', 'wb') as file:
            pickle.dump(self, file)

    def solve_modal(self, modes=None, no_inertia=False, filename='Results_Modal', dir_name='', save=True):

        time_start = time.time()

        try:
            self.get_K_str()
        except:
            self.get_P_r()
            self.get_K_str()
        self.get_M_str(no_inertia=no_inertia)

        if modes is None:
            omega, phi = sc.linalg.eigh(self.K[np.ix_(self.dof_free, self.dof_free)],
                                        self.M[np.ix_(self.dof_free, self.dof_free)])
        elif isinstance(modes, int):
            if np.linalg.det(self.M) == 0: warnings.warn(
                'Might need to use linalg.eig if the matrix M is non-invertible')
            omega, phi = sc.sparse.linalg.eigsh(self.K[np.ix_(self.dof_free, self.dof_free)], modes,
                                                self.M[np.ix_(self.dof_free, self.dof_free)], which='SM')

        else:
            warnings.warn("Required modes should be either int or None")

        self.eig_vals = np.sort(np.real(np.sqrt(omega))).copy()
        self.eig_modes = (np.real(phi).T)[np.argsort((np.sqrt(omega)))].T.copy()

        if save:
            time_end = time.time()
            total_time = time_end - time_start
            print('Simulation done... writing results to file')

            filename = filename + '.h5'
            file_path = os.path.join(dir_name, filename)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('eig_vals', data=self.eig_vals)
                hf.create_dataset('eig_modes', data=self.eig_modes)

                hf.attrs['Simulation_Time'] = total_time

    def detect_interfaces(self):

        self.interf_counter = 0

        def detect_overlap(pair1, pair2, factor=None):

            if not factor == 1:
                pair1[0][1] *= factor
                pair1[1][1] *= factor
                pair2[0][1] *= factor
                pair2[1][1] *= factor

            sorted1 = sorted(pair1, key=lambda point: point[0] + point[1])
            sorted2 = sorted(pair2, key=lambda point: point[0] + point[1])

            if np.sum(sorted1[0]) - np.sum(sorted2[1]) >= -1e-8 or np.sum(sorted2[0]) - np.sum(sorted1[1]) >= -1e-8:
                return False, None

            else:
                edge1 = sorted([sorted1[0], sorted2[0]], key=lambda point: point[0] + point[1])
                edge2 = sorted([sorted1[1], sorted2[1]], key=lambda point: point[0] + point[1])

                if not factor == 1:
                    edge1[0][1] /= factor
                    edge1[1][1] /= factor
                    edge2[0][1] /= factor
                    edge2[1][1] /= factor

                return True, [edge1[1], edge2[0]]

        def detect_interface_2blocks(cand, anta):

            interfaces = []

            triplets_cand = cand.compute_triplets()
            triplets_anta = anta.compute_triplets()

            for triplet1 in triplets_cand:
                for triplet2 in triplets_anta:

                    if np.all(np.isclose(triplet1['ABC'], triplet2['ABC'], rtol=1e-8)):

                        # Handle the case where x+y = constant
                        if triplet1['ABC'][0] == triplet1['ABC'][1]:
                            factor = 2
                        else:
                            factor = 1
                        overlap, nodes = detect_overlap(triplet1['Vertices'], triplet2['Vertices'], factor=factor)

                        if overlap:
                            interface = {}
                            unit_vector = (nodes[1] - nodes[0]) / np.linalg.norm(nodes[1] - nodes[0])
                            normal_vector = np.array([[0, -1], [1, 0]]) @ unit_vector
                            if np.dot(cand.ref_point - nodes[0], normal_vector) > 0:
                                interface['Block A'] = cand
                                interface['Block B'] = anta
                            else:
                                interface['Block A'] = anta
                                interface['Block B'] = cand
                            interface['x_e1'] = nodes[0]
                            interface['x_e2'] = nodes[1]

                            interfaces.append(interface)

            if len(interfaces) == 0:
                return False, None
            else:
                return True, interfaces

        def distance(a, b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        interfaces = []

        for i, cand in enumerate(self.list_blocks):

            for j, anta in enumerate(self.list_blocks[i + 1:]):

                # Check if blocks have same dofs
                if cand.connect == anta.connect: continue
                # Check if influence circles intersect 
                if distance(cand.circle_center, anta.circle_center) >= (
                        cand.circle_radius + anta.circle_radius) * 1.01: continue

                contact, interface = detect_interface_2blocks(cand, anta)
                self.interf_counter += 1

                if not contact: continue

                interfaces.extend(interface)

        print(f'{len(interfaces)} interface{"s" if len(interfaces) != 1 else ""} detected')

        return interfaces

    def plot_stiffness(self, save=None):

        E = []
        vertices = []

        for j, CF in enumerate(self.list_cfs):

            for i, CP in enumerate(CF.cps):
                E.append(np.around(CP.sp1.law.stiff['E'], 3))
                E.append(np.around(CP.sp2.law.stiff['E'], 3))
                vertices.append(CP.vertices_fibA)
                vertices.append(CP.vertices_fibB)

        from matplotlib.colors import Normalize
        from matplotlib import cm

        def normalize(smax, smin):

            if (smax - smin) == 0 and smax < 0:
                return Normalize(vmin=1.1 * smin / 1e9, vmax=0.9 * smax / 1e9, clip=False)
            elif (smax - smin) == 0 and smax == 0:
                return Normalize(vmin=-1e-6, vmax=1e-6, clip=False)
            elif (smax - smin) == 0:
                return Normalize(vmin=0.9 * smin / 1e9, vmax=1.1 * smax / 1e9, clip=False)
            else:
                return Normalize(vmin=smin / 1e9, vmax=smax / 1e9, clip=False)

        def plot(stiff, vertex):
            smax = np.max(stiff)
            smin = np.min(stiff)

            plt.axis('equal')
            plt.axis('off')
            plt.title(f"Axial stiffness [GPa]")

            norm = normalize(smax, smin)
            cmap = cm.get_cmap('coolwarm', 200)

            for i in range(len(stiff)):
                if smax - smin == 0:
                    index = norm(np.around(stiff[i], 6) / 1e9)
                else:
                    index = norm(np.around(stiff[i], 6) / 1e9)
                color = cmap(index)
                vertices_x = np.append(vertex[i][:, 0], vertex[i][0, 0])
                vertices_y = np.append(vertex[i][:, 1], vertex[i][0, 1])
                plt.fill(vertices_x, vertices_y, color=color)

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(plt.gca())

            cax = divider.append_axes("right", size="10%", pad=0.2)
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

        plt.figure()

        plot(E, vertices)

        if save is not None: plt.savefig(save)

    def plot_stresses(self, angle=None, save=None):

        # Compute maximal stress and minimal stress: 

        tau = []
        sigma = []
        vertices = []

        for j, CF in enumerate(self.list_cfs):

            if (angle is None) or (abs(CF.angle - angle) < 1e-6):

                for i, CP in enumerate(CF.cps):
                    tau.append(np.around(CP.sp1.law.stress['t'], 3))
                    tau.append(np.around(CP.sp2.law.stress['t'], 3))
                    sigma.append(np.around(CP.sp1.law.stress['s'], 3))
                    sigma.append(np.around(CP.sp2.law.stress['s'], 3))
                    vertices.append(CP.vertices_fibA)
                    vertices.append(CP.vertices_fibB)

        from matplotlib.colors import Normalize
        from matplotlib import cm

        def normalize(smax, smin):

            if (smax - smin) == 0 and smax < 0:
                return Normalize(vmin=1.1 * smin / 1e6, vmax=0.9 * smax / 1e6, clip=False)
            elif (smax - smin) == 0 and smax == 0:
                return Normalize(vmin=-1e-6, vmax=1e-6, clip=False)
            elif (smax - smin) == 0:
                return Normalize(vmin=0.9 * smin / 1e6, vmax=1.1 * smax / 1e6, clip=False)
            else:
                return Normalize(vmin=smin / 1e6, vmax=smax / 1e6, clip=False)

        def plot(stress, vertex, name_stress=None):
            smax = np.max(stress)
            smin = np.min(stress)

            print(f"Maximal {'axial' if name_stress == 'sigma' else 'shear'} stress is {np.around(smax / 1e6, 3)} MPa")
            print(f"Minimum {'axial' if name_stress == 'sigma' else 'shear'} stress is {np.around(smin / 1e6, 3)} MPa")
            # Plot sigmas 

            plt.axis('equal')
            plt.axis('off')
            plt.title(f"{'Axial' if name_stress == 'sigma' else 'Shear'} stresses [MPa]")

            norm = normalize(smax, smin)
            cmap = cm.get_cmap('coolwarm', 200)

            for i in range(len(sigma)):
                if smax - smin == 0:
                    index = norm(np.around(stress[i], 6) / 1e6)
                else:
                    index = norm(np.around(stress[i], 6) / 1e6)
                color = cmap(index)
                vertices_x = np.append(vertex[i][:, 0], vertex[i][0, 0])
                vertices_y = np.append(vertex[i][:, 1], vertex[i][0, 1])
                plt.fill(vertices_x, vertices_y, color=color)

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(plt.gca())

            cax = divider.append_axes("right", size="10%", pad=0.2)
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

        plt.figure()

        plt.subplot(2, 1, 1)
        plot(sigma, vertices, name_stress='sigma')
        plt.subplot(2, 1, 2)
        plot(tau, vertices, name_stress='tau')

        if save is not None: plt.savefig(save)

    def plot_modes(self, modes=None, scale=1, save=False, lims=None, folder=None, show=True):

        if not hasattr(self, 'eig_modes'): warnings.warn('Eigen modes were not determined yet')

        if modes is None:
            modes = self.nb_dof_free

        if len(self.eig_vals) < modes: warnings.warn('Asking for too many modes, fewer were computed')

        for i in range(modes):

            self.U[self.dof_free] = scale * self.eig_modes.T[i]

            self.plot_structure(scale=1, plot_cf=False, plot_forces=False, plot_supp=False, lims=lims, show=show)

            w = np.around(self.eig_vals[i], 3)
            if not w == 0:
                T = np.around(2 * np.pi / w, 3)
            else:
                T = float('inf')
            plt.title(fr'Mode {i + 1} with $\omega_{{{i + 1}}} = {w}$ rad/s - $T_{{{i + 1}}} = {T}$ s ')
            if save:
                if folder is not None:
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    plt.savefig(folder + f'/Mode_{i + 1}.eps')
                else:
                    plt.savefig(f'Mode_{i + 1}.pdf')

            plt.close()

    def plot_structure(self, scale=0, plot_cf=True, plot_forces=True, plot_supp=True, show=True, save=None, lims=None):

        if lims is None:
            plt.figure(None, dpi=400, figsize=(6, 6))
        else:
            x_len = lims[0][1] - lims[0][0]
            y_len = lims[1][1] - lims[1][0]
            if x_len > y_len:
                plt.figure(None, dpi=400, figsize=(6, 6 * y_len / x_len))
            else:
                plt.figure(None, dpi=400, figsize=(6 * x_len / y_len, 6))

        plt.axis('equal')
        plt.axis('off')

        for bl in self.list_blocks:
            bl.disps = self.U[bl.dofs]
            bl.plot_block(scale=scale)

        for FE in self.list_fes:
            if scale == 0:
                FE.PlotUndefShapeElem()
            else:
                defs = self.U[FE.dofs]
                FE.PlotDefShapeElem(defs, scale=scale)

        if plot_cf:
            for cf in self.list_cfs:
                cf.plot_cf(scale)

        if plot_forces:
            for i in self.dof_free:

                if self.P[i] != 0:
                    node_id = int(i / 3)
                    dof = i % 3

                    start = self.list_nodes[node_id] + scale * self.U[
                        3 * node_id * np.ones(2, dtype=int) + np.array([0, 1], dtype=int)]
                    arr_len = .3

                    if dof == 0:
                        end = arr_len * np.array([1, 0]) * np.sign(self.P[i])
                        plt.arrow(start[0], start[1], end[0], end[1], head_width=0.05, head_length=0.075, fc='green',
                                  ec='green')
                    elif dof == 1:
                        end = arr_len * np.array([0, 1]) * np.sign(self.P[i])
                        plt.arrow(start[0], start[1], end[0], end[1], head_width=0.05, head_length=0.075, fc='green',
                                  ec='green')
                    else:
                        if np.sign(self.P[i]) == 1:
                            plt.plot(start[0], start[1], marker='o', markerfacecolor='None', markeredgecolor='green',
                                     markersize=10)
                            plt.plot(start[0], start[1], marker='.', markerfacecolor='green', markeredgecolor='green',
                                     markersize=5)
                        else:
                            plt.plot(start[0], start[1], marker='o', markerfacecolor='None', markeredgecolor='green',
                                     markersize=10)
                            plt.plot(start[0], start[1], marker='x', markerfacecolor='None', markeredgecolor='green',
                                     markersize=10)

                if self.P_fixed[i] != 0:

                    node_id = int(i / 3)
                    dof = i % 3

                    start = self.list_nodes[node_id] + scale * self.U[
                        3 * node_id * np.ones(2, dtype=int) + np.array([0, 1], dtype=int)]
                    arr_len = .3

                    if dof == 0:
                        end = arr_len * np.array([1, 0]) * np.sign(self.P_fixed[i])
                        plt.arrow(start[0], start[1], end[0], end[1], head_width=0.05, head_length=0.075, fc='red',
                                  ec='red')
                    elif dof == 1:
                        end = arr_len * np.array([0, 1]) * np.sign(self.P_fixed[i])
                        plt.arrow(start[0], start[1], end[0], end[1], head_width=0.05, head_length=0.075, fc='red',
                                  ec='red')
                    else:
                        if np.sign(self.P_fixed[i]) == 1:
                            plt.plot(start[0], start[1], marker='o', markerfacecolor='None', markeredgecolor='red',
                                     markersize=10)
                            plt.plot(start[0], start[1], marker='.', markerfacecolor='red', markeredgecolor='red',
                                     markersize=5)
                        else:
                            plt.plot(start[0], start[1], marker='o', markerfacecolor='None', markeredgecolor='red',
                                     markersize=10)
                            plt.plot(start[0], start[1], marker='x', markerfacecolor='None', markeredgecolor='red',
                                     markersize=10)

        if plot_supp:

            for fix in self.dof_fix:

                node_id = int(fix / 3)
                dof = fix % 3

                node = self.list_nodes[node_id] + scale * self.U[
                    3 * node_id * np.ones(2, dtype=int) + np.array([0, 1], dtype=int)]

                import matplotlib as mpl

                if dof == 0:
                    mark = mpl.markers.MarkerStyle(marker=5)
                elif dof == 1:
                    mark = mpl.markers.MarkerStyle(marker=6)
                else:
                    mark = mpl.markers.MarkerStyle(marker="x")

                plt.plot(node[0], node[1], marker=mark, color='blue', markersize=8)

        if lims is not None:
            plt.xlim(lims[0][0], lims[0][1])
            plt.ylim(lims[1][0], lims[1][1])

        if save is not None: plt.savefig(save)

        if not show: plt.close()
