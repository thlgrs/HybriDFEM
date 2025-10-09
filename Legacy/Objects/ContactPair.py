# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:05:21 2024

@author: ibouckaert
"""

# warnings.filterwarnings("error")
import os
import warnings
from copy import deepcopy
from warnings import warn

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.optimize import root

import Spring as sp


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"


warnings.formatwarning = custom_warning_format


def T2x2(a): return np.array([[np.cos(a), -np.sin(a)],
                              [np.sin(a), np.cos(a)]])


class CP_2D:

    def __init__(self, x_cp, l_Ax, l_Ay, l_Bx, l_By, angle, h_cp, b, block_A=None, block_B=None, contact=None,
                 surface=None, material=None, lin_geom=True, reinf=False):

        self.x_cp = deepcopy(x_cp)
        self.angle = angle
        self.h = h_cp
        self.b = b

        self.long = np.around(np.array([np.cos(self.angle), np.sin(self.angle)]), 10)
        self.tran = np.around(np.array([-np.sin(self.angle), np.cos(self.angle)]), 10)

        self.l_Ax = l_Ax
        self.l_Ay = l_Ay
        self.l_Bx = l_Bx
        self.l_By = l_By

        self.lin_geom = lin_geom
        self.reinf = reinf

        self.vertices_fibA = np.array([self.x_cp, self.x_cp, self.x_cp, self.x_cp])
        self.vertices_fibB = np.array([self.x_cp, self.x_cp, self.x_cp, self.x_cp])

        self.vertices_fibA[0] += - self.h / 2 * self.tran
        self.vertices_fibA[1] += self.h / 2 * self.tran
        self.vertices_fibA[2] += self.h / 2 * self.tran - self.l_Ax * self.long
        self.vertices_fibA[3] += - self.h / 2 * self.tran - self.l_Ax * self.long

        self.vertices_fibB[0] += - self.h / 2 * self.tran + self.l_Bx * self.long
        self.vertices_fibB[1] += self.h / 2 * self.tran + self.l_Bx * self.long
        self.vertices_fibB[2] += self.h / 2 * self.tran
        self.vertices_fibB[3] += - self.h / 2 * self.tran

        if l_Ax < 0 or l_Bx < 0:
            warn('l_Ax and l_Bx should be positive')
            print(self.x_cp, l_Ax, l_Bx)

        if (contact is None) and (surface is None) and (material is None):
            if block_A is None and block_B is None: warn('Must refer to a block or define contact/surface law')
            self.bl_A = block_A
            self.bl_B = block_B

            self.sp1 = sp.Spring_2D(self.l_Ax, self.l_Ay, self.h, self.b, block=self.bl_A)
            self.sp2 = sp.Spring_2D(self.l_Bx, self.l_By, self.h, self.b, block=self.bl_B)

        elif self.reinf:

            self.sp1 = sp.Spring_2D(self.l_Ax, self.l_Ay, self.h, self.b, material=material)
            self.sp2 = sp.Spring_2D(self.l_Bx, self.l_By, self.h, self.b, material=material)


        else:

            self.sp1 = sp.Spring_2D(self.l_Ax, self.l_Ay, self.h, self.b, block=None, contact=contact, surface=surface)
            self.sp2 = sp.Spring_2D(self.l_Bx, self.l_By, self.h, self.b, block=None, contact=contact, surface=surface)

    def commit(self):
        self.sp1.commit()
        self.sp2.commit()

    def revert_commit(self):
        self.sp1.revert_commit()
        self.sp2.revert_commit()

    def get_pc_loc(self, qf_loc):

        # print('q_loc', qf_loc)
        # print('qf_loc', qf_loc)
        self.qc_loc = qf_loc.copy()

        # print('qc_loc', self.qc_loc)
        self.get_q_c()
        self.get_p_c()
        self.pc_loc = np.transpose(self.Gamma) @ self.p_c

        # print('pc_loc', self.pc_loc)

        return self.pc_loc

    def get_q_c(self):

        self.q_c = np.zeros(8)
        T = T2x2(self.angle)

        (L_Ax, L_Ay) = T @ np.array([self.l_Ax, self.l_Ay])
        (L_Bx, L_By) = np.transpose(T) @ np.array([self.l_Bx, self.l_By])

        c = np.cos(self.angle)
        s = np.sin(self.angle)

        if not self.lin_geom:

            c_q3 = np.cos(self.qc_loc[2])
            s_q3 = np.sin(self.qc_loc[2])
            c_q6 = np.cos(self.qc_loc[5])
            s_q6 = np.sin(self.qc_loc[5])

            # #
            self.Gamma = np.array([[c, -s, -L_Ay * c_q3 - L_Ax * s_q3, 0, 0, 0],
                                   [s, c, L_Ax * c_q3 - L_Ay * s_q3, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, c, -s, -L_By * c_q6 + L_Bx * s_q6],
                                   [0, 0, 0, s, c, -L_Bx * c_q6 - L_By * s_q6],
                                   [0, 0, 0, 0, 0, 1]])

            self.q_c[:3] = np.array([self.qc_loc[0] * c - self.qc_loc[1] * s - L_Ay * s_q3 - L_Ax * (1 - c_q3),
                                     self.qc_loc[0] * s + self.qc_loc[1] * c + L_Ax * s_q3 - L_Ay * (1 - c_q3),
                                     self.qc_loc[2]])

            self.q_c[3:6] = np.array([self.qc_loc[3] * c - self.qc_loc[4] * s - L_By * s_q6 + L_Bx * (1 - c_q6),
                                      self.qc_loc[3] * s + self.qc_loc[4] * c - L_Bx * s_q6 - L_By * (1 - c_q6),
                                      self.qc_loc[5]])

        else:
            self.Gamma = np.array([[c, -s, -L_Ay, 0, 0, 0],
                                   [s, c, L_Ax, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, c, -s, -L_By],
                                   [0, 0, 0, s, c, -L_Bx],
                                   [0, 0, 0, 0, 0, 1]])

            self.q_c[:6] = self.Gamma @ self.qc_loc

        # print('q_c', self.q_c)

    def get_p_c(self):

        self.get_q_bsc()

        # print('q_bsc', self.q_bsc)

        self.get_p_bsc()

        # print('p_bsc', self.p_bsc)

        self.p_c = np.transpose(self.qc_to_bsc) @ self.p_bsc

        # print('p_c', self.p_c)

    def get_q_bsc(self):

        # if self.lin_geom: 
        self.qc_to_bsc = np.array([[-1, 0, 0, 1, 0, 0],
                                   [0, -1, 0, 0, 1, 0],
                                   [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1]])
        # else: 
        #     self.qc_to_bsc = np.array([[-1,  0, 0, 1, 0, 0],
        #                                [ 0, -1, 0, 0, 1, 0],
        #                                [ 0,  0, 1/2, 0, 0, 1/2],
        #                                [ 0,  0, 1/2, 0, 0, 1/2]])   

        self.q_bsc = self.qc_to_bsc @ self.q_c[:6]
        # # print('q_bsc',  self.q_bsc)

    def get_p_bsc(self):

        if self.sp1 == self.sp2:  # and self.q_bsc[2] == self.q_bsc[3]:
            self.solve_springs_new()
        else:
            self.solve_springs()
            # print('Hello')

        self.q_c[6] = self.q_c[0] + self.dL_A[0]
        self.q_c[7] = self.q_c[1] + self.dL_A[1]

        self.p_bsc = np.zeros(4)

        self.p_bsc[0] = self.p_xy_A[0]
        self.p_bsc[1] = self.p_xy_A[1]

        if not self.lin_geom:
            self.p_bsc[2] = self.p_xy_A[0] * self.dL_A[1] - self.p_xy_A[1] * self.dL_A[0]
            self.p_bsc[3] = self.p_xy_B[0] * self.dL_B[1] - self.p_xy_B[1] * self.dL_B[0]

        # print('p_bsc', self.p_bsc)  

    def update_springs(self, delta_L):

        self.sp1.update(delta_L[:2])
        self.sp2.update(delta_L[2:])

    def solve_springs_new(self):

        if self.lin_geom:
            T = T2x2(self.angle)

        else:
            mid_rot = (self.q_bsc[2] + self.q_bsc[3]) / 2
            T = T2x2(self.angle + mid_rot)

        if not hasattr(self, 'dL_ns'):
            self.dL_ns = np.zeros(4)
            self.dL_A = np.zeros(2)
            self.dL_B = np.zeros(2)
            self.update_springs(self.dL_ns)
            self.p_xy_A = T @ self.sp1.get_forces()
            self.p_xy_B = T @ self.sp2.get_forces()

            # print('q_bsc', self.q_bsc)
        dL_new = T.T @ self.q_bsc[:2] / 2
        # print('dL_new', dL_new)
        incr = dL_new - self.dL_ns[:2]
        delta_L = np.append(incr, incr)
        # print('delta_L', delta_L)
        self.update_springs(delta_L)

        self.dL_ns += delta_L
        self.dL_A = T @ self.dL_ns[:2]
        self.dL_B = T @ self.dL_ns[2:]

        self.p_xy_A = T @ self.sp1.get_forces()
        self.p_xy_B = T @ self.sp2.get_forces()

    def solve_springs(self):

        # print('q_bsc', self.q_bsc)

        def jacobian(x=None):

            k_A = self.sp1.get_k_spring()
            k_B = self.sp2.get_k_spring()

            J = np.zeros((4, 4))
            J[np.ix_([0, 1], [0, 1])] = T_A
            J[np.ix_([0, 1], [2, 3])] = T_B
            J[np.ix_([2, 3], [0, 1])] = T_A @ k_A
            J[np.ix_([2, 3], [2, 3])] = - T_B @ k_B

            try:
                np.linalg.inv(k_A)
            except:
                k_A += self.sp1.get_k_spring0() * 1e-6
            try:
                np.linalg.inv(k_B)
            except:
                k_B += self.sp2.get_k_spring0() * 1e-6

            # if np.around(k_A[0,0] == 0): k_A[0,0] = self.sp1.get_k_spring0()[0,0] * 1e-10
            # if np.around(k_A[1,1] == 0): k_A[1,1] = self.sp1.get_k_spring0()[1,1] * 1e-10
            # if np.around(k_B[0,0] == 0): k_B[0,0] = self.sp2.get_k_spring0()[0,0] * 1e-10
            # if np.around(k_B[1,1] == 0): k_B[1,1] = self.sp2.get_k_spring0()[1,1] * 1e-10
            J = np.zeros((4, 4))
            J[np.ix_([0, 1], [0, 1])] = T_A
            J[np.ix_([0, 1], [2, 3])] = T_B
            J[np.ix_([2, 3], [0, 1])] = T_A @ k_A
            J[np.ix_([2, 3], [2, 3])] = - T_B @ k_B

            return J

        def objective(delta_L):

            self.update_springs(delta_L)

            self.dL_ns += delta_L

            self.dL_A = T_A @ self.dL_ns[:2]
            self.dL_B = T_B @ self.dL_ns[2:]

            self.p_xy_A = T_A @ self.sp1.get_forces()
            self.p_xy_B = T_B @ self.sp2.get_forces()

            R[:2] = self.dL_A + self.dL_B - self.q_bsc[:2]
            R[2:] = self.p_xy_A - self.p_xy_B

            return R

        def bounds():

            # max_x = max(0, self.q_bsc[0]) + 1e-8
            # max_y = max(0, self.q_bsc[1]) + 1e-8

            # min_x = min(0, self.q_bsc[0]) - 1e-8
            # min_y = min(0, self.q_bsc[1]) - 1e-8

            min_x = -np.inf
            max_x = np.inf

            min_y = -np.inf
            max_y = np.inf

            return sc.optimize.Bounds([min_x, min_y, min_x, min_y], [max_x, max_y, max_x, max_y])

        def is_within_bounds(solution, bounds=bounds()):
            return all(lower <= val <= upper for val, lower, upper in zip(solution, bounds.lb, bounds.ub))

        if self.lin_geom:
            T_A = T2x2(self.angle)
            T_B = T2x2(self.angle)
        else:
            T_A = T2x2(self.angle + self.q_bsc[2])
            T_B = T2x2(self.angle + self.q_bsc[3])
            # T_A = T2x2(self.angle + mid_rot)
            # T_B = T2x2(self.angle + mid_rot)

        # print(self.x_cp)

        if not hasattr(self, 'dL_ns'):
            self.dL_ns = np.zeros(4)
            self.dL_A = np.zeros(2)
            self.dL_B = np.zeros(2)
            self.update_springs(self.dL_ns)
            self.p_xy_A = T_A @ self.sp1.get_forces()
            self.p_xy_B = T_B @ self.sp2.get_forces()

        R = np.zeros(4)

        R[:2] = self.dL_A + self.dL_B - self.q_bsc[:2]
        R[2:] = self.p_xy_A - self.p_xy_B

        # print('R_init', R)
        norm_R = np.linalg.norm(R)

        tol_disps = 1e-4
        tol_forces = 1
        max_iterations = 100
        iterations = 0
        conv = False

        while (not conv) and iterations < max_iterations:

            J = jacobian()
            # print('J', np.around(J, 8))
            # print('R', R)

            delta_L = - np.linalg.solve(J, R)

            # print(delta_L)

            R = objective(delta_L)

            # print('pxy', self.p_xy_A, self.p_xy_B)

            norm_R_forces = np.linalg.norm(R[2:])
            norm_R_disps = np.linalg.norm(R[:2])

            if norm_R_forces < tol_forces and norm_R_disps < tol_disps and is_within_bounds(
                    np.append(self.dL_A, self.dL_B)):
                # print(f'Converged after {iterations} iterations')
                conv = True
                # print('p_xy', self.p_xy_A)
                break

            iterations += 1

        if not conv:
            # warnings.warn(f"Inside loop did not converge {norm_R}")
            raise Exception(f"Inside loop did not converge {norm_R}")

    def to_ommit(self):

        cp1 = self.sp1.to_ommit()
        cp2 = self.sp2.to_ommit()

        return (cp1 and cp2)

    def get_kc_loc(self):

        self.get_k_AB_loc()

        self.kc_loc = np.transpose(self.Gamma) @ self.k_AB_loc @ self.Gamma

        if not self.lin_geom:
            (L_Ax, L_Ay) = T2x2(self.angle) @ np.array([self.sp1.l_n, self.sp1.l_s])
            (L_Bx, L_By) = np.transpose(T2x2(self.angle)) @ np.array([self.sp2.l_n, self.sp2.l_s])

            c_q3 = np.cos(self.qc_loc[2])
            s_q3 = np.sin(self.qc_loc[2])
            c_q6 = np.cos(self.qc_loc[5])
            s_q6 = np.sin(self.qc_loc[5])

            mid_rot = (self.qc_loc[2] + self.qc_loc[5]) / 2

            c_q3 = np.cos(mid_rot)
            s_q3 = np.sin(mid_rot)
            c_q6 = np.cos(mid_rot)
            s_q6 = np.sin(mid_rot)

            self.kc_loc[2, 2] += self.p_c[0] * (L_Ay * s_q3 - L_Ax * c_q3) + self.p_c[1] * (-L_Ax * s_q3 - L_Ay * c_q3)
            self.kc_loc[5, 5] += self.p_c[3] * (L_By * s_q6 + L_Bx * c_q6) + self.p_c[4] * (L_Bx * s_q6 - L_By * c_q6)

        return self.kc_loc

    def get_kc_loc0(self):

        self.get_k_AB_loc0()

        T = T2x2(self.angle)

        (L_Ax, L_Ay) = T @ np.array([self.l_Ax, self.l_Ay])
        (L_Bx, L_By) = np.transpose(T) @ np.array([self.l_Bx, self.l_By])

        c = np.cos(self.angle)
        s = np.sin(self.angle)

        self.Gamma0 = np.array([[c, -s, -L_Ay, 0, 0, 0],
                                [s, c, L_Ax, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, c, -s, -L_By],
                                [0, 0, 0, s, c, -L_Bx],
                                [0, 0, 0, 0, 0, 1]])

        self.kc_loc0 = np.transpose(self.Gamma0) @ self.k_AB_loc0 @ self.Gamma0

        return self.kc_loc0

    def get_kc_loc_LG(self):

        self.get_k_AB_loc_LG()

        T = T2x2(self.angle)

        (L_Ax, L_Ay) = T @ np.array([self.l_Ax, self.l_Ay])
        (L_Bx, L_By) = np.transpose(T) @ np.array([self.l_Bx, self.l_By])

        c = np.cos(self.angle)
        s = np.sin(self.angle)

        self.Gamma0 = np.array([[c, -s, -L_Ay, 0, 0, 0],
                                [s, c, L_Ax, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, c, -s, -L_By],
                                [0, 0, 0, s, c, -L_Bx],
                                [0, 0, 0, 0, 0, 1]])

        self.kc_loc_LG = np.transpose(self.Gamma0) @ self.k_AB_loc_LG @ self.Gamma0

        return self.kc_loc_LG

    def get_k_AB_loc(self):

        # self.k_AB_loc = np.zeros((6,6))

        k_A = self.sp1.get_k_spring()
        k_B = self.sp2.get_k_spring()

        if self.lin_geom:
            T_a = T2x2(self.angle)
            T_b = T2x2(self.angle)
            dxA = 0
            dyA = 0
            dxB = 0
            dyB = 0

        else:

            mid_rot = (self.q_bsc[2] + self.q_bsc[3]) / 2
            T_a = T2x2(self.angle + mid_rot)
            T_b = T2x2(self.angle + mid_rot)
            # T_a = T2x2(self.angle)
            # T_b = T2x2(self.angle)    
            dxA = self.dL_A[0]
            dyA = self.dL_A[1]
            dxB = self.dL_B[0]
            dyB = self.dL_B[1]

        def is_diagonal(A):
            return np.all(A == np.diag(np.diagonal(A)))

        k_eq = np.zeros((2, 2))

        if is_diagonal(k_A) and is_diagonal(k_B):
            for i in range(2):
                if np.around(k_A[i, i] + k_B[i, i], 15) != 0:
                    k_eq[i, i] = (k_A[i, i] * k_B[i, i]) / (k_A[i, i] + k_B[i, i])
        else:
            eigvals, P1 = np.linalg.eig(k_A)
            eigvals, P2 = np.linalg.eig(k_B)

            if np.allclose(P1, P2):

                A_d = np.linalg.inv(P1) @ k_A @ P1
                B_d = np.linalg.inv(P1) @ k_B @ P1
                k_eq_d = np.zeros((2, 2))

                for i in range(2):

                    if np.around(A_d[i, i] + B_d[i, i], 15) != 0:
                        k_eq_d[i, i] = (A_d[i, i] * B_d[i, i]) / (A_d[i, i] + B_d[i, i])

                k_eq = P1 @ k_eq_d @ np.linalg.inv(P1)

        k_eq_XY = T_a @ k_eq @ np.transpose(T_a)
        # k_A_XY = T_a @ k_A @ np.transpose(T_a) 
        # k_B_XY = T_b @ k_B @ np.transpose(T_b) 

        # print('K_A_XY', k_A_XY, k_B_XY)

        # rank_A = np.linalg.matrix_rank(k_A_XY, tol=1e-10)
        # rank_B = np.linalg.matrix_rank(k_B_XY, tol=1e-10)

        # # Both are full rank
        # if rank_A == 2 and rank_B == 2: 
        #     # print('A and B are invertible')
        #     k_eq = np.linalg.inv(np.linalg.inv(k_A_XY) + np.linalg.inv(k_B_XY))

        # # One is full rank, the other has rank 1
        # elif rank_A * rank_B == 2: 

        #     # print('Either A or B have rank 1')
        #     if rank_A == 1: eigvals, P = np.linalg.eig(k_A_XY)
        #     elif rank_B == 1: eigvals, P = np.linalg.eig(k_B_XY)

        #     A_d = np.linalg.inv(P)@k_A_XY@P
        #     B_d = np.linalg.inv(P)@k_B_XY@P

        #     k_eq_d = np.diag([(A_d[0,0]*B_d[0,0])/(A_d[0,0]+B_d[0,0]),(A_d[1,1]*B_d[1,1])/(A_d[1,1]+B_d[1,1])])            

        #     k_eq = P@k_eq_d@np.linalg.inv(P)

        #     # k_eq = np.zeros((2,2))
        #     # for i in range(2): 
        #     #     for j in range(2): 
        #     #         if np.around(k_A_XY[i][j] + k_B_XY[i][j],10) != 0:
        #     #             k_eq[i][j] = (k_A_XY[i][j] * k_B_XY[i][j]) /  (k_A_XY[i][j] + k_B_XY[i][j])

        #     # print(k_eq)
        # # Both have rank one, check if they have identical null spaces 
        # elif rank_A * rank_B == 1: 

        #     eigvals, P1 = np.linalg.eig(k_A_XY)
        #     eigvals, P2 = np.linalg.eig(k_B_XY)

        #     # if np.allclose(P1, P2, 1e-3):          
        #         # print('Same null space')
        #     A_d = np.linalg.inv(P1)@k_A_XY@P1
        #     B_d = np.linalg.inv(P1)@k_B_XY@P1

        #     k_eq_d = np.zeros((2,2))

        #     for i in range(2):

        #         if np.around(A_d[i,i]+B_d[i,i],15) != 0: 
        #             k_eq_d[i,i] = (A_d[i,i]*B_d[i,i])/(A_d[i,i]+B_d[i,i])

        #     k_eq = P1@k_eq_d@np.linalg.inv(P1)

        #     # else: 
        #     #     k_eq = np.zeros((2,2))

        #         # print(k_eq)
        #         # print('null space of A and B spans R2')

        # # They all have rank 0
        # else: 
        #     # print(f'rank of A is {rank_A} and rank of B is {rank_B}')
        #     k_eq = np.zeros((2,2))

        kx = k_eq_XY[0, 0]
        ky = k_eq_XY[1, 1]
        kxy = k_eq_XY[0, 1]
        kyx = k_eq_XY[1, 0]

        k11 = kx
        k21 = kyx
        k31 = kyx * dxA - kx * dyA
        k41 = -k11
        k51 = -k21
        k61 = kyx * dxB - kx * dyB

        k12 = kxy
        k22 = ky
        k32 = ky * dxA - kxy * dyA
        k42 = -k12
        k52 = -k22
        k62 = ky * dxB - kxy * dyB

        k13 = kxy * dxA - kx * dyA
        k23 = ky * dxA - kyx * dyA
        k33 = kx * dyA ** 2 + ky * dxA ** 2 - (kxy + kyx) * dxA * dyA
        k43 = - k13
        k53 = - k23
        k63 = kx * dyA * dyB - kyx * dyA * dxB - kxy * dxA * dyB + ky * dxA * dxB

        k44 = kx
        k54 = kyx
        k64 = kx * dyB - kyx * dxB
        k14 = -k44
        k24 = -k54
        k34 = kx * dyA - kyx * dxA

        k45 = kxy
        k55 = ky
        k65 = kxy * dyB - ky * dxB
        k15 = -k45
        k25 = -k55
        k35 = kxy * dyA - ky * dxA

        k46 = - kxy * dxB + kx * dyB
        k56 = kyx * dyB - ky * dxB
        k66 = kx * dyB ** 2 + ky * dxB ** 2 - (kxy + kyx) * dxB * dyB
        k16 = -k46
        k26 = -k56
        k36 = kx * dyB * dyA - kyx * dyB * dxA - kxy * dxB * dyA + ky * dxA * dxB

        self.k_AB_loc = np.array([[k11, k12, k13, k14, k15, k16],
                                  [k21, k22, k23, k24, k25, k26],
                                  [k31, k32, k33, k34, k35, k36],
                                  [k41, k42, k43, k44, k45, k46],
                                  [k51, k52, k53, k54, k55, k56],
                                  [k61, k62, k63, k64, k65, k66]])

        # def check_symmetric(a, tol=1):
        #     return np.all(np.abs(a-a.T) < tol)

        # if not check_symmetric(self.k_AB_loc):
        #     print('kAB_loc1',  np.around(self.k_AB_loc,10))
        #     print(k_A_XY, k_B_XY)
        #     print(k_eq)
        #     print(dxA, dxB, dyA, dyB)
        #     raise Exception('Matrix is not symmetric')
        # # print('kAB_loc1',  np.around(self.k_AB_loc,10))

        # Uncomment the next 4 lines to activate previous computation of kcAB
        # self.get_k_c()

        # if np.linalg.cond(self.k_c[6:,6:]) > 1e10: 
        #     print('High condition number', np.linalg.cond(self.k_c[6:,6:]))
        # self.k_AB_loc = self.k_c[:6,:6] - self.k_c[:6,6:] @ np.linalg.solve(self.k_c[6:,6:],self.k_c[6:,:6])

        # self.k_AB_loc = self.k_c[:6,:6] - self.k_c[:6,6:] @ np.linalg.inv(self.k_c[6:,6:])@self.k_c[6:,:6]

        # except: 
        #     try: self.k_AB_loc = self.k_c[:6,:6] - self.k_c[:6,6:] @ sc.linalg.pinv(self.k_c[6:,6:]) @ self.k_c[6:,:6]
        #     except: 
        #         self.k_AB_loc = self.k_c[:6,:6]

        # print('kAB_loc2',  np.around(self.k_AB_loc,10))

    def get_k_AB_loc_LG(self):

        k_A = self.sp1.get_k_spring()
        k_B = self.sp2.get_k_spring()

        T_a = T2x2(self.angle)
        T_b = T2x2(self.angle)

        k_A_XY = T_a @ k_A @ np.transpose(T_a)
        k_B_XY = T_b @ k_B @ np.transpose(T_b)

        kxA = k_A_XY[0, 0]
        kxyA = k_A_XY[0, 1]
        kyxA = k_A_XY[1, 0]
        kyA = k_A_XY[1, 1]

        kxB = k_B_XY[0, 0]
        kxyB = k_B_XY[0, 1]
        kyxB = k_B_XY[1, 0]
        kyB = k_B_XY[1, 1]

        if np.linalg.cond(k_A_XY) < 1e10 and np.linalg.cond(k_B_XY) < 1e10:
            k_eq = np.linalg.inv(np.linalg.inv(k_A_XY) + np.linalg.inv(k_B_XY))
            kx = k_eq[0, 0]
            ky = k_eq[1, 1]
            kxy = k_eq[0, 1]
            kyx = k_eq[1, 0]
        else:
            if np.around((kxA + kxB), 6) == 0:
                kx = 0
            else:
                kx = (kxA * kxB) / (kxA + kxB)
            if np.around((kxyA + kxyB), 6) == 0:
                kxy = 0
            else:
                kxy = (kxyA * kxyB) / (kxyA + kxyB)
            if np.around((kyxA + kyxB), 6) == 0:
                kyx = 0
            else:
                kyx = (kyxA * kyxB) / (kyxA + kyxB)
            if np.around((kyA + kyB), 6) == 0:
                ky = 0
            else:
                ky = (kyA * kyB) / (kyA + kyB)

            # print(k_A_XY, k_B_XY)
            # raise Exception('This is keq', kx, kxy, kyx, ky)

        k11 = kx
        k21 = kyx
        k41 = -k11
        k51 = -k21

        k12 = kxy
        k22 = ky
        k42 = -k12
        k52 = -k22

        k44 = kx
        k54 = kyx
        k14 = -k44
        k24 = -k54

        k45 = kxy
        k55 = ky
        k15 = -k45
        k25 = -k55

        self.k_AB_loc_LG = np.array([[k11, k12, 0, k14, k15, 0],
                                     [k21, k22, 0, k24, k25, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [k41, k42, 0, k44, k45, 0],
                                     [k51, k52, 0, k54, k55, 0],
                                     [0, 0, 0, 0, 0, 0]])

    def get_k_AB_loc0(self):

        self.get_k_c0()

        self.k_AB_loc0 = self.k_c0[:6, :6] - self.k_c0[:6, 6:] @ sc.linalg.solve(self.k_c0[6:, 6:], self.k_c0[6:, :6])

    def get_k_c(self):

        k_A = self.sp1.get_k_spring()
        k_B = self.sp2.get_k_spring()

        if self.lin_geom:
            T_a = T2x2(self.angle)
            T_b = T2x2(self.angle)
            dxA = 0
            dyA = 0
            dxB = 0
            dyB = 0

        else:
            T_a = T2x2(self.angle + self.q_bsc[2])
            T_b = T2x2(self.angle + self.q_bsc[3])
            dxA = self.dL_A[0]
            dyA = self.dL_A[1]
            dxB = self.dL_B[0]
            dyB = self.dL_B[1]

        k_A_XY = T_a @ k_A @ np.transpose(T_a)
        k_B_XY = T_b @ k_B @ np.transpose(T_b)

        kxA = k_A_XY[0, 0]
        kxyA = k_A_XY[0, 1]
        kyxA = k_A_XY[1, 0]
        kyA = k_A_XY[1, 1]

        kxB = k_B_XY[0, 0]
        kxyB = k_B_XY[0, 1]
        kyxB = k_B_XY[1, 0]
        kyB = k_B_XY[1, 1]

        # First column of kc
        k_11 = kxA
        k_21 = kyxA
        k_31 = - kxA * dyA + kyxA * dxA
        k_71 = - kxA
        k_81 = - kyxA

        # Second column
        k_12 = kxyA
        k_22 = kyA
        k_32 = - kxyA * dyA + kyA * dxA
        k_72 = -kxyA
        k_82 = -kyA

        # Third column
        k_13 = - dyA * kxA + dxA * kxyA
        k_23 = - dyA * kyxA + dxA * kyA
        k_73 = -k_13
        k_83 = -k_23
        k_33 = kxA * dyA ** 2 + kyA * dxA ** 2 - (kxyA + kyxA) * dxA * dyA

        # Fourth column
        k_44 = kxB
        k_54 = kyxB
        k_64 = - dxB * kyxB + dyB * kxB
        k_74 = - kxB
        k_84 = - kyxB

        # Fifth column
        k_45 = kxyB
        k_55 = kyB
        k_65 = - kyB * dxB + kxyB * dyB
        k_75 = -kxyB
        k_85 = -kyB

        # Sixth column
        k_46 = kxB * dyB - kxyB * dxB
        k_56 = kyxB * dyB - kyB * dxB
        k_76 = - k_46
        k_86 = - k_56
        k_66 = kxB * dyB ** 2 + kyB * dxB ** 2 - (kxyB + kyxB) * dxB * dyB

        # Seventh column
        k_17 = -kxA
        k_27 = -kyxA
        k_37 = kxA * dyA - kyxA * dxA
        k_47 = -kxB
        k_57 = -kyxB
        k_67 = - kxB * dyB + kyxB * dxB
        k_77 = kxA + kxB
        k_87 = kyxA + kyxB

        # Eigtht column
        k_18 = -kxyA
        k_28 = -kyA
        k_38 = - kyA * dxA + kxyA * dyA
        k_48 = -kxyB
        k_58 = -kyB
        k_68 = - kxB * dyB + kxyB * dxB
        k_78 = kxyA + kxyB
        k_88 = kyA + kyB

        self.k_c = np.array([[k_11, k_12, k_13, 0, 0, 0, k_17, k_18],
                             [k_21, k_22, k_23, 0, 0, 0, k_27, k_28],
                             [k_31, k_32, k_33, 0, 0, 0, k_37, k_38],
                             [0, 0, 0, k_44, k_45, k_46, k_47, k_48],
                             [0, 0, 0, k_54, k_55, k_56, k_57, k_58],
                             [0, 0, 0, k_64, k_65, k_66, k_67, k_68],
                             [k_71, k_72, k_73, k_74, k_75, k_76, k_77, k_78],
                             [k_81, k_82, k_83, k_84, k_85, k_86, k_87, k_88]])

        # print('kc', np.around(self.k_c))

    def get_k_c0(self):

        k_A = self.sp1.get_k_spring0()
        k_B = self.sp2.get_k_spring0()

        T_a = T2x2(self.angle)
        T_b = T2x2(self.angle)

        k_A_XY = T_a @ k_A @ np.transpose(T_a)
        k_B_XY = T_b @ k_B @ np.transpose(T_b)

        dxA = 0
        dyA = 0
        dxB = 0
        dyB = 0

        kxA = k_A_XY[0, 0]
        kxyA = k_A_XY[0, 1]
        kyxA = k_A_XY[1, 0]
        kyA = k_A_XY[1, 1]

        kxB = k_B_XY[0, 0]
        kxyB = k_B_XY[0, 1]
        kyxB = k_B_XY[1, 0]
        kyB = k_B_XY[1, 1]

        # First column of kc
        k_11 = kxA
        k_21 = kyxA
        k_31 = - kxA * dyA + kyxA * dxA
        k_71 = - kxA
        k_81 = - kyxA

        # Second column
        k_12 = kxyA
        k_22 = kyA
        k_32 = - kxyA * dyA + kyA * dxA
        k_72 = -kxyA
        k_82 = -kyA

        # Third column
        k_13 = - dyA * kxA + dxA * kxyA
        k_23 = - dyA * kyxA + dxA * kyA
        k_73 = -k_13
        k_83 = -k_23
        k_33 = - k_13 * dyA + k_23 * dxA

        # Fourth column
        k_44 = kxB
        k_54 = kyxB
        k_64 = - dxB * kyxB + dyB * kxB
        k_74 = - kxB
        k_84 = - kyxB

        # Fifth column
        k_45 = kxyB
        k_55 = kyB
        k_65 = - kyB * dxB + kxyB * dyB
        k_75 = -kxyB
        k_85 = -kyB

        # Sixth column
        k_46 = - kxB * dyB + kxyB * dxB
        k_56 = - kyxB * dyB + kyB * dxB
        k_76 = - k_46
        k_86 = - k_56
        k_66 = - k_76 * dyB + k_86 * dxB

        # Seventh column
        k_17 = -kxA
        k_27 = -kyxA
        k_37 = kxA * dyA - kyxA * dxA
        k_47 = -kxB
        k_57 = -kyxB
        k_67 = - kxB * dyB + kyxB * dxB
        k_77 = kxA + kxB
        k_87 = kyxA + kyxB

        # Eigtht column
        k_18 = -kxyA
        k_28 = -kyA
        k_38 = - kyA * dxA + kxyA * dyA
        k_48 = -kxyB
        k_58 = -kyB
        k_68 = - kxB * dyB + kxyB * dxB
        k_78 = kxyA + kxyB
        k_88 = kyA + kyB

        self.k_c0 = np.array([[k_11, k_12, k_13, 0, 0, 0, k_17, k_18],
                              [k_21, k_22, k_23, 0, 0, 0, k_27, k_28],
                              [k_31, k_32, k_33, 0, 0, 0, k_37, k_38],
                              [0, 0, 0, k_44, k_45, k_46, k_47, k_48],
                              [0, 0, 0, k_54, k_55, k_56, k_57, k_58],
                              [0, 0, 0, k_64, k_65, k_66, k_67, k_68],
                              [k_71, k_72, k_73, k_74, k_75, k_76, k_77, k_78],
                              [k_81, k_82, k_83, k_84, k_85, k_86, k_87, k_88]])

    def plot(self, scale):

        tA = mpl.markers.MarkerStyle(marker=0)
        tA._transform = tA.get_transform().rotate(self.angle)
        tB = mpl.markers.MarkerStyle(marker=1)
        tB._transform = tB.get_transform().rotate(self.angle)

        if self.reinf:
            color = 'red'
        else:
            color = 'blue'

        plt.plot(self.x_cp[0], self.x_cp[1], color=color, marker=tA, markersize=3)
        plt.plot(self.x_cp[0], self.x_cp[1], color=color, marker=tB, markersize=3)
