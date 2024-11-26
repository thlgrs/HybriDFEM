# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:05:21 2024

@author: ibouckaert
"""

import os
import warnings
from copy import deepcopy
from warnings import warn

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from Objects import Spring as sp


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"


warnings.formatwarning = custom_warning_format


def T2x2(a): return np.array([[np.cos(a), -np.sin(a)],
                              [np.sin(a), np.cos(a)]])


class CP_2D:

    def __init__(self, x_cp, l_Ax, l_Ay, l_Bx, l_By, angle, h_cp, b, block_A=None, block_B=None, contact=None,
                 surface=None, lin_geom=True):

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

        if l_Ax < 0 or l_Bx < 0:
            warn('l_Ax and l_Bx should be positive')
            print(self.x_cp, l_Ax, l_Bx)

        if contact is None and surface is None:
            if block_A is None and block_B is None: warn('Must refer to a block or define contact/surface law')
            self.bl_A = block_A
            self.bl_B = block_B

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

        elif contact is None:
            self.surface = surface
            self.bl_A = None
            self.bl_B = None
        else:
            self.contact = contact
            self.bl_A = None
            self.bl_B = None

        self.sp1 = sp.Spring_2D(self.l_Ax, self.l_Ay, self.h, self.b, block=self.bl_A, contact=contact, surface=surface)
        self.sp2 = sp.Spring_2D(self.l_Bx, self.l_By, self.h, self.b, block=self.bl_B, contact=contact, surface=surface)

    def commit(self):
        self.sp1.commit()
        self.sp2.commit()

    def revert_commit(self):
        self.sp1.revert_commit()
        self.sp2.revert_commit()

    def get_pc_loc(self, qf_loc):

        self.qc_loc = qf_loc.copy()
        self.get_q_c()
        self.get_p_c()
        self.pc_loc = np.transpose(self.Gamma) @ self.p_c

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

    def get_p_c(self):

        self.get_q_bsc()

        self.get_p_bsc()

        self.p_c = np.transpose(self.qc_to_bsc) @ self.p_bsc

    def get_q_bsc(self):

        self.qc_to_bsc = np.array([[-1, 0, 0, 1, 0, 0],
                                   [0, -1, 0, 0, 1, 0],
                                   [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1]])

        self.q_bsc = self.qc_to_bsc @ self.q_c[:6]

    def get_p_bsc(self):

        self.solve_springs()

        self.q_c[6] = self.q_c[0] + self.dL_A[0]
        self.q_c[7] = self.q_c[1] + self.dL_A[1]

        self.p_bsc = np.zeros(4)

        self.p_bsc[0] = self.p_xy_A[0]
        self.p_bsc[1] = self.p_xy_A[1]
        self.p_bsc[2] = self.p_xy_A[0] * self.dL_A[1] - self.p_xy_A[1] * self.dL_A[0]
        self.p_bsc[3] = self.p_xy_B[0] * self.dL_B[1] - self.p_xy_B[1] * self.dL_B[0]

    def update_springs(self):

        self.sp1.set_elongs(self.dL_ns[:2])
        self.sp2.set_elongs(self.dL_ns[2:])

        self.sp1.update()
        self.sp2.update()

    def solve_springs(self):

        k_A = self.sp1.get_k_spring()
        k_B = self.sp2.get_k_spring()

        if self.lin_geom:
            T_A = T2x2(self.angle)
            T_B = T2x2(self.angle)
        else:
            T_A = T2x2(self.angle + self.q_bsc[2])
            T_B = T2x2(self.angle + self.q_bsc[3])

        # if self.bl_A.mat.shear_def = True: 

        A = np.zeros((4, 4))

        A[np.ix_([0, 1], [0, 1])] = T_A
        A[np.ix_([0, 1], [2, 3])] = T_B
        A[np.ix_([2, 3], [0, 1])] = - T_A @ k_A
        A[np.ix_([2, 3], [2, 3])] = T_B @ k_B

        b = np.array([self.q_bsc[0], self.q_bsc[1], 0, 0])

        self.dL_ns = np.linalg.solve(A, b)

        self.dL_A = T_A @ self.dL_ns[:2]
        self.dL_B = T_B @ self.dL_ns[2:]

        self.update_springs()

        self.p_xy_A = T_A @ self.sp1.get_forces()
        self.p_xy_B = T_B @ self.sp2.get_forces()

        if np.linalg.norm(self.p_xy_A - self.p_xy_B) > 0.1:
            self.solve_springs_NL()

    def solve_springs_NL(self):

        if self.lin_geom:
            T_A = T2x2(self.angle)
            T_B = T2x2(self.angle)
        else:
            T_A = T2x2(self.angle + self.q_bsc[2])
            T_B = T2x2(self.angle + self.q_bsc[3])

        def jacobian():

            k_A = self.sp1.get_k_spring()
            k_B = self.sp2.get_k_spring()

            J = np.zeros((4, 4))
            J[np.ix_([0, 1], [0, 1])] = - T_A @ k_A
            J[np.ix_([0, 1], [2, 3])] = T_B @ k_B
            J[np.ix_([2, 3], [0, 1])] = T_A
            J[np.ix_([2, 3], [2, 3])] = T_B

            return J

        def solution():

            F_a = self.sp1.get_forces()
            F_b = self.sp2.get_forces()

            sol = np.zeros(4)

            sol[2:] = T_A @ self.dL_ns[:2] + T_B @ self.dL_ns[2:] - self.q_bsc[:2]
            sol[:2] = T_A @ F_a - T_B @ F_b

            return sol

        tol = 10
        loop_iters = 0
        max_iters = 1000
        loop_conv = False

        while not loop_conv:

            if loop_iters < max_iters:

                self.dL_ns += np.linalg.solve(jacobian(), solution())

                self.update_springs()

                self.p_xy_A = T_A @ self.sp1.get_forces()
                self.p_xy_B = T_B @ self.sp2.get_forces()

                if np.linalg.norm(self.p_xy_A - self.p_xy_B) < tol:
                    loop_conv = True
                else:
                    loop_iters += 1

            else:
                rank = np.linalg.matrix_rank(jacobian())
                raise Exception(f"Inside loop did not converge, Jacobian with rank {rank}")

        self.dL_A = T_A @ self.dL_ns[:2]
        self.dL_B = T_B @ self.dL_ns[2:]

    def to_ommit(self):

        return self.sp1.to_ommit() or self.sp2.to_ommit()

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

    def get_k_AB_loc(self):

        self.get_k_c()

        self.k_AB_loc = self.k_c[:6, :6] - self.k_c[:6, 6:] @ np.linalg.solve(self.k_c[6:, 6:], self.k_c[6:, :6])

    def get_k_AB_loc0(self):

        self.get_k_c0()

        self.k_AB_loc0 = self.k_c0[:6, :6] - self.k_c0[:6, 6:] @ np.linalg.solve(self.k_c0[6:, 6:], self.k_c0[6:, :6])

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
        kyA = k_A_XY[1, 1]

        kxB = k_B_XY[0, 0]
        kxyB = k_B_XY[0, 1]
        kyB = k_B_XY[1, 1]

        k11 = kxA
        k21 = kxyA
        k31 = kxyA * dxA - kxA * dyA
        k71 = -kxA
        k81 = -kxyA

        k22 = kyA
        k32 = kyA * dxA - kxyA * dyA
        k72 = -kxyA
        k82 = -kyA

        k33 = kxA * dyA ** 2 + kyA * dxA ** 2 - 2 * kxyA * dxA * dyA
        k73 = -k31
        k83 = -k32

        k44 = kxB
        k54 = kxyB
        k64 = kxB * dyB - kxyB * dxB
        k74 = -kxB
        k84 = -kxyB

        k55 = kyB
        k65 = kxyB * dyB - kyB * dxB
        k75 = -kxyB
        k85 = -kyB

        k66 = kxB * dyB ** 2 + kyB * dxB ** 2 - 2 * kxyB * dxB * dyB
        k76 = -k64
        k86 = -k65

        k77 = kxA + kxB
        k87 = kxyA + kxyB

        k88 = kyA + kyB

        self.k_c = np.array([[k11, k21, k31, 0, 0, 0, k71, k81],
                             [k21, k22, k32, 0, 0, 0, k72, k82],
                             [k31, k32, k33, 0, 0, 0, k73, k83],
                             [0, 0, 0, k44, k54, k64, k74, k84],
                             [0, 0, 0, k54, k55, k65, k75, k85],
                             [0, 0, 0, k64, k65, k66, k76, k86],
                             [k71, k72, k73, k74, k75, k76, k77, k87],
                             [k81, k82, k83, k84, k85, k86, k87, k88]])

    def get_k_c0(self):

        k_A = self.sp1.get_k_spring0()
        k_B = self.sp2.get_k_spring0()

        T_a = T2x2(self.angle)
        T_b = T2x2(self.angle)

        k_A_XY = T_a @ k_A @ np.transpose(T_a)
        k_B_XY = T_b @ k_B @ np.transpose(T_b)

        kxA = k_A_XY[0, 0]
        kxyA = k_A_XY[0, 1]
        kyA = k_A_XY[1, 1]

        kxB = k_B_XY[0, 0]
        kxyB = k_B_XY[0, 1]
        kyB = k_B_XY[1, 1]

        k11 = kxA
        k21 = kxyA
        k71 = -kxA
        k81 = -kxyA

        k22 = kyA
        k72 = -kxyA
        k82 = -kyA

        k44 = kxB
        k54 = kxyB
        k74 = -kxB
        k84 = -kxyB

        k55 = kyB
        k75 = -kxyB
        k85 = -kyB

        k77 = kxA + kxB
        k87 = kxyA + kxyB

        k88 = kyA + kyB

        self.k_c0 = np.array([[k11, k21, 0, 0, 0, 0, k71, k81],
                              [k21, k22, 0, 0, 0, 0, k72, k82],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, k44, k54, 0, k74, k84],
                              [0, 0, 0, k54, k55, 0, k75, k85],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [k71, k72, 0, k74, k75, 0, k77, k87],
                              [k81, k82, 0, k84, k85, 0, k87, k88]])

    def plot(self, scale):

        tA = mpl.markers.MarkerStyle(marker=0)
        tA._transform = tA.get_transform().rotate(self.angle)
        tB = mpl.markers.MarkerStyle(marker=1)
        tB._transform = tB.get_transform().rotate(self.angle)

        plt.plot(self.x_cp[0], self.x_cp[1], color='blue', marker=tA, markersize=3)
        plt.plot(self.x_cp[0], self.x_cp[1], color='blue', marker=tB, markersize=3)
