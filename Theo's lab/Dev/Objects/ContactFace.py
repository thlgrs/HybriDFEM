# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:57:55 2024

@author: ibouckaert
"""

import os
import warnings
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

from .ContactPair import CP_2D


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"


warnings.formatwarning = custom_warning_format


class CF_2D:

    def __init__(self, cf, nb_cp, lin_geom, offset=-1, contact=None, surface=None, weights=None):

        self.xe1 = cf['x_e1'].copy()
        self.xe2 = cf['x_e2'].copy()

        self.bl_A = cf['Block A']
        self.bl_B = cf['Block B']

        if self.bl_A.b != self.bl_B.b: warnings.warn('Cannot handle blocks with different depths')
        self.b = min(self.bl_A.b, self.bl_B.b)

        self.t = (self.xe2 - self.xe1) / np.linalg.norm(self.xe2 - self.xe1)
        self.n = np.array([[0, -1], [1, 0]]) @ self.t
        self.angle = np.arctan2(self.t[1], self.t[0])

        # print(self.angle)
        self.cps = []

        self.lin_geom = lin_geom

        if offset == -1:  # Stress - strain

            if ((self.bl_A.material is None) or (self.bl_B.material is None)) and surface is None:
                warn('No material or surface law was defined')

            elif surface is None:

                h_cp = np.linalg.norm(self.xe2 - self.xe1) / nb_cp
                x_cp = self.xe1 + .5 * h_cp * self.t

                for i in np.arange(nb_cp):
                    x_cp = self.xe1 + (i + .5) * h_cp * self.t

                    l_Ax = np.dot((x_cp - self.bl_A.ref_point), -self.n)
                    l_Ay = np.dot((x_cp - self.bl_A.ref_point), self.t)
                    l_Bx = np.dot((x_cp - self.bl_B.ref_point), self.n)
                    l_By = np.dot((x_cp - self.bl_B.ref_point), self.t)

                    self.cps.append(
                        CP_2D(x_cp, l_Ax, l_Ay, l_Bx, l_By, self.angle - np.pi / 2, h_cp, self.b, surface=surface,
                              block_A=self.bl_A, block_B=self.bl_B, lin_geom=self.lin_geom))

            else:

                if isinstance(nb_cp, list):

                    d_tot = 0

                    for i, pos in enumerate(nb_cp):

                        if abs(pos) > 1: warn('Placing a CP outside a CF')

                        x_cp = .5 * (1 - pos) * self.xe1 + .5 * (1 + pos) * self.xe2

                        l_Ax = np.dot((x_cp - self.bl_A.ref_point), -self.n)
                        l_Ay = np.dot((x_cp - self.bl_A.ref_point), self.t)
                        l_Bx = np.dot((x_cp - self.bl_B.ref_point), self.n)
                        l_By = np.dot((x_cp - self.bl_B.ref_point), self.t)

                        if weights is not None:
                            if np.around(np.sum(weights), 10) != 1: warnings.warn('CP weights don\'t add up to one')
                            if len(weights) != len(nb_cp): warnings.warn(
                                'Number of weights is not coherent with number of CPs')
                            h_cp = weights[i] * np.linalg.norm(self.xe2 - self.xe1)
                            d_tot = np.around(np.sum(weights), 10)
                        else:
                            if i == 0:
                                d = (nb_cp[0] + nb_cp[1]) / 2 + 1
                            elif i == len(nb_cp) - 1:
                                d = 1 - (nb_cp[i] + nb_cp[i - 1]) / 2
                            else:
                                d = (nb_cp[i + 1] - nb_cp[i - 1]) / 2
                            if d <= 0: warn('Distances for CPs are not coherent')
                            # print(d)
                            d_tot += d / 2
                            h_cp = np.linalg.norm(self.xe2 - self.xe1) * d / 2

                        self.cps.append(CP_2D(x_cp, l_Ax, l_Ay, l_Bx, l_By, self.angle - np.pi / 2, h_cp, self.b,
                                              surface=surface, lin_geom=self.lin_geom))

                    if np.around(d_tot, 10) != 1:
                        print(d_tot)
                        warn('Distances for CPs are not coherent')

                else:
                    h_cp = np.linalg.norm(self.xe2 - self.xe1) / nb_cp

                    for i in np.arange(nb_cp):
                        x_cp = self.xe1 + (i + .5) * h_cp * self.t

                        # print(x_cp)

                        l_Ax = np.dot((x_cp - self.bl_A.ref_point), -self.n)
                        l_Ay = np.dot((x_cp - self.bl_A.ref_point), self.t)
                        l_Bx = np.dot((x_cp - self.bl_B.ref_point), self.n)
                        l_By = np.dot((x_cp - self.bl_B.ref_point), self.t)

                        self.cps.append(CP_2D(x_cp, l_Ax, l_Ay, l_Bx, l_By, self.angle - np.pi / 2, h_cp, self.b,
                                              surface=surface, lin_geom=self.lin_geom))

        elif offset >= 0:

            if nb_cp != 2:
                warn('Trying to use contact law with more than 2 CPs')

            if offset >= np.linalg.norm(self.xe2 - self.xe1) / 2:
                warn('Offset exceeds dimensions of CF')

            if not contact and not surface:
                warn('Don\'t forget to specify a force-displacement law')

            h_cp = np.linalg.norm(self.xe2 - self.xe1) / nb_cp
            x_cp = self.xe1 + offset * self.t

            for i in np.arange(2):
                l_Ax = np.dot((x_cp - self.bl_A.ref_point), -self.n)
                l_Ay = np.dot((x_cp - self.bl_A.ref_point), self.t)
                l_Bx = np.dot((x_cp - self.bl_B.ref_point), self.n)
                l_By = np.dot((x_cp - self.bl_B.ref_point), self.t)

                self.cps.append(
                    CP_2D(x_cp, l_Ax, l_Ay, l_Bx, l_By, self.angle - np.pi / 2, h_cp, self.b, contact=contact,
                          lin_geom=self.lin_geom))

                x_cp += (np.linalg.norm(self.xe2 - self.xe1) - 2 * offset) * self.t

        else:
            warn('The definition of the CPs is not valid')

    def set_lin_geom(self, lin_geom):

        self.lin_geom = lin_geom

        for cp in self.cps:
            cp.lin_geom = lin_geom

    def change_cps(self, nb_cp, offset=-1, surface=None, contact=None, weights=None):

        self.cps = []

        if offset == -1:  # Stress - strain

            if ((self.bl_A.material is None) or (self.bl_B.material is None)) and surface is None:
                warn('No material or surface law was defined')

            elif surface is None:

                h_cp = np.linalg.norm(self.xe2 - self.xe1) / nb_cp
                x_cp = self.xe1 + .5 * h_cp * self.t

                for i in np.arange(nb_cp):
                    x_cp = self.xe1 + (i + .5) * h_cp * self.t

                    l_Ax = np.dot((x_cp - self.bl_A.ref_point), -self.n)
                    l_Ay = np.dot((x_cp - self.bl_A.ref_point), self.t)
                    l_Bx = np.dot((x_cp - self.bl_B.ref_point), self.n)
                    l_By = np.dot((x_cp - self.bl_B.ref_point), self.t)

                    self.cps.append(
                        CP_2D(x_cp, l_Ax, l_Ay, l_Bx, l_By, self.angle - np.pi / 2, h_cp, self.b, surface=surface,
                              block_A=self.bl_A, block_B=self.bl_B, lin_geom=self.lin_geom))

            else:

                if isinstance(nb_cp, list):

                    d_tot = 0

                    for i, pos in enumerate(nb_cp):

                        if abs(pos) > 1: warn('Placing a CP outside a CF')

                        x_cp = .5 * (1 - pos) * self.xe1 + .5 * (1 + pos) * self.xe2

                        l_Ax = np.dot((x_cp - self.bl_A.ref_point), -self.n)
                        l_Ay = np.dot((x_cp - self.bl_A.ref_point), self.t)
                        l_Bx = np.dot((x_cp - self.bl_B.ref_point), self.n)
                        l_By = np.dot((x_cp - self.bl_B.ref_point), self.t)

                        if weights is not None:
                            if np.around(np.sum(weights), 10) != 1: warnings.warn('CP weights don\'t add up to one')
                            if len(weights) != len(nb_cp): warnings.warn(
                                'Number of weights is not coherent with number of CPs')
                            h_cp = weights[i] * np.linalg.norm(self.xe2 - self.xe1)
                            d_tot = np.sum(weights)
                        else:
                            if i == 0:
                                d = (nb_cp[0] + nb_cp[1]) / 2 + 1
                            elif i == len(nb_cp) - 1:
                                d = 1 - (nb_cp[i] + nb_cp[i - 1]) / 2
                            else:
                                d = (nb_cp[i + 1] - nb_cp[i - 1]) / 2
                            if d <= 0: warn('Distances for CPs are not coherent')

                            d_tot += d / 2
                            h_cp = np.linalg.norm(self.xe2 - self.xe1) * d / 2

                        self.cps.append(CP_2D(x_cp, l_Ax, l_Ay, l_Bx, l_By, self.angle - np.pi / 2, h_cp, self.b,
                                              surface=surface, lin_geom=self.lin_geom))

                    if np.around(d_tot, 10) != 1:
                        print(d_tot)
                        warn('Distances for CPs are not coherent')

                else:
                    h_cp = np.linalg.norm(self.xe2 - self.xe1) / nb_cp

                    for i in np.arange(nb_cp):
                        x_cp = self.xe1 + (i + .5) * h_cp * self.t

                        # print(x_cp)

                        l_Ax = np.dot((x_cp - self.bl_A.ref_point), -self.n)
                        l_Ay = np.dot((x_cp - self.bl_A.ref_point), self.t)
                        l_Bx = np.dot((x_cp - self.bl_B.ref_point), self.n)
                        l_By = np.dot((x_cp - self.bl_B.ref_point), self.t)

                        # print(h_cp, self.b)

                        self.cps.append(CP_2D(x_cp, l_Ax, l_Ay, l_Bx, l_By, self.angle - np.pi / 2, h_cp, self.b,
                                              surface=surface, lin_geom=self.lin_geom))

        elif offset >= 0:

            if nb_cp != 2:
                warn('Trying to use contact law with more than 2 CPs')

            if offset >= np.linalg.norm(self.xe2 - self.xe1) / 2:
                warn('Offset exceeds dimensions of CF')

            if not contact and not surface:
                warn('Don\'t forget to specify a force-displacement law')

            h_cp = np.linalg.norm(self.xe2 - self.xe1) / nb_cp
            x_cp = self.xe1 + offset * self.t

            for i in np.arange(2):
                l_Ax = np.dot((x_cp - self.bl_A.ref_point), -self.n)
                l_Ay = np.dot((x_cp - self.bl_A.ref_point), self.t)
                l_Bx = np.dot((x_cp - self.bl_B.ref_point), self.n)
                l_By = np.dot((x_cp - self.bl_B.ref_point), self.t)

                self.cps.append(
                    CP_2D(x_cp, l_Ax, l_Ay, l_Bx, l_By, self.angle - np.pi / 2, h_cp, self.b, contact=contact,
                          lin_geom=self.lin_geom))

                x_cp += (np.linalg.norm(self.xe2 - self.xe1) - 2 * offset) * self.t

        else:
            warn('The definition of the CPs is not valid')

    def add_reinforcement(self, pos, A, material=None, height=None):

        if isinstance(pos, float) or isinstance(pos, int): pos = [pos]

        if isinstance(A, float) or isinstance(A, int):
            list_A = []
            for i in range(len(pos)):
                list_A.append(A)
            A = list_A
        elif isinstance(A, list):
            if len(A) != len(pos): warn('List of areas is not coherent wiht list of positions')

        if isinstance(pos, list):

            for i, p in enumerate(pos):

                if abs(p) > 1: warn('Placing a CP outside a CF')

                x_cp = .5 * (1 - p) * self.xe1 + .5 * (1 + p) * self.xe2

                l_Ax = np.dot((x_cp - self.bl_A.ref_point), -self.n)
                l_Ay = np.dot((x_cp - self.bl_A.ref_point), self.t)
                l_Bx = np.dot((x_cp - self.bl_B.ref_point), self.n)
                l_By = np.dot((x_cp - self.bl_B.ref_point), self.t)

                if height is None:
                    h_cp = A[i]
                    b = 1
                else:
                    h_cp = height
                    b = A[i] / h_cp

                if material is None:
                    material = self.block_A.material.copy()

                self.cps.append(
                    CP_2D(x_cp, l_Ax, l_Ay, l_Bx, l_By, self.angle - np.pi / 2, h_cp, b, material=material,
                          lin_geom=self.lin_geom, reinf=True))

        else:
            warn('Reinforcement position should be either an int or a list')

    def commit(self):

        for cp in self.cps:
            cp.commit()

    # Method to revert committed changes to the contact pairs
    def revert_commit(self):

        for cp in self.cps:
            cp.revert_commit()

    def get_pf_glob(self, qf_glob):

        self.qf_glob = qf_glob.copy()

        self.get_pf_loc()

        self.pf_glob = np.zeros(6)

        T = T3x3(self.angle - np.pi / 2)

        self.pf_glob[:3] = np.transpose(T) @ self.pf_loc[:3]
        self.pf_glob[3:] = np.transpose(T) @ self.pf_loc[3:]

        return self.pf_glob

    def get_pf_loc(self):

        T = T3x3(self.angle - np.pi / 2)

        self.qf_loc = np.zeros(6)
        self.qf_loc[:3] = T @ self.qf_glob[:3]
        self.qf_loc[3:] = T @ self.qf_glob[3:]

        self.pf_loc = np.zeros(6)

        for cp in self.cps:

            pc_loc = cp.get_pc_loc(self.qf_loc)

            if not cp.to_ommit():
                self.pf_loc += pc_loc

    def get_kf_glob(self):

        self.get_kf_loc()

        self.kf_glob = np.zeros((6, 6))

        T = T3x3(self.angle - np.pi / 2)

        self.kf_glob[:3, :3] = np.transpose(T) @ self.kf_loc[:3, :3] @ T
        self.kf_glob[:3, 3:] = np.transpose(T) @ self.kf_loc[:3, 3:] @ T
        self.kf_glob[3:, :3] = np.transpose(T) @ self.kf_loc[3:, :3] @ T
        self.kf_glob[3:, 3:] = np.transpose(T) @ self.kf_loc[3:, 3:] @ T

        return self.kf_glob

    def get_kf_glob_LG(self):

        self.get_kf_loc_LG()

        self.kf_glob_LG = np.zeros((6, 6))

        T = T3x3(self.angle - np.pi / 2)

        self.kf_glob_LG[:3, :3] = np.transpose(T) @ self.kf_loc_LG[:3, :3] @ T
        self.kf_glob_LG[:3, 3:] = np.transpose(T) @ self.kf_loc_LG[:3, 3:] @ T
        self.kf_glob_LG[3:, :3] = np.transpose(T) @ self.kf_loc_LG[3:, :3] @ T
        self.kf_glob_LG[3:, 3:] = np.transpose(T) @ self.kf_loc_LG[3:, 3:] @ T

        return self.kf_glob_LG

    def get_kf_glob0(self):

        self.get_kf_loc0()

        self.kf_glob0 = np.zeros((6, 6))

        T = T3x3(self.angle - np.pi / 2)

        self.kf_glob0[:3, :3] = np.transpose(T) @ self.kf_loc0[:3, :3] @ T
        self.kf_glob0[:3, 3:] = np.transpose(T) @ self.kf_loc0[:3, 3:] @ T
        self.kf_glob0[3:, :3] = np.transpose(T) @ self.kf_loc0[3:, :3] @ T
        self.kf_glob0[3:, 3:] = np.transpose(T) @ self.kf_loc0[3:, 3:] @ T

        return self.kf_glob0

    def get_kf_loc(self):

        self.kf_loc = np.zeros((6, 6))

        for cp in self.cps:

            kc_loc = cp.get_kc_loc()
            # print(kc_loc[:3,:3])

            if not cp.to_ommit():
                self.kf_loc += kc_loc
        #     else: 
        #         print('One CP was omitted ')

        # print('kf_loc', self.kf_loc)        
        # print(self.kf_loc[3:,3:])

    def get_kf_loc_LG(self):

        self.kf_loc_LG = np.zeros((6, 6))

        for cp in self.cps:

            kc_loc_LG = cp.get_kc_loc_LG()

            if not cp.to_ommit():
                self.kf_loc_LG += kc_loc_LG

    def get_kf_loc0(self):

        self.kf_loc0 = np.zeros((6, 6))

        for cp in self.cps:
            kc_loc0 = cp.get_kc_loc0()

            self.kf_loc0 += kc_loc0

    def plot_cf(self, scale, disp1=0, disp2=0):

        x1 = np.array([self.xe1[0], self.xe2[0]]) + scale * disp1
        x2 = np.array([self.xe1[0], self.xe2[0]]) + scale * disp2
        y = np.array([self.xe1[1], self.xe2[1]])

        c = (self.xe1 + self.xe2) / 2

        import matplotlib as mpl
        tA = mpl.markers.MarkerStyle(marker=">")
        tA._transform = tA.get_transform().rotate(self.angle)

        plt.plot(x1, y, marker='.', markersize=0, color='red', linewidth=.4)
        plt.plot(x2, y, marker='.', markersize=0, color='red', linewidth=.4)
        plt.plot(c[0], c[1], marker=tA, markersize=5, color='red', linewidth=.75)

        for cp in self.cps:
            if not cp.to_ommit():
                cp.plot(scale)


def T3x3(a):
    return np.array([[np.cos(a), np.sin(a), 0],
                     [-np.sin(a), np.cos(a), 0],
                     [0, 0, 1]])
