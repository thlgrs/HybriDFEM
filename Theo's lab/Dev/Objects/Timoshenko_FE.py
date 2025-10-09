# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:52:50 2024

@author: ibouckaert
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt  # For creating plots
from .FE import FE


class Timoshenko_FE_2D(FE):

    def __init__(self, Node1, Node2, E, nu, b, h, lin_geom=True, rho=0.):

        self.N1 = Node1
        self.N2 = Node2

        self.Lx = self.N2[0] - self.N1[0]
        self.Ly = self.N2[1] - self.N1[1]

        self.nodes = [self.N1, self.N2]
        self.L = np.sqrt(self.Lx ** 2 + self.Ly ** 2)
        self.alpha = np.arctan2(self.Ly, self.Lx)

        self.d = np.zeros(6)

        self.E = E
        self.nu = nu
        self.chi = (6 + 5 * self.nu) / (5 * (1 + self.nu))

        self.lin_geom = lin_geom

        self.G = E / (2 * (1 + nu))

        self.A = b * h
        self.I = b * h ** 3 / 12
        self.b = b
        self.h = h

        self.connect = np.zeros(2)

        self.psi = self.E * self.I * self.chi / (self.G * self.A)

        c = np.cos(self.alpha)
        s = np.sin(self.alpha)

        self.dofs = np.zeros(6, dtype=int)

        self.r_C = np.array([[c, s, 0, 0, 0, 0],
                             [-s, c, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, c, s, 0],
                             [0, 0, 0, -s, c, 0],
                             [0, 0, 0, 0, 0, 1]])

        self.rho = rho

        # self.get_mass()

    def get_mass(self, no_inertia=False):

        m_node = self.b * self.h * self.L / 2 * self.rho
        if no_inertia:
            I_node = 0
        else:
            I_node = (((self.L / 2) ** 2 + self.h ** 2) * (1 / 12) + (self.L / 4) ** 2)

        self.mass = np.diag(m_node * np.array([1, 1, I_node, 1, 1, I_node]))

        return self.mass

    def make_connect(self, connect, node_number):

        self.connect[node_number] = connect

        if node_number == 0:
            self.dofs[:3] = np.array([0, 1, 2], dtype=int) + 3 * connect * np.ones(3, dtype=int)
        elif node_number == 1:
            self.dofs[3:] = np.array([0, 1, 2], dtype=int) + 3 * connect * np.ones(3, dtype=int)

    def get_k_glob(self):

        self.get_k_loc()

        self.k_glob = np.transpose(self.r_C) @ self.k_loc @ self.r_C

        return self.k_glob

    def get_k_glob0(self):

        self.get_k_loc0()

        self.k_glob0 = np.transpose(self.r_C) @ self.k_loc0 @ self.r_C

        return self.k_glob0

    def get_k_loc(self):

        self.get_k_bsc()

        if self.lin_geom:
            self.gamma_C = np.array([[-1, 0, 0, 1, 0, 0],
                                     [0, 1 / self.L, 1, 0, -1 / self.L, 0],
                                     [0, 1 / self.L, 0, 0, -1 / self.L, 1]])

        self.k_loc_mat = np.transpose(self.gamma_C) @ self.k_bsc @ self.gamma_C

        if not self.lin_geom:
            self.G1, self.G23 = self.G1_G23(self.l, self.beta)

            self.k_loc_geom = self.G1 * self.p_bsc[0] + self.G23 * (self.p_bsc[1] + self.p_bsc[2])

        if self.lin_geom:
            self.k_loc = deepcopy(self.k_loc_mat)
        else:
            self.k_loc = self.k_loc_mat + self.k_loc_geom

    def get_k_loc0(self):

        self.get_k_bsc()

        self.gamma_C = np.array([[-1, 0, 0, 1, 0, 0],
                                 [0, 1 / self.L, 1, 0, -1 / self.L, 0],
                                 [0, 1 / self.L, 0, 0, -1 / self.L, 1]])

        self.k_loc_mat0 = np.transpose(self.gamma_C) @ self.k_bsc @ self.gamma_C

        self.k_loc0 = deepcopy(self.k_loc_mat0)

    def get_k_bsc(self):

        l = self.L
        ps = self.psi

        self.k_bsc_ax = self.E * self.A * np.array([[1 / l, 0, 0],
                                                    [0, 0, 0],
                                                    [0, 0, 0]])

        self.k_bsc_fl = self.E * self.I / (l * (l * l + 12 * ps)) * np.array([[0, 0, 0],
                                                                              [0, 4 * l * l + 12 * ps,
                                                                               2 * l * l - 12 * ps],
                                                                              [0, 2 * l * l - 12 * ps,
                                                                               4 * l * l + 12 * ps]])

        self.k_bsc = self.k_bsc_ax + self.k_bsc_fl

    def get_p_glob(self, q_glob):

        self.q_glob = q_glob

        self.p_glob = np.zeros(6)

        self.q_loc = self.r_C @ self.q_glob

        self.get_p_loc()

        self.p_glob = np.transpose(self.r_C) @ self.p_loc

        return self.p_glob

    def get_p_loc(self):

        self.get_p_bsc()

        self.p_loc = np.transpose(self.gamma_C) @ self.p_bsc

    def get_p_bsc(self):

        if self.lin_geom:
            self.gamma_C = np.array([[-1, 0, 0, 1, 0, 0],
                                     [0, 1 / self.L, 1, 0, -1 / self.L, 0],
                                     [0, 1 / self.L, 0, 0, -1 / self.L, 1]])
            self.q_bsc = self.gamma_C @ self.q_loc

        else:
            self.l = np.sqrt((self.L + self.q_loc[3] - self.q_loc[0]) ** 2 + (self.q_loc[4] - self.q_loc[1]) ** 2)
            self.beta = np.arctan2((self.q_loc[4] - self.q_loc[1]), (self.L + self.q_loc[3] - self.q_loc[0]))

            c = np.cos(self.beta)
            s = np.sin(self.beta)

            cl = c / self.l
            sl = s / self.l

            self.gamma_C = np.array([[-c, -s, 0, c, s, 0],
                                     [-sl, cl, 1, sl, -cl, 0],
                                     [-sl, cl, 0, sl, -cl, 1]])
            self.q_bsc = np.zeros(3)

            self.q_bsc[0] = self.l - self.L
            self.q_bsc[1] = self.q_loc[2] - self.beta
            self.q_bsc[2] = self.q_loc[5] - self.beta

        self.get_k_bsc()

        self.p_bsc = self.k_bsc @ self.q_bsc

    def G1_G23(self, l, beta):
        sb = np.sin(beta)
        cb = np.cos(beta)

        G_1 = (1 / l) * np.array([[sb ** 2, -cb * sb, 0, -sb ** 2, cb * sb, 0],
                                  [-cb * sb, cb ** 2, 0, cb * sb, -cb ** 2, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [-sb ** 2, cb * sb, 0, sb ** 2, -cb * sb, 0],
                                  [cb * sb, -cb ** 2, 0, -cb * sb, cb ** 2, 0],
                                  [0, 0, 0, 0, 0, 0]])

        G_23 = (1 / l ** 2) * np.array([[-2 * cb * sb, cb ** 2 - sb ** 2, 0, 2 * cb * sb, sb ** 2 - cb ** 2, 0],
                                        [cb ** 2 - sb ** 2, 2 * cb * sb, 0, sb ** 2 - cb ** 2, -2 * cb * sb, 0],
                                        [0, 0, 0, 0, 0, 0],
                                        [2 * cb * sb, sb ** 2 - cb ** 2, 0, -2 * cb * sb, cb ** 2 - sb ** 2, 0],
                                        [sb ** 2 - cb ** 2, -2 * cb * sb, 0, cb ** 2 - sb ** 2, 2 * cb * sb, 0],
                                        [0, 0, 0, 0, 0, 0]])

        return G_1, G_23

    def PlotDefShapeElem(self, defs, scale=1):

        disc = 100

        defs_loc = self.r_C @ defs

        x_loc = np.linspace(0, self.L, disc)
        y_loc = np.zeros(disc)

        phi1 = scale * defs_loc[1] * (1 - 3 * x_loc ** 2 / self.L ** 2 + 2 * x_loc ** 3 / self.L ** 3)
        phi2 = scale * defs_loc[2] * (x_loc - 2 * x_loc ** 2 / self.L + x_loc ** 3 / self.L ** 2)
        phi3 = scale * defs_loc[4] * (3 * x_loc ** 2 / self.L ** 2 - 2 * x_loc ** 3 / self.L ** 3)
        phi4 = scale * defs_loc[5] * (-x_loc ** 2 / self.L + x_loc ** 3 / self.L ** 2)

        y_loc += phi1 + phi2 + phi3 + phi4
        x_loc += np.linspace(scale * defs_loc[0], 0, disc) + np.linspace(0, scale * defs_loc[3], disc)

        # Rotation to align with element in global axes
        x_glob = np.zeros(disc)
        y_glob = np.zeros(disc)

        s, c = self.Ly / self.L, self.Lx / self.L

        r = np.array([[c, -s], [s, c]])

        for i in range(disc):
            x_glob[i], y_glob[i] = r @ np.array([x_loc[i], y_loc[i]])

        # Positioning at correct position 

        x_undef = np.linspace(self.N1[0], self.N2[0], disc)
        y_undef = np.linspace(self.N1[1], self.N2[1], disc)

        x_def = x_glob + self.N1[0]
        y_def = y_glob + self.N1[1]

        plt.plot(x_def, y_def, linewidth=1.5, color='black')
        plt.plot(x_def[0], y_def[0], color='black', marker='o', markersize=3)
        plt.plot(x_def[-1], y_def[-1], color='black', marker='o', markersize=3)

    def PlotUndefShapeElem(self):

        disc = 10

        x_loc = np.linspace(0, self.L, disc)
        y_loc = np.zeros(disc)

        # Rotation to align with element in global axes
        x_glob = np.zeros(disc)
        y_glob = np.zeros(disc)

        s, c = self.Ly / self.L, self.Lx / self.L

        r = np.array([[c, -s], [s, c]])

        for i in range(disc):
            x_glob[i], y_glob[i] = r @ np.array([x_loc[i], y_loc[i]])

        # Positioning at correct position 

        x_undef = np.linspace(self.N1[0], self.N2[0], disc)
        y_undef = np.linspace(self.N1[1], self.N2[1], disc)

        plt.plot(x_undef, y_undef, linewidth=1.5, color='black')
        plt.plot(x_undef[0], y_undef[0], color='black', marker='o', markersize=3)
        plt.plot(x_undef[-1], y_undef[-1], color='black', marker='o', markersize=3)

        return x_undef, y_undef
