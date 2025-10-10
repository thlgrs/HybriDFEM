# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 13:38:12 2025

@author: ibouckaert
"""

import numpy as np
import os
import h5py
import sys
import pathlib
import matplotlib.pyplot as plt

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as cont
import ContactPair as cp
import Surface as surf

save_path = os.path.dirname(os.path.abspath(__file__))

R_med = 200e-3  # m
t = 30e-3  # m

R_int = R_med - t / 2
R_ext = R_med + t / 2

mu = 1
rho = 2600  # kg/m^3

CPS = 2  # 2 CPs for each CF
kn = 40e6
ks = kn

St = st.Structure_2D()

N0 = np.array([0, 0], dtype=float)
St.add_arch(N0, np.pi / 20, np.pi * 19 / 20, R_med, 16, t, rho)

# adding the base support
N1 = np.array([-300e-3, -50e-3], dtype=float)
N2 = np.array([300e-3, -50e-3], dtype=float)
N3 = np.array([300e-3, 0], dtype=float)
N4 = np.array([-300e-3, 0], dtype=float)

vertices_1 = np.array([N1, N2, N3, N4])
St.add_block(vertices_1, 100.)

# adding the left part of the support arch
N5 = np.array([-R_ext, 0], dtype=float)
N6 = np.array([-R_ext * np.sin(np.pi * 9 / 20), R_ext * np.cos(np.pi * 9 / 20)], dtype=float)

vertices_2 = np.array([N0, N5, N6])
St.add_block(vertices_2, 100.)

# adding the right part of the support arch
N7 = np.array([R_ext, 0], dtype=float)
N8 = np.array([R_ext * np.sin(np.pi * 9 / 20), R_ext * np.cos(np.pi * 9 / 20)], dtype=float)

vertices_3 = np.array([N0, N7, N8])
St.add_block(vertices_3, 100.)

St.make_nodes()
St.make_cfs(True, nb_cps=10, offset=-1, surface=surf.Surface(kn, ks))  # linear geometry, point-contact model

# BCs and Forces

St.fixNode(16, [0, 1, 2])  # fix the base block in all the 3 dofs
St.fixNode(17, [0, 1, 2])  # fix the left support
St.fixNode(18, [0, 1, 2])  # fix the right support

for i in range(0, 16):
    M = St.list_blocks[i].m  # extracting the mass
    W = 9.81 * M
    St.loadNode(i, [1], -W, fixed=False)
    # St.loadNode(i, [0], -W, fixed=False)

St.get_P_r()
St.get_K_str()

St.plot_structure(plot_cf=False, scale=1)  # structure in the undeformed configuration

St.solve_forcecontrol(10, dir_name=save_path, filename='Results_arch_Stroobants_fc')

T = 0.18  # s
A_max = 0.28  # g
w = 2 * np.pi / T

St.solve_modal(4)
St.plot_modes(4, scale=0.1)

Meth = 'CDM'


def excitation(x): return -A_max * np.sin(w * x)


St.set_damping_properties(xsi=0., damp_type='RAYLEIGH')

St.solve_dyn_nonlinear(0.6, 1e-3, lmbda=excitation, Meth=Meth)
# duration of the simulation , time step
