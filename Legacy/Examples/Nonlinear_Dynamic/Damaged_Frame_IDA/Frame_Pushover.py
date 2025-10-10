# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""
# %% Library imports

import numpy as np
import os
import h5py
import sys
import pathlib
import importlib
from copy import deepcopy
import pickle
import pandas as pd


def reload_modules():
    importlib.reload(st)
    importlib.reload(mat)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat

reload_modules()

# %% Structure parameters

N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, 1], dtype=float)
N3 = np.array([0, 2], dtype=float)
N4 = np.array([0, 3], dtype=float)
N5 = np.array([3, 3], dtype=float)
# %%
N6 = np.array([3, 2], dtype=float)

N7 = np.array([3, 1], dtype=float)
N8 = np.array([3, 0], dtype=float)

B_b = .2
H_b = .2
H_c = .2 * 2 ** (1 / 3)

CPS = 25
BLOCKS = 30

E = 30e9
NU = 0.0
FY = 20e6

RHO = 2000
LIN = True

MAT = mat.Bilinear_Mat(E, NU, FY, 0.00)
# MAT = mat.Plastic_Stiffness_Deg(E,NU,FY)
# MAT = mat.Plastic_Mat(E,NU,FY)
# MAT = mat.Material(E,NU)
# MAT.plot_stress_strain()
# %% Building structure

St = st.Structure_2D()

St.add_fe(N1, N2, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_fe(N2, N3, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_fe(N3, N4, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_beam(N4, N5, BLOCKS, H_b, RHO, b=B_b, material=MAT)
St.add_fe(N5, N6, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_fe(N6, N7, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_fe(N7, N8, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)

St.make_nodes()
St.make_cfs(LIN, nb_cps=CPS)

# %% BCs and Forces

St.fixNode(N1, [0, 1])
St.fixNode(N8, [0, 1])
# Apply self-weight statically
St.get_M_str()
W_tot = 0

for i in range(len(St.list_blocks)):
    m = St.list_blocks[i].m
    W = 9.81 * m
    Node = St.get_node_id(St.list_blocks[i].ref_point)
    St.loadNode(Node, 1, -W, fixed=True)
    St.loadNode(Node, 0, -W)

for i in range(len(St.list_fes)):
    m = St.list_fes[i].mass[0, 0]
    W = 9.81 * m
    Node1 = St.get_node_id(St.list_fes[i].N1)
    Node2 = St.get_node_id(St.list_fes[i].N2)
    St.loadNode(Node1, 0, -W)
    St.loadNode(Node2, 0, -W)
    St.loadNode(Node1, 1, -W, fixed=True)
    St.loadNode(Node2, 1, -W, fixed=True)

W = 100e3
Node = St.get_node_id(N4)
# print(Node)
# St.loadNode(Node, 0, W)
# St.loadNode(Node, 1, -100e3, fixed=True)
# St.loadNode(St.get_node_id(N5), 1, -100e3, fixed=True)

St.plot_structure(scale=0, plot_cf=False, plot_forces=False, plot_supp=False, save='Frame_undef.eps')

St.solve_dispcontrol(200, 500e-3, Node, 0, tol=.1)
# St.solve_forcecontrol(100)
St.plot_structure(scale=1, plot_cf=False, plot_forces=False, plot_supp=True, save='Frame_def.eps')

# %% Plot
results = f'Results_DispControl.h5'
Node = 8
with h5py.File(results, 'r') as hf:
    # Import what you need
    P = hf['P_r_conv'][3 * Node]
    U = hf['U_conv'][3 * Node] * 1000

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 13

plt.figure(None, figsize=(6, 6))
plt.xlabel(r'Top horizontal displacement [mm]')
plt.ylabel(r'$F$ [kN]')
plt.grid()
W = 2000 * 9.81 * .2 * .2 * (3 / 29)
plt.plot(U, P / W)
# plt.xlim([0, 100])
# plt.ylim([0, 1.05])
print(max(P / W))
