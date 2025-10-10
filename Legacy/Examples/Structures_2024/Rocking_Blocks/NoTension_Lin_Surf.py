# %% -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""
import numpy as np
import os
import h5py
import sys
import pathlib
import importlib

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Surface as surf
import ContactPair as cp

N1 = np.array([0, 0])

H = 2
B = 2

nb_cps = 10

kn = 2e3
ks = 2e3

# %%
RHO = 1 / (H * B * 9.81)

PATTERN = [[1], [1]]

St = st.Structure_2D()

# St.add_block(vertices, RHO, b=1)
St.add_wall(N1, B, H, PATTERN, RHO, b=1., material=None)

nb_cps = [-1, -.5, 0, .5, 1]
St.make_nodes()
St.make_cfs(False, nb_cps=nb_cps, surface=surf.NoTension_CD(kn, ks), offset=-1)

# %% BCs and Forces

Total_mass = 0

St.fixNode(0, [0, 1, 2])

W = 1.

# St.loadNode(1, 1, -W)
# St.solve_forcecontrol(3)

# St.reset_loading()

St.loadNode(1, 0, W)
St.loadNode(1, 1, -W, fixed=True)
# St.loadNode(2, 0, W)
# St.loadNode(2, 1, -W, fixed=True)

St.plot_structure(scale=0, plot_forces=False)

# %% Simulation
Node = [1]
d_end = .2e-1

LIST = np.array([])
LIST = np.append(LIST, np.linspace(0, d_end, 100))

LIST = LIST.tolist()

St.solve_dispcontrol(LIST, 0, Node, 0, tol=1e-4, filename=f'NoTension_Lin_Surf', max_iter=10)

St.save_structure('Wallet')

# %% Plot structure
St.plot_structure(scale=1, plot_cf=True, plot_forces=False)
# %% Plot Force-displacement curve
import matplotlib.pyplot as plt

file = 'NoTension_Lin_Surf.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U = hf['U_conv'][-3] * 1000
    P = hf['P_r_conv'][-3]

plt.figure(None, figsize=(6, 6))
plt.xlim(0, d_end * 1000)
plt.ylim(0, 2)

ks = 1000
kn = 1000

solver = np.array([[4 * ks, 4 * ks, -1],
                   [0, 4 * ks, 0],
                   [4 * ks, 6 * ks, 0]])
target = np.array([0, -W, 0])

elastic_lim = np.linalg.solve(solver, target)
print(elastic_lim)

solver = np.array([[4 * ks, 4 * ks, -1],
                   [0, 1, 0],
                   [4, 5, 0]])

target = np.array([-elastic_lim[2], 3 * elastic_lim[1], elastic_lim[1]])

plastic_lim = np.linalg.solve(solver, target)
print(plastic_lim)

D_el = np.zeros(100)
F_el = np.linspace(0, elastic_lim[2], 100)

F_1 = np.linspace(0, plastic_lim[2] - elastic_lim[2], 1000)

K_el = ks * np.array([[4, 0, 4],
                      [0, 4, 0],
                      [4, 0, 6]])

K_1 = np.array([[4, 0, 4],
                [0, 3, 1],
                [4, 1, 5]]) * ks

plt.plot(U, P, linewidth=.75, color='black', marker='.', label='HybriDFEM')
plt.plot(elastic_lim[0] * 1000, elastic_lim[2], 'rx', label='First spring yields', markersize=10)
plt.plot((elastic_lim[0] + plastic_lim[0]) * 1000, plastic_lim[2], 'rx', label='First spring yields', markersize=10)

plt.legend()
plt.grid(True)
