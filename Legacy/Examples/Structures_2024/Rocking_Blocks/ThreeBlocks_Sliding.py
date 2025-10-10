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
import Contact as ct
import Spring as sp
import ContactPair as cp

N1 = np.array([0, 0])

H = .5
L = .5
B = 1

kn = 1e7
ks = 1e7
mu = 1

# %%
RHO = 1000.

vertices_1 = np.array([[L, 0],
                       [L, H],
                       [0, H],
                       [0, 0]])

vertices_2 = np.array([[L, H],
                       [L, 2 * H],
                       [0, 2 * H],
                       [0, H]])

vertices_3 = np.array([[L, 2 * H],
                       [L, 3 * H],
                       [0, 3 * H],
                       [0, 2 * H]])

St = st.Structure_2D()

St.add_block(vertices_1, RHO, b=1)
St.add_block(vertices_2, RHO, b=1)
St.add_block(vertices_3, RHO, b=1)

St.make_nodes()
St.make_cfs(False, nb_cps=2, contact=ct.Coulomb(kn, ks, mu), offset=0.00)

# %% BCs and Forces

Total_mass = 0

St.fixNode(0, [0, 1, 2])
St.fixNode(2, [0, 1, 2])
# St.fixNode(1,2)


W = St.list_blocks[1].m * 9.81
St.loadNode(1, 0, W)
St.loadNode(1, 1, -W, fixed=True)
W_c = W

St.plot_structure(scale=0, plot_cf=True, plot_forces=True)

# %%
Node = len(St.list_blocks) - 2
# %%
D0 = St.U[3 * Node]

LIST = np.linspace(0, 1e-2, 100)
LIST = LIST.tolist()

# St.solve_forcecontrol(200,tol=.1)
St.solve_dispcontrol(LIST, 0, Node, 0, tol=1e-4, filename=f'Wallet_DispControl', max_iter=100)
# St.solve_workcontrol(1000, tol=1e-3, filename=f'Wallet_DispControl',max_iter=100)
St.save_structure('Wallet')

# %% Plot structure
St.plot_structure(scale=1, plot_cf=False, plot_forces=False)
# %%
import matplotlib.pyplot as plt

file = 'Wallet_DispControl.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U = hf['U_conv'][3 * Node] * 1000
    P = hf['P_r_conv'][3 * Node] / W_c

plt.plot(U, P, linewidth=.75, color='black', marker='o')
plt.grid()

print(f'Load multiploer is {np.max(P)}')
# %%
