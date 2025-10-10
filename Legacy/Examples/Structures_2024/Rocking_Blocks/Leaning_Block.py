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


def reload_modules():
    importlib.reload(st)
    importlib.reload(ct)
    importlib.reload(sp)
    importlib.reload(cp)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as ct
import Spring as sp
import ContactPair as cp

reload_modules()

N1 = np.array([0, 0])

H = 1
L = .5
B = 1

kn = 1e8
ks = 1e8
mu = .6

# %%
RHO = 1000.

vertices_s = np.array([[2 * L, 0],
                       [2 * L, H],
                       [L, H],
                       [L, 2 * H],
                       [0, 2 * H],
                       [0, 0]])

vertices_b = np.array([[2 * L, H],
                       [2 * L, 2 * H],
                       [L, 2 * H],
                       [L, H]])

St = st.Structure_2D()

St.add_block(vertices_s, RHO, b=1)
St.add_block(vertices_b, RHO, b=1)

St.make_nodes()
St.make_cfs(False, nb_cps=2, contact=ct.Coulomb_CD(kn, ks, mu), offset=0.02)

# %% BCs and Forces

Total_mass = 0

St.fixNode(0, [0, 1, 2])

W = St.list_blocks[1].m * 9.81
St.loadNode(1, 0, W)
St.loadNode(1, 1, -W, fixed=True)
W_c = W

St.plot_structure(scale=0, plot_cf=True, plot_forces=True)

# %%
Node = len(St.list_blocks) - 1
# %%
D0 = St.U[3 * Node]

LIST = np.linspace(0, 1e-2, 300)
LIST = np.append(LIST, np.linspace(1e-2, 1e-1, 100))
LIST = np.append(LIST, np.linspace(1e-1, 2.5e-1, 100))
# LIST = np.append(LIST, np.linspace(1e-6, 1e-5, 10))
# LIST = np.append(LIST, np.linspace(1e-5, 1e-4, 10))
# LIST = np.append(LIST, np.linspace(1e-4, 1e-3, 10))
# # LIST = np.append(LIST, np.linspace(0, 1, 200))
# LIST = np.append(LIST, np.linspace(1, 0, 300))

# LIST = np.append(LIST, np.linspace(1e-3, 1e-2, 100))
# LIST = np.append(LIST, np.linspace(1e-2, 1e-1, 100))
# LIST = np.append(LIST, np.linspace(1e-1, 2e-1, 100))

# LIST = np.append(LIST, np.linspace(1e-4, 1e-3, 100))
# LIST = np.append(LIST, np.linspace(1e-3, 1e-2, 100))

LIST = LIST.tolist()

# St.get_P_r()
# print(St.P_r)
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
    U = hf['U_conv'][-3] * 1000
    P = hf['P_r_conv'][-3] / W_c

plt.plot(U, P, linewidth=.75, color='black', marker='o')
plt.grid()

print(f'Load multiploer is {np.max(P)}')
# %%
