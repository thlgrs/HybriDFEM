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
import Surface as surf
import Spring as sp
import ContactPair as cp

reload_modules()

N1 = np.array([0, 0])

H_b = .175
L_b = .4
B = 1

kn = 1e10
ks = 1e10
mu = .65
psi = mu
r_b = 0.02

# %%
RHO = 1000.

Blocks_Bed = 3
Blocks_Head = 6

Full_e = [.5, 1., 1., .5]
Full_u = [1.] * 3

Wind_u = [1., -1., 1.]
Wind_e = [.5, .5, -1., .5, .5]

PATTERN = [Full_e, Full_u, Full_e, \
           Wind_u, Wind_e, Wind_u, Wind_e, Wind_u, \
           Full_e, Full_u, Full_e, \
           Wind_u, Wind_e, Wind_u, Wind_e, Wind_u, \
           Full_e, Full_u, Full_e]

vertices = np.array([[Blocks_Bed * L_b, -H_b],
                     [Blocks_Bed * L_b, 0],
                     [0, 0],
                     [0, -H_b]])

St = st.Structure_2D()

St.add_block(vertices, RHO, b=1)
St.add_wall(N1, L_b, H_b, PATTERN, RHO, b=B, material=None)

St.make_nodes()
nb_cps = [-1, 1]
St.make_cfs(True, nb_cps=2, contact=ct.Coulomb(kn, ks, mu, psi=psi), offset=r_b)

# %% BCs and Forces

Total_mass = 0

St.fixNode(0, [0, 1, 2])

St.plot_structure(scale=1, plot_cf=False, plot_forces=False, plot_supp=True)

Node = len(St.list_blocks) - 1

for i in range(1, len(St.list_blocks)):
    W = St.list_blocks[i].m * 10
    St.loadNode(i, 0, W)
    St.loadNode(i, 1, -W, fixed=True)
    W_c = W

St.plot_structure(scale=0, plot_cf=True, plot_forces=False, plot_supp=False)

# %%
LIST = np.linspace(0, 1e-3, 3000)
LIST = LIST.tolist()

# St.get_P_r()

St.solve_dispcontrol(LIST, 0, Node, 0, tol=1e-3, filename=f'Wall5_rb={r_b}_psi={psi}', max_iter=100)
# St.solve_forcecontrol(1000)
St.save_structure('Wallet')

# %% Plot structure

St.plot_structure(scale=500, plot_cf=False, plot_forces=False)
# %%
import matplotlib.pyplot as plt

file = f'Wall5_rb={r_b}_psi={psi}.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U = hf['U_conv'][-3] * 1000
    P = hf['P_r_conv'][-3] / W_c
plt.figure(None, dpi=400, figsize=(6, 6))
plt.plot(U, P, linewidth=0, color='black', marker='o', markersize=2, label='HybriDFEM')
plt.xlabel('Control displacement [mm]')
plt.ylabel('Load multiplier')
# plt.plot(np.array([0,110]), 0.643*np.ones(2), color='red', linestyle=':', label='Ferris & Tin-Loi, 2001')
# plt.xlim((0,20))
# plt.ylim((0,1))
plt.grid(True)
plt.legend()

print(f'Load multiplier is {np.max(P)}')
# %%
