# -*- coding: utf-8 -*-
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
import pickle
from copy import deepcopy


def reload_modules():
    importlib.reload(st)
    importlib.reload(surf)
    importlib.reload(ct)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Surface as surf
import Contact as ct

reload_modules()

N1 = np.array([0, 0])

H_b = .2
L_b = .2
B = 1

kn = 1e6
ks = 1e6
mu = 0.65

# %%
RHO = 2000

PATTERN = [[1], [1]]

St = st.Structure_2D()

# St.add_block(vertices, RHO, b=1)
St.add_wall(N1, L_b, H_b, PATTERN, RHO, b=B, material=None)

St.make_nodes()
St.make_cfs(True, nb_cps=2, contact=ct.Coulomb(kn, ks, mu), offset=0.02)

# %% BCs and Forces

for i in range(len(PATTERN[0])):
    St.fixNode(i, [0, 1, 2])

# St.fixNode(1,[2])

for i in range(len(PATTERN[0]), len(St.list_blocks)):
    W = St.list_blocks[i].m * 9.81
    St.loadNode(i, 1, -W, fixed=True)
    St.loadNode(i, 0, W)

St.plot_structure(scale=0, plot_cf=True, plot_forces=True, plot_supp=True)

nb_modes = 10
n_incr = 2
Node = len(St.list_blocks) - 1
MAX_D = L_b / 8
delta_D = MAX_D / n_incr
STEPS = 200

LIST_D = np.array([])

for i in range(n_incr):
    LIST_D = np.append(LIST_D, np.linspace(-delta_D * i, delta_D * (i + 1), STEPS * (i + 1)))
    LIST_D = np.append(LIST_D, np.linspace(delta_D * (i + 1), -delta_D * (i + 1), 2 * STEPS * (i + 1)))
# print(LIST_D)
LIST_D = LIST_D.tolist()

plt.figure(None, figsize=(6, 6))
plt.plot(LIST_D)

St.solve_dispcontrol(LIST_D, 0, Node, 0, tol=1, filename='Sliding')

St.plot_structure(scale=1, plot_cf=False, plot_forces=False)

# %% Save results

Node = len(St.list_blocks) - 1

results = f'Sliding.h5'

with h5py.File(results, 'r') as hf:
    # Import what you need
    P = hf['P_r_conv'][3 * Node] / (0.2 * 0.2 * 9.81 * RHO)
    U = hf['U_conv'][3 * Node] * 1000

filename = 'Results_Modal_Deg.h5'

with h5py.File(filename, 'w') as hf:
    hf.create_dataset('U', data=U)
    hf.create_dataset('P', data=P)
    # hf.create_dataset('w', data=w)

print(np.around(U, 4))
print(np.around(P, 4))

# %%
# U0 = U[1]
# U = np.append(np.zeros(1), U[1:] - U0)
# P = np.append(np.zeros(1), P[1:])

import matplotlib.pyplot as plt

plt.figure(None, figsize=(6, 6))
plt.xlabel(r'Top horizontal displacement [mm]')
plt.ylabel(r'Load multiplier $\alpha$')
plt.grid()
# plt.xlim([0, 100])
# plt.ylim([0, 1.05])

plt.plot(U, P, color='black', linewidth=.75, label=r'$F-\Delta$')
plt.legend(fontsize=13)

# %%
