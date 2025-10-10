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
L_b = .4
B = 1

kn = 1e8
ks = 1e8
mu = 0.65

# %%
RHO = 1000

Blocks_Bed = 3
Blocks_Head = Blocks_Bed

Line1 = []
Line2 = []

for i in range(Blocks_Bed):
    Line1.append(1.)
    if i == 0:
        Line2.append(.5)
        Line2.append(1.)
    elif i == Blocks_Bed - 1:
        Line2.append(.5)
    else:
        Line2.append(1.)

vertices = np.array([[Blocks_Bed * L_b, -H_b],
                     [Blocks_Bed * L_b, 0],
                     [0, 0],
                     [0, -H_b]])

PATTERN = []

for i in range(Blocks_Head):
    if i % 2 == 0:
        PATTERN.append(Line2)
    else:
        PATTERN.append(Line1)

St = st.Structure_2D()

St.add_block(vertices, RHO, b=1)
St.add_wall(N1, L_b, H_b, PATTERN, RHO, b=B, material=None)

St.make_nodes()
St.make_cfs(False, nb_cps=2, contact=ct.Coulomb(kn, ks, mu), offset=0.02)

# %% BCs and Forces

# for i in range(len(PATTERN[0])): 
St.fixNode(0, [0, 1, 2])

for i in range(1, len(St.list_blocks)):
    W = St.list_blocks[i].m * 9.81
    St.loadNode(i, 0, W)
    St.loadNode(i, 1, -W, fixed=True)

Node = len(St.list_blocks) - 1
# St.fixNode(Node,0)

St.plot_structure(scale=0, plot_cf=True, plot_forces=True, plot_supp=True)
St.save_structure('Wall/Plastic_Frame')

nb_modes = 5
n_incr = 10
delta_D = 0.01e-3
STEPS = 20

w = np.zeros([nb_modes, n_incr + 1])

St.solve_modal(filename='Wall/Step_0_Modal', save=True)
w[:, 0] = St.eig_vals[:nb_modes]
print(f'Natural frequencies for step 0: {np.around(St.eig_vals[:nb_modes], 3)}')
St.plot_modes(nb_modes, scale=-2, save=True, folder=f'Wall/Step_0', show=False)

D_end = 0.

for i in range(n_incr):
    with open(f'Wall/Plastic_Frame.pkl', 'rb') as file:
        St = pickle.load(file)

    LIST_D = np.linspace(0, delta_D * (i + 1), STEPS)
    print(LIST_D[-1])
    # LIST_D = np.append(LIST_D)
    LIST_D = LIST_D.tolist()
    St.solve_dispcontrol(LIST_D, 0, Node, 0, tol=1e-6, filename=f'Wall/Step_{i + 1}_DispControl', max_iter=1000)

    with h5py.File(f'Wall/Step_{i + 1}_DispControl.h5', 'r') as hf:
        lam = hf['Lambda'][-1]
    print(lam)

    # St.P[St.dof_free] = deepcopy(St.P_r[St.dof_free])
    # LIST_F = np.append(np.ones(1)*lam, np.linspace(lam, 0, STEPS[i]*5))

    LIST_F = np.linspace(lam * .95, 0, STEPS)
    LIST_F = LIST_F.tolist()
    St.solve_forcecontrol(LIST_F, tol=1e-6, filename=f'Wall/Step_{i + 1}_ForceControl', max_iter=1000)
    St.save_structure('Wall/Plastic_Frame')
    St.plot_structure(scale=1, plot_cf=False, plot_forces=False)

    St.solve_modal(filename=f'Wall/Step_{i + 1}_Modal_C', save=True)
    print(f'Natural frequencies for step {i + 1}: {np.around(St.eig_vals[:nb_modes], 3)}')
    w[:, i + 1] = St.eig_vals[:nb_modes]

    St.plot_modes(nb_modes, scale=-2, save=True, folder=f'Wall/Step_{i + 1}', show=False)

# %% Save results

P = np.array([])
U = np.array([])
for i in range(1, n_incr + 1):
    results = f'Wall/Step_{i}_DispControl.h5'

    with h5py.File(results, 'r') as hf:
        # Import what you need
        P = np.append(P, hf['P_r_conv'][3 * Node] / (.2 * .2 * 9.81 * RHO))
        U = np.append(U, hf['U_conv'][3 * Node] * 1000)

    results = f'Wall/Step_{i}_ForceControl.h5'

    with h5py.File(results, 'r') as hf:
        # Import what you need
        P = np.append(P, hf['P_r_conv'][3 * Node] / (.2 * .2 * 9.81 * RHO))
        U = np.append(U, hf['U_conv'][3 * Node] * 1000)

filename = 'Results_Modal_Deg.h5'

with h5py.File(filename, 'w') as hf:
    hf.create_dataset('U', data=U)
    hf.create_dataset('P', data=P)
    hf.create_dataset('w', data=w)

# %%
# U0 = U[1]
# U = np.append(np.zeros(1), U[1:] - U0)
# P = np.append(np.zeros(1), P[1:])

import matplotlib.pyplot as plt

plt.figure(None, figsize=(6, 6))
plt.xlabel(r'Top horizontal displacement [mm]')
plt.ylabel(r'Load multiplier $\alpha$')
plt.grid()
plt.xlim([0, .2])
plt.ylim([0, 1.05])

plt.plot(U, P, color='black', linewidth=.75, label=r'$F-\Delta$')
plt.legend(fontsize=13)

# %%
