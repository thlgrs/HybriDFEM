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

kn = 1e8
mu = .65
psi = 0.01
r_b = 0.02
n_cp = 3

# %%
RHO = 1000.

Blocks_Bed = 5
Blocks_Head = 7

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
    if i % 2 != 0:
        PATTERN.append(Line2)
    else:
        PATTERN.append(Line1)

St = st.Structure_2D()

# St.add_block(vertices, RHO, b=1)
St.add_wall(N1, L_b, H_b, PATTERN, RHO, b=B, material=None)
St.make_nodes()

lim = (2 / 0.175) * r_b - 1
nb_cps = np.linspace(lim, -lim, n_cp)
print(nb_cps)
St.make_cfs(False, nb_cps=nb_cps.tolist(), surface=surf.Coulomb(kn, kn, mu, psi=psi), offset=-1)

for cf in St.list_cfs:
    if abs(cf.angle) < 1e-10:
        lim = (2 / 0.2) * r_b - 1
        nb_cps = np.linspace(lim, -lim, n_cp)
        cf.change_cps(nb_cp=nb_cps.tolist(), surface=surf.Coulomb(kn, kn, mu, psi=psi), offset=-1)

# %% BCs and Forces
for i in range(len(PATTERN[0])):
    St.fixNode(i, [0, 1, 2])

for i in range(len(PATTERN[0]), len(St.list_blocks)):
    W = St.list_blocks[i].m * 10
    # St.loadNode(i, 0, W)
    St.loadNode(i, 1, -W)

St.solve_forcecontrol(10, tol=W * 1e-5)
St.reset_loading()

for i in range(1, len(St.list_blocks)):
    W = St.list_blocks[i].m * 10
    St.loadNode(i, 0, W)
    St.loadNode(i, 1, -W, fixed=True)

St.plot_structure(scale=0, plot_cf=True, plot_forces=True, plot_supp=True)
St.save_structure('F&TL_Wall1')

# %% Simulation params

LIST = np.array([0])
# LIST = np.append(LIST, np.linspace(LIST[-1], 1e-5, 100))
LIST = np.append(LIST, np.linspace(LIST[-1], 4e-1, 50000))

LIST = LIST.tolist()
Node = len(St.list_blocks) - 1

St.solve_dispcontrol(LIST, 0, Node, 0, tol=10, filename=f'Wall1_rb={r_b}_psi={psi}', max_iter=100)

# %% Plot structure
St.plot_structure(scale=1, plot_cf=False, plot_forces=False)
