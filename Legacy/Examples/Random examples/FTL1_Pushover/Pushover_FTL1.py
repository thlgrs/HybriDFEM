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


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as ct

reload_modules()

N1 = np.array([0, 0])

H_b = .175
L_b = .4
B = 1

kn = 1e9
ks = 1e9
mu = .65

# %%
RHO = 1000

Blocks_Bed = 5
Blocks_Head = 6

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

Total_mass = 0

St.fixNode(0, [0, 1, 2])
# St.fixNode(1,[2])

for i in range(1, len(St.list_blocks)):
    W = St.list_blocks[i].m * 9.81
    St.loadNode(i, 0, 0.1 * W)
    St.loadNode(i, 1, -W, fixed=True)

St.plot_structure(scale=0, plot_cf=True, plot_forces=False)

# %%
Node = len(St.list_blocks) - 1
print(Node)

LIST = np.linspace(0, 3e-3, 100)
# LIST = np.append(LIST, np.linspace(2e-4, 1e-3, 200))
# LIST = np.append(LIST, np.linspace(1e-3, 5e-3, 200))
# LIST = np.append(LIST,np.linspace(1e-3, 3e-1, 200))
LIST = LIST.tolist()

# St.get_P_r()

St.solve_dispcontrol(LIST, 0, Node, 0, tol=1, filename=f'Wallet_DispControl')
St.save_structure('Wallet')

# %% Plot structure
St.plot_structure(scale=100, plot_cf=True, plot_forces=False)
# %%
