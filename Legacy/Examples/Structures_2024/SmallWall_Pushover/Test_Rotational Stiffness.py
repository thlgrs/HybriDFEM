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
    importlib.reload(cp)
    importlib.reload(mt)
    importlib.reload(sp)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as ct
import ContactPair as cp
import Material as mt
import Spring as sp

reload_modules()

N1 = np.array([0, 0])

H_b = .2
L_b = .2
B = 1

E = 30e9

# %%
RHO = 1000
vertices_1 = np.array([[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]], dtype=float)
vertices_2 = np.array([[0, 0.2], [0.2, 0.2], [0.2, 0.4], [0., 0.4]], dtype=float)

St = st.Structure_2D()

St.add_block(vertices_1, RHO, b=1, material=mt.Material(E, 0.2), ref_point=np.array([0.1, 0.1]))
St.add_block(vertices_2, RHO, b=1, material=mt.Material(E, 0.2), ref_point=np.array([0.1, 0.3]))

St.make_nodes()
# St.make_cfs(True, nb_cps=2, contact=ct.Coulomb(kn, ks, mu), offset=0.02)
St.make_cfs(True, nb_cps=1)

# %% BCs and Forces

Total_mass = 0

St.fixNode(0, [0, 1, 2])
# St.fixNode(1,[0,1])

W = .2 * .2 * RHO * 9.81
St.loadNode(1, 0, 0.1 * W)
St.loadNode(1, 1, -W, fixed=True)

St.plot_structure(scale=0, plot_cf=True, plot_forces=True)

# %%
Node = 1

LIST = np.linspace(0, 1e-1, 10)
# LIST = np.append(LIST, np.linspace(2e-3, 0, 100))
# LIST = np.append(LIST, np.linspace(0, -2e-3, 100))
# LIST = np.append(LIST, np.linspace(2e-4, 2e-3, 100))
# LIST = np.append(LIST, np.linspace(2e-3, 270e-3, 100))
# LIST = np.append(LIST,np.linspace(1e-3, 3e-1, 200))
LIST = LIST.tolist()

# St.get_P_r()

St.solve_dispcontrol(LIST, 0, Node, 0, tol=.1, filename=f'Wallet_DispControl', max_iter=1000)
# St.solve_forcecontrol(10, tol=0.1,filename=f'Wallet_ForceControl')

St.save_structure('Wallet')

# %% Plot structure
St.plot_structure(scale=1, plot_cf=True, plot_forces=False)
# %%
import matplotlib.pyplot as plt

file = 'Wallet_DispControl.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U = hf['U_conv'][-3] * 1000
    P = hf['P_r_conv'][-3] / 1000

plt.plot(U, P, linewidth=.75, color='black', marker='o')
plt.grid()
print(np.around(St.P_r, 3))
print(np.around(St.U, 3))
