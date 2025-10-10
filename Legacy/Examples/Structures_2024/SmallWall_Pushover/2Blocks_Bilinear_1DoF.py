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


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as ct
import ContactPair as cp

reload_modules()

N1 = np.array([0, 0])

H = .2
B = .2

kn = 1e3
ks = 1e3
fy = .6

# %%
RHO = 1 / (H * B * 9.81)

PATTERN = [[1], [1]]

St = st.Structure_2D()

# St.add_block(vertices, RHO, b=1)
St.add_wall(N1, B, H, PATTERN, RHO, b=1., material=None)

off = 0.00

St.make_nodes()

Contact = ct.Bilinear(kn, ks, fy / 2, 0.00)
print(Contact.disps['y'])
St.make_cfs(False, nb_cps=2, contact=Contact, offset=off)

# %% BCs and Forces

Total_mass = 0

St.fixNode(0, [0, 1, 2])
St.fixNode(1, [0, 2])

W = 1.
St.loadNode(1, 1, 1 * W)
# St.loadNode(1, 0, 1*W)
# St.loadNode(1, 1, -W, fixed=True)

St.plot_structure(scale=0, plot_cf=True, plot_forces=True)

# %%
Node = 1
d_end = 1e-2

LIST = np.array([])
LIST = np.append(LIST, np.linspace(0, d_end, 11))

LIST = LIST.tolist()

St.solve_dispcontrol(LIST, 0, Node, 1, tol=1e-4, filename=f'Wallet_DispControl', max_iter=100)
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
    U = hf['U_conv'][-2] * 1000
    P = hf['P_r_conv'][-2]

plt.figure(None, figsize=(6, 6))
plt.xlim(0, d_end * 1000)
plt.ylim(0, 2)

D = np.linspace(0, d_end, 100)

k_s = ks / 2
k_n = kn / 2

plt.plot(U, P, linewidth=.75, color='black', marker='*', label='HybriDFEM')
plt.grid()
plt.legend()

# %%
