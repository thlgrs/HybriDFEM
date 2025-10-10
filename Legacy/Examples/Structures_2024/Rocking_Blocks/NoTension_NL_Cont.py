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

kn = 2e3
ks = 2e3

# %%
RHO = 1 / (H * B * 9.81)

PATTERN = [[1], [1]]

St = st.Structure_2D()

St.add_wall(N1, B, H, PATTERN, RHO, b=1., material=None)

off = 0.00

St.make_nodes()
St.make_cfs(False, nb_cps=2, contact=ct.NoTension_EP(kn, ks), offset=off)

# %% BCs and Forces

Total_mass = 0

St.fixNode(0, [0, 1, 2])

W = 1.
St.loadNode(1, 0, 1 * W)
St.loadNode(1, 1, -W, fixed=True)

St.plot_structure(scale=0, plot_cf=True, plot_forces=True)

# %%
Node = 1
d_end = 1.5e-2

LIST = np.array([])
LIST = np.append(LIST, np.linspace(0, d_end, 11))
LIST = np.append(LIST, np.linspace(d_end, -d_end, 21))
LIST = np.append(LIST, np.linspace(-d_end, d_end, 10))
LIST = np.append(LIST, np.linspace(d_end, -d_end, 10))
LIST = (LIST).tolist()

St.solve_dispcontrol(LIST, 0, Node, 0, tol=1e-4, filename=f'Wallet_DispControl', max_iter=10)

St.save_structure('Wallet')

# %% Plot structure
St.plot_structure(scale=1, plot_cf=True, plot_forces=False)
# %%
import matplotlib.pyplot as plt

file = 'Wallet_DispControl.h5'
with h5py.File(file, 'r') as hf:
    print(file)
    U = hf['U_conv'][-3] * 1000
    P = hf['P_r_conv'][-3]

# %% Analytical
plt.figure(None, figsize=(6, 6))
plt.xlim(-d_end * 1000, d_end * 1000)
plt.ylim(-1.5, 1.5)

D = np.linspace(0, d_end, 100)

k_s = ks / 2
k_n = kn / 2

K_twocontacts = np.array([[2 * k_s, 0, H * k_s],
                          [0, 2 * k_n, 0],
                          [H * k_s, 0, H ** 2 * k_s / 2 + k_n * (B - 2 * off) ** 2 / 2]])

K_onecontact = K_twocontacts / 2

F_twocontacts = np.linspace(0, 1, 10)
D_twocontacts = np.zeros(10)

for i in range(10):
    F = np.array([F_twocontacts[i], W, 0])
    D_twocontacts[i] = np.linalg.solve(K_twocontacts, F)[0]

plt.plot(D_twocontacts * 1000, F_twocontacts, linewidth=.75, color='red', marker=None, label='Bending')

Theta = np.arctan2(H / 2, B / 2 - off)
D_theta_max = np.pi / 2 - Theta

Rotation_Block = np.linspace(0, D_theta_max, 100)
L_diag = np.sqrt((H / 2) ** 2 + (B / 2 - off) ** 2)
U_h = L_diag * np.cos(Theta) - L_diag * np.cos(Theta + Rotation_Block)
U_v = L_diag * np.sin(Theta + Rotation_Block) - L_diag * np.sin(Theta)
F_rocking = np.zeros(100)

for i in range(100):
    F_rocking[i] = W * (B / 2 - off - U_h[i]) / (H / 2 + U_v[i] - 3 * W / kn)

plt.plot(U_h * 1000, F_rocking, linewidth=.75, color='orange', marker=None, label='Rocking')

plt.plot(U, P, linewidth=.75, color='black', marker='*', label='HybriDFEM')
plt.grid()
plt.legend()
print(np.around(St.P_r, 3))
print(np.around(St.U, 3))

# %%
