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
mu = .65

# %%
RHO = 1 / (H * B * 9.81)

PATTERN = [[1], [1]]

St = st.Structure_2D()
St.add_wall(N1, B, H, PATTERN, RHO, b=1., material=None)

off = 0.00

St.make_nodes()
St.make_cfs(False, nb_cps=2, contact=ct.Coulomb(kn, ks, mu), offset=off)

# %% BCs and Forces

Total_mass = 0

St.fixNode(0, [0, 1, 2])
# St.fixNode(1,[0,2])

W = 1.
St.loadNode(1, 0, 1 * W)
St.loadNode(1, 1, -W, fixed=True)

St.plot_structure(scale=0, plot_cf=True, plot_forces=True)

# %%
Node = 1

LIST = np.array([])
LIST = np.append(LIST, np.linspace(0, .12, 100))

LIST = LIST.tolist()

# St.get_P_r()

St.solve_dispcontrol(LIST, 0, Node, 0, tol=1e-5, filename=f'Wallet_DispControl', max_iter=100)
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
    P = hf['P_r_conv'][-3]

plt.figure(None, figsize=(6, 6))
plt.xlim(0, 100)
plt.ylim(0, 1)

D = np.linspace(0, 2e-3, 100)

k_s = ks / 2
k_n = kn / 2

K_twocontacts = np.array([[2 * k_s, 0, H * k_s],
                          [0, 2 * k_n, 0],
                          [H * k_s, 0, H ** 2 * k_s / 2 + k_n * (B - 2 * off) ** 2 / 2]])

K_onecontact = np.array([[k_s, 0, H * k_s / 2],
                         [0, 2 * k_n, 0],
                         [H * k_s / 2 + + mu * k_n * (B / 2 - off), 0,
                          (H ** 2) * k_s / 4 + 2 * k_n * (B / 2 - off) ** 2 + mu * k_n * (B / 2 - off) * (H / 2)]])

F_bending = np.linspace(0, 1, 100)
D_bending = np.linspace(0, 2e-3, 100)

LeftContact_Sliding = (mu * W / 2) / ((1 / 2) + mu * H / (4 * (B / 2 - off)))

for i in range(100):

    F = np.array([F_bending[i], -W, 0])
    D_bending[i] = np.linalg.solve(K_twocontacts, F)[0]

    if F_bending[i] >= LeftContact_Sliding:
        F = np.array([LeftContact_Sliding, -W, 0])
        D_bending[i] = np.linalg.solve(K_twocontacts, F)[0]
        dF = F_bending[i] - LeftContact_Sliding
        F = np.array([dF, -W, 0])
        D_bending[i] += np.linalg.solve(K_onecontact, F)[0]

plt.plot(D_bending * 1000, F_bending, linewidth=.75, color='red', marker=None, label='Bending')

F = np.array([LeftContact_Sliding, -W, 0])
D_leftcontactsliding = np.linalg.solve(K_twocontacts, F)[0]

plt.plot(D_leftcontactsliding * 1000, LeftContact_Sliding, linewidth=0, color='red', marker='*',
         label='Left contact starts sliding')

Theta = np.arctan2(H / 2, B / 2 - off)
D_theta_max = np.pi / 2 - Theta

Rotation_Block = np.linspace(0, D_theta_max, 100)
L_diag = np.sqrt((H / 2) ** 2 + (B / 2 - off) ** 2)
U_h = L_diag * np.cos(Theta) - L_diag * np.cos(Theta + Rotation_Block)
U_v = L_diag * np.sin(Theta + Rotation_Block) - L_diag * np.sin(Theta)
F_rocking = np.zeros(100)

for i in range(100):
    F_rocking[i] = W * (B / 2 - off - U_h[i]) / (H / 2 + U_v[i])

plt.plot(U_h * 1000, F_rocking, linewidth=.75, color='green', marker=None, label='Rocking')

Sliding_Starts = mu * W

plt.plot(D * 1000, Sliding_Starts * np.ones(100), linewidth=.75, color='orange', marker=None, label='Sliding')

plt.plot(U, P, linewidth=.75, color='black', marker=None, label='HybriDFEM')
plt.grid()
plt.legend()
print(np.around(St.P_r, 3))
print(np.around(St.U, 3))

# %%
