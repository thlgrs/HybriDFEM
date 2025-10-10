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

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as ct
import Surface as surf
import ContactPair as cp

N1 = np.array([0, 0])

H = 2
B = 2

kn = 2e2
ks = 2e2

# %%
RHO = 1 / (H * B * 9.81)

PATTERN = [[1], [1]]

St = st.Structure_2D()

# St.add_block(vertices, RHO, b=1)
St.add_wall(N1, B, H, PATTERN, RHO, b=1., material=None)

St.make_nodes()

mu = .5

St.make_cfs(False, nb_cps=2, contact=ct.Coulomb(kn, ks, mu, psi=1), offset=0.)

# %% BCs and Forces

Total_mass = 0

St.fixNode(0, [0, 1, 2])
# St.fixNode(1,[2])
# St.fixNode(2,[2])

W = 1.
# %%


# St.loadNode(1, 0, W)
# St.loadNode(2, 0, W)
St.loadNode(1, 0, W)
St.loadNode(1, 1, -1 * W, fixed=True)
# St.loadNode(2, 1, -W, fixed=True)

St.plot_structure()

# %% Simulation
Node = 1
d_end = 2e-2

LIST = np.array([])
# %%
# LIST = np.linspace(0,.5,1000)
LIST = np.append(LIST, np.linspace(0, d_end, 50))
LIST = np.append(LIST, np.linspace(d_end, 0, 50))
LIST = np.append(LIST, np.linspace(0, d_end, 50))
LIST = np.append(LIST, np.linspace(d_end, 0, 50))
LIST = np.append(LIST, np.linspace(0, d_end, 50))
LIST = np.append(LIST, np.linspace(d_end, 0, 50))
LIST = np.append(LIST, np.linspace(0, d_end, 50))
LIST = np.append(LIST, np.linspace(d_end, 0, 50))
# LIST = np.append(LIST, np.linspace(0,2*d_end,100))
# LIST = np.append(LIST, np.linspace(2*d_end, 0,100))
# LIST = np.append(LIST, np.linspace(0,3*d_end,150))
# LIST = np.append(LIST, np.linspace(3*d_end, 0,150))
# LIST = np.append(LIST, np.linspace(0,4*d_end,200))
# LIST = np.append(LIST, np.linspace(4*d_end, 0,200))
# LIST = np.append(LIST, np.linspace(-d_end, d_end,20))
# LIST = np.append(LIST, np.linspace(d_end, -d_end,20))
# LIST = np.append(LIST, np.linspace(1e-10,1e-6, 100))
# LIST = np.append(LIST, np.linspace(1e-6, 1e-4,1000))
# LIST = np.append(LIST, np.linspace(2*d_end, -2*d_end, 40))

# LIST = np.append(LIST, np.linspace(-d_end, 2*d_end, 60))
# LIST = np.append(LIST, np.linspace(2*d_end, 0, 80))
LIST = LIST.tolist()

St.solve_dispcontrol(LIST, 0, Node, 0, tol=1e-5, filename=f'Coulomb_Lin_Cont', max_iter=100)
# St.solve_workcontrol(36, tol=1e-4, filename='NoTension_Lin_Cont', max_iter=100)
# St.solve_forcecontrol(10, tol=1e-4)
print(St.U[-3:] * 1000)
St.save_structure('Wallet')

# %% Plot structure
St.plot_structure(scale=10, plot_cf=True, plot_forces=False)
# %% Plot Force-displacement curve
import matplotlib.pyplot as plt

file = 'Coulomb_Lin_Cont.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U = hf['U_conv'][-3] * 1000
    U_v = hf['U_conv'][-2] * 1000
    P = hf['P_r_conv'][-3]

plt.figure(None, figsize=(6, 6))
# plt.xlim(0,d_end*1000)
# plt.ylim(0,1)

D = np.linspace(0, d_end, 100)

k_s = ks / 2
k_n = kn / 2

K_twocontacts = np.array([[2 * k_s, 0, H * k_s],
                          [0, 2 * k_n, 0],
                          [H * k_s, 0, H ** 2 * k_s / 2 + k_n * (B) ** 2 / 2]])

K_sliding = np.array([[0, -2 * mu * k_n, H * k_s / 2],
                      [0, 2 * k_n, 0],
                      [0, -2 * mu * k_n / 10, H ** 2 * k_s / 4 + k_n * (B) ** 2 / 2]])

K_onecontact = np.array([[k_s, -mu * k_n, (k_s + mu * k_n) / 10],
                         [0, 2 * k_n, 0],
                         [k_s / 10, -mu * k_n / 10, ((2 + mu) * k_n + k_s) / 100]])

print(K_onecontact / 1e7)

print(K_sliding / 1e7)

F_twocontacts = np.linspace(0, 1.25, 100)
D_twocontacts = np.zeros(100)
D_onecontact = np.zeros(100)

for i in range(100):
    F = np.array([F_twocontacts[i], W, 0])
    D_twocontacts[i] = np.linalg.solve(K_twocontacts, F)[0]
    D_onecontact[i] = np.linalg.solve(K_onecontact, F)[0]

plt.plot(D_twocontacts * 1000, F_twocontacts, linewidth=.75, color='red', marker=None, label='Bending- k_s')
plt.plot(D_onecontact * 1000, F_twocontacts + .29, linewidth=.75, color='blue', marker=None, label='Bending - k_s/2')

Loss_LeftContact = 2 * W * (B / 2) / H

plt.plot(D * 1000, Loss_LeftContact * np.ones(100), linewidth=.75, color='orange', marker=None, label='Rocking')
plt.plot(D * 1000, mu * np.ones(100), linewidth=.75, color='green', marker=None, label='sliding')

plt.plot(U, P, linewidth=.75, color='black', marker='o', label='HybriDFEM')

plt.grid()
plt.legend()

# %%
plt.figure(None, figsize=(6, 6))

plt.plot(U, U_v, linewidth=.5, color='grey', marker='.', label='HybriDFEM')
plt.grid(True)
