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

H_b = .2
L_b = .4
B = 1

kn = 1e8
ks = 1e8
mu = .65
psi = .2

# %%
RHO = 1000.

Blocks_Bed = 3
Blocks_Head = 3

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
St.make_cfs(False, nb_cps=2, contact=ct.Coulomb(kn, ks, mu, psi=psi), offset=0.02)

# %% BCs and Forces

Total_mass = 0

St.fixNode(0, [0, 1, 2])

# for i in range(1,len(St.list_blocks)):
#     W = St.list_blocks[i].m * 10
#     # St.loadNode(i, 0, .65*W)
#     St.loadNode(i, 1, -W)

# St.solve_forcecontrol(100)

# St.reset_loading()


for i in range(1, len(St.list_blocks)):
    W = St.list_blocks[i].m * 10
    St.loadNode(i, 0, W)
    St.loadNode(i, 1, -W, fixed=True)
W_c = W

St.plot_structure(scale=0, plot_cf=True, plot_forces=True)

# %%
Node = len(St.list_blocks) - 1
# %%
# D0 = St.U[3*Node]
# print(D0)

LIST = np.linspace(0, 3.27e-1, 9000)
# LIST = np.append(LIST, np.linspace(1e-2, 2.7e-1, 1000))
# LIST = np.append(LIST, np.linspace(2e-5, 2e-4, 100))
# LIST = np.append(LIST, np.linspace(2e-4, 2e-3, 100))
# LIST = np.append(LIST, np.linspace(2e-3, 5e-3, 100))
# LIST = np.append(LIST, np.linspace(5e-3, 1e-2, 100))
# LIST = np.append(LIST, np.linspace(1e-2, 5e-2, 200))
# LIST = np.append(LIST, np.linspace(5e-2, 2.7e-1, 2000))
# LIST = np.append(LIST, np.linspace(2e-1, 1e-2, 1000))
# LIST = np.append(LIST, np.linspace(1e-2, 1e-1, 100))
# LIST = np.append(LIST, np.linspace(1e-1, 2e-1, 100))

# LIST = np.append(LIST, np.linspace(1e-4, 1e-3, 100))
# LIST = np.append(LIST, np.linspace(1e-3, 1e-2, 100))

LIST = LIST.tolist()

# St.get_P_r()
# print(St.P_r)
# St.solve_forcecontrol(200,tol=.1)
St.solve_dispcontrol(LIST, 0, Node, 0, tol=1e-1, filename=f'Wallet_DispControl_psi={psi}', max_iter=100)
# St.solve_workcontrol(1000, tol=1e-3, filename=f'Wallet_DispControl',max_iter=100)
St.save_structure('Wallet')

# %% Plot structure
St.plot_structure(scale=1, plot_cf=False, plot_forces=False)
# %%
import matplotlib.pyplot as plt

plt.figure(None, (6, 6))
file = 'Wallet_DispControl_psi=0.65.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U_a = hf['U_conv'][-3] * 1000
    r_a = hf['U_conv'][-1] * 1000
    P_a = hf['P_r_conv'][-3] / W_c

file = 'Wallet_DispControl_Nonassociate.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    last_conv = hf['Last_conv'][()]
    U_na = hf['U_conv'][-3, :last_conv] * 1000
    r_na = hf['U_conv'][-1, :last_conv] * 1000
    P_na = hf['P_r_conv'][-3, :last_conv] / W_c

file = 'Wallet_DispControl_psi=0.325.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U_h = hf['U_conv'][-3] * 1000
    r_h = hf['U_conv'][-1] * 1000
    P_h = hf['P_r_conv'][-3] / W_c

plt.plot(U_a, P_a, linewidth=.75, color='red', label=r'$\psi = \mu$')
plt.plot(U_h, P_h, linewidth=.75, color='blue', label=r'$\psi = \mu/2$')
plt.plot(U_na, P_na, linewidth=.75, color='green', label=r'$\psi = 0$')
# plt.xlim(0,330)
# plt.ylim(0,.65)
plt.grid()

print(f'Load multiplier is {np.around(np.max(P_a), 3)} for associate flow rule')
print(f'Load multiplier is {np.around(np.max(P_na), 3)} for non-associate flow rule')
# %%
