# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""
# %% Library imports

import numpy as np
import os
import h5py
import sys
import pathlib
import importlib
from copy import deepcopy
import pickle


def reload_modules():
    importlib.reload(st)
    importlib.reload(mat)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat

reload_modules()

# %% Structure parameters

N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, 1], dtype=float)
N3 = np.array([0, 2], dtype=float)
N4 = np.array([0, 3], dtype=float)
N5 = np.array([3, 3], dtype=float)
N6 = np.array([3, 2], dtype=float)
N7 = np.array([3, 1], dtype=float)
N8 = np.array([3, 0], dtype=float)

B_b = .2
H_b = .2
H_c = .2 * 2 ** (1 / 3)

CPS = 50
BLOCKS = 30

E = 30e9
NU = 0.0
FY = 20e6
ALPHA = .0

RHO = 2000.

MAT = mat.Bilinear_Mat(E, NU, FY, 0.)
# MAT = mat.Plastic_Mat(E, NU, FY)
# MAT.plot_stress_strain()
# %% Building structure

St = st.Structure_2D()

St.add_fe(N1, N2, E, NU, H_c, b=B_b, lin_geom=True, rho=RHO)
St.add_fe(N2, N3, E, NU, H_c, b=B_b, lin_geom=True, rho=RHO)
St.add_fe(N3, N4, E, NU, H_c, b=B_b, lin_geom=True, rho=RHO)
St.add_beam(N4, N5, BLOCKS, H_b, RHO, b=B_b, material=MAT)
St.add_fe(N5, N6, E, NU, H_c, b=B_b, lin_geom=True, rho=RHO)
St.add_fe(N6, N7, E, NU, H_c, b=B_b, lin_geom=True, rho=RHO)
St.add_fe(N7, N8, E, NU, H_c, b=B_b, lin_geom=True, rho=RHO)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

# %% BCs and Forces

St.fixNode(N1, [0, 1])
St.fixNode(N8, [0, 1])

Node = St.get_node_id(N4)

St.loadNode(Node, [0], 1e4)
# St.plot_structure(scale=0, plot_cf=False, plot_forces=False, plot_supp=False, save=None)
# plt.savefig('Def_start.eps')
# Save modes in undeformed configuration

nb_modes = 10
n_incr = 50
INCR = 2e-3
STEPS = 2

# %% Simulation
w = np.zeros([nb_modes, n_incr + 1])
M_mod = np.zeros([nb_modes, n_incr + 1])

St.save_structure('Coupled/Plastic_Frame')
St.solve_modal(filename='Coupled/Step_0_Modal', save=True)
w[:, 0] = St.eig_vals[:nb_modes]

M_line = np.diag(St.M[np.ix_(St.dof_free, St.dof_free)]).copy()

for i in range(len(M_line)):
    if i % 3 == 0:
        M_line[i] *= 1
    else:
        M_line[i] = 0

M_tot = np.sum(M_line)
print(f'Total mass: {M_tot} kg')

for j in range(nb_modes):
    M_j = St.eig_modes[:, j].T @ St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:, j]
    M_j_s = (St.eig_modes[:, j].T @ M_line) ** 2 / M_j
    M_mod[j, 0] = M_j_s / M_tot
    print(f'Modal mass: {M_mod[j, 0]}')

# St.plot_modes(nb_modes, scale=10, save=True, folder='Coupled/Step_0', show=True)

LIST = [0]

for i in range(n_incr):

    with open(f'Coupled/Plastic_Frame.pkl', 'rb') as file:
        St = pickle.load(file)

    LIST = np.linspace(LIST[-1], INCR * (i + 1), STEPS)
    LIST = LIST.tolist()
    St.solve_dispcontrol(LIST, 0, Node, 0, tol=1, filename=f'Coupled/Step_{i + 1}_DispControl_C')
    St.save_structure('Coupled/Plastic_Frame')

    St.solve_modal(filename=f'Coupled/Step_{i + 1}_Modal_C', save=True)
    print(f'Natural frequencies for step {i + 1}: {np.around(St.eig_vals[:nb_modes], 3)}')
    w[:, i + 1] = St.eig_vals[:nb_modes]

    for j in range(nb_modes):
        M_j = St.eig_modes[:, j].T @ St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:, j]
        M_j_s = (St.eig_modes[:, j].T @ M_line) ** 2 / M_j
        M_mod[j, i + 1] = M_j_s / M_tot

    # St.plot_modes(nb_modes, scale=10, save=True, folder=f'Coupled/Step_{i+1}', show=True)
# %% Plot evolution of frequencies

# %% Plot results

P = np.array([])
U = np.array([])
for i in range(1, n_incr + 1):
    results = f'Coupled/Step_{i}_DispControl_C.h5'

    with h5py.File(results, 'r') as hf:
        # Import what you need
        P = np.append(P, hf['P_r_conv'][3 * Node] / 1000)
        U = np.append(U, hf['U_conv'][3 * Node] * 1000)

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 13

plt.figure(None, figsize=(6, 6))
plt.xlabel(r'Top horizontal displacement [mm]')
plt.ylabel(r'$F / F_{max}$ or $ \omega_{i}/ \omega_{i,0}$')
plt.grid()
plt.xlim([0, 100])
plt.ylim([0, 1.05])

U_new = np.append(U[0::2], U[-1])
markers = ['o', 's', 'D', 'v', '^']

for i in range(5):
    plt.plot(U_new, w[i] / w[i, 0], linewidth=0.2, linestyle='-', markerfacecolor='white', color='black', markersize=4,
             label=r'$\omega_' + f'{i + 1}$', marker=markers[i % 5])

k = np.zeros(len(U_new) - 1)
k_init = (P[1] - P[0]) / (U[1] - U[0])
P_new = np.append(P[0::2], P[-1])

for i in range(len(k)):
    k[i] = (P_new[i + 1] - P_new[i]) / (U_new[i + 1] - U_new[i])
    k[i] /= k_init

# plt.plot(U_new[:-1], k, markerfacecolor='white',color='grey',markersize=2,label=r'$dF/d\Delta$')
print(U)
plt.plot(U, P / max(P), color='black', linewidth=.75, label=r'$F-\Delta$')
plt.legend(fontsize=13)

plt.savefig('Freqs_Force_Disp.eps')

# %%
with open(f'Coupled/Plastic_Frame.pkl', 'rb') as file:
    St = pickle.load(file)
St.plot_structure(scale=5, plot_cf=False, plot_forces=False, plot_supp=False, save=None)
plt.savefig('Def_end.eps')

# %% Plot evolution of frequencies
list_w = np.zeros((11, 10))
freq_index = np.arange(1, 11)
list_indexes = [0, 2, 3, 4, 5, 10]

plt.figure(None, figsize=(6, 6))
markers = ['x', '^', 'D', 'v', 's', 'o']
m = 0

for i in range(11):
    if i in list_indexes:
        plt.plot(freq_index, w[:, i * 5] / w[:, 0], label=rf'$\Delta = {i * 10}$~mm', \
                 marker=markers[m % 6], markersize=4, linestyle='-', \
                 linewidth=0.75, markerfacecolor='white', color='black')
        m += 1
plt.xlabel('Frequency number')
plt.ylabel('Normalized frequency $\omega_i / \omega_{i,0}$')
plt.grid('True')
plt.xlim([1, 10])
plt.ylim([0, 1.05])
custom_labels = [r'$\omega_{' + f'{i}' + r'}$' for i in freq_index]
plt.xticks(freq_index, labels=custom_labels)
plt.legend()

plt.savefig('Evolution_frequencies.eps')
# %%
with open(f'Coupled/Plastic_Frame.pkl', 'rb') as file:
    St = pickle.load(file)
St.solve_modal(filename=f'Coupled/Step_50_Modal_C', save=True)
# St.plot_modes(nb_modes, scale=2, save=True, folder=f'Coupled/Step_50', show=True)
# %% Plot frequencies without normalization

plt.figure(None, figsize=(6, 6))
plt.xlabel(r'Top horizontal displacement [mm]')
plt.ylabel(r'$\omega_i$')
plt.grid()
plt.xlim([0, 100])
plt.ylim([0, 1100])

for i in range(5):
    plt.plot(U_new, w[i], linewidth=0.2, linestyle='-', markerfacecolor='white', color='black', markersize=4,
             label=r'$\omega_' + f'{i + 1}$', marker=markers[i % 5])

plt.legend()

plt.savefig('Freqs_Disp.eps')

# %% Plot effective modal mass

plt.figure(None, figsize=(6, 6))
plt.xlabel(r'Top horizontal displacement [mm]')
plt.ylabel(r'$M^*_i / M_{tot}$')
plt.grid()
plt.xlim([0, 100])
plt.ylim([0, 1])

for i in range(5):
    plt.plot(U_new, M_mod[i], linewidth=0.2, linestyle='-', markerfacecolor='white', color='black', markersize=4,
             label=r'$M^*_' + f'{i + 1}$', marker=markers[i % 5])

plt.legend(fontsize=13)

plt.savefig('Modal_mass_vs_disps.eps')

# %% Plot full Pushover

plt.figure(None, figsize=(6, 6))
plt.xlabel(r'Top horizontal displacement $\Delta$ [mm]')
plt.ylabel(r'$F$ [kN]')
plt.grid()
plt.xlim([0, 100])
plt.ylim([0, 30])
# plt.plot(U_new[:-1], k, markerfacecolor='white',color='grey',markersize=2,label=r'$dF/d\Delta$')
plt.plot(U, P, color='black', linewidth=.75, label=None)
# plt.legend(fontsize=16)

plt.savefig('Pushover_Frame.eps')
# %%
