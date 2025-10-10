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
LIN = True

# MAT = mat.Bilinear_Mat(E, NU, FY)
MAT = mat.Plastic_Stiffness_Deg(E, NU, FY)
MAT.plot_stress_strain()
# %% Building structure

St = st.Structure_2D()

St.add_fe(N1, N2, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_fe(N2, N3, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_fe(N3, N4, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_beam(N4, N5, BLOCKS, H_b, RHO, b=B_b, material=MAT)
St.add_fe(N5, N6, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_fe(N6, N7, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_fe(N7, N8, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)

St.make_nodes()
St.make_cfs(LIN, nb_cps=CPS)

# %% BCs and Forces

St.fixNode(N1, [0, 1])
St.fixNode(N8, [0, 1])

Node = St.get_node_id(N4)

St.loadNode(Node, [0], 1e4)
St.loadNode(Node, [1], -100e3)

Node2 = St.get_node_id(N5)
St.loadNode(Node2, [1], -100e3)

St.plot_structure(scale=0, plot_cf=False, plot_forces=False, plot_supp=False, save=None)
# plt.savefig('Def_start.eps')
# Save modes in undeformed configuration

nb_modes = 10
n_incr = 6
INCR = 25e-3
STEPS = 15

# %% Simulation
w = np.zeros([nb_modes, n_incr + 1])
M_mod = np.zeros([nb_modes, n_incr + 1])

St.save_structure('Coupled/Plastic_Frame')
St.solve_modal(filename='Coupled/Step_0_Modal', save=True)
w[:, 0] = St.eig_vals[:nb_modes]

St.plot_modes(nb_modes, scale=10, save=True, folder='Coupled/Step_0', show=True)

LIST = [0]

for i in range(n_incr):
    with open(f'Coupled/Plastic_Frame.pkl', 'rb') as file:
        St = pickle.load(file)

    # loading
    print(f'Loading step {i + 1}')
    LIST = np.linspace(LIST[-1], INCR * (i + 1), STEPS)
    LIST = LIST.tolist()
    St.solve_dispcontrol(LIST, 0, Node, 0, tol=.1, filename=f'Coupled/Step_{i + 1}_DispControl_L')
    St.save_structure('Coupled/Plastic_Frame')

    # unloading
    print(f'Unloading step {i + 1}')
    LIST = np.linspace(LIST[-1], 0, STEPS)
    LIST = LIST.tolist()
    St.solve_dispcontrol(LIST, 0, Node, 0, tol=.1, filename=f'Coupled/Step_{i + 1}_DispControl_U')
    St.save_structure('Coupled/Plastic_Frame')

    St.solve_modal(filename=f'Coupled/Step_{i + 1}_Modal_C', save=True)
    print(f'Natural frequencies for step {i + 1}: {np.around(St.eig_vals[:nb_modes], 3)}')
    w[:, i + 1] = St.eig_vals[:nb_modes]

    St.plot_modes(nb_modes, scale=10, save=True, folder=f'Coupled/Step_{i + 1}', show=True)
# %% Plot evolution of frequencies

# %% Plot results

P = np.array([])
U = np.array([])
for i in range(1, n_incr + 1):
    results = f'Coupled/Step_{i}_DispControl_L.h5'

    with h5py.File(results, 'r') as hf:
        # Import what you need
        P = np.append(P, hf['P_r_conv'][3 * Node] / 1000)
        U = np.append(U, hf['U_conv'][3 * Node] * 1000)

    results2 = f'Coupled/Step_{i}_DispControl_U.h5'

    with h5py.File(results2, 'r') as hf:
        # Import what you need
        P = np.append(P, hf['P_r_conv'][3 * Node] / 1000)
        U = np.append(U, hf['U_conv'][3 * Node] * 1000)

filename = 'Results_Modal_Deg.h5'

with h5py.File(filename, 'w') as hf:
    hf.create_dataset('U', data=U)
    hf.create_dataset('P', data=P)
    hf.create_dataset('w', data=w)

# %%
import matplotlib.pyplot as plt

plt.figure(None, figsize=(6, 6))
plt.xlabel(r'Top horizontal displacement [mm]')
plt.ylabel(r'$F / F_{max}$ or $ \omega_{i}/ \omega_{i,0}$')
plt.grid()
# plt.xlim([0, 100])
# plt.ylim([0, 1.05])

plt.plot(U, P, color='black', linewidth=.75, label=r'$F-\Delta$')
plt.legend(fontsize=13)

# %%
