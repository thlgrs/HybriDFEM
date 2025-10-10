# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:32:01 2025

@author: ibouckaert
"""

import numpy as np
import os
import h5py
import sys
import pathlib

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat
import Surface as surf
import Contact as ct
import ContactPair as cp

save_path = os.getcwd()

# A = (110e-3) * (950e-3) #m^2
phi_r = 35 * np.pi / 180  # rad

E_m = 43e6  # Pa
nu = 0.0
c = 2e6

BLOCKS = 18

h_b = 1.5 / BLOCKS  # m
t_w = 110e-3  # m
r_b = 0.001 * t_w

kn = E_m / h_b * ((t_w) / (t_w - 2 * r_b))  # Pa/m
ks = kn / (2 * (1 + nu))  # Pa/m

# print(kn)
h_b2 = 38e-3
kn2 = E_m / h_b2 * ((t_w) / (t_w - 2 * r_b))  # Pa/m
ks2 = kn2 / (2 * (1 + nu))  # Pa/m

mu = np.tan(phi_r)
# mu=.75
# mu=100

St = st.Structure_2D()

St.add_geometry(filepath=r"Griffith_wall_text_file_12.txt", rho=1800, material=None, gravity=False)
N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, 1.5], dtype=float)

# St.add_beam(N1,N2,BLOCKS+2,t_w,1800,b=1,end_1=True, end_2=True)

# St.plot_structure()
cps = np.linspace(-.98, .98, 5)
cps = cps.tolist()
St.make_nodes()
# St.make_cfs(lin_geom=False, nb_cps=cps, offset=-1, surface=surf.Coulomb(2*kn,2*ks,mu,c=c,psi=0.0))

St.make_cfs(lin_geom=False, nb_cps=cps, offset=-1, surface=surf.NoTension_EP(2 * kn, 2 * ks))

# St.list_cfs[-1].change_cps(nb_cp=cps, offset=-1, surface=surf.Coulomb(2*kn2,2*ks2,mu,c=c))
# St.list_cfs[-2].change_cps(nb_cp=cps, offset=-1, surface=surf.Coulomb(2*kn2,2*ks2,mu,c=c))
St.plot_structure()
St.get_M_str()

for i in range(BLOCKS + 2):
    M = St.list_blocks[i].m
    W = 10 * M
    St.loadNode(i, [1], -W, fixed=True)

M_tot = sum(blk.m for blk in St.list_blocks)
W_tot = 10 * M_tot

St.fixNode(0, [0, 1, 2])
St.fixNode(BLOCKS + 1, [0, 2])

Node = int(np.floor((BLOCKS + 2) / 2))
# print(Node)
Node = 9
St.loadNode(Node, 0, W)
# St.loadNode(Node, 2, -W*86e-3,fixed=True)
# St.fixNode(Node, 0)
# St.loadNode(BLOCKS+1,1,-W_tot,fixed=True)

# St.loadNode(len(St.list_blocks)-1, 1, -W_tot, fixed=True)

St.plot_structure(scale=0, plot_cf=False, plot_forces=True)

# St.solve_modal(1)
# St.plot_modes(1)
# %%

LIST = np.linspace(0, 0.5 * t_w, 1000)
# LIST = np.append(LIST, np.linspace(LIST[-1], 0.5*t_w, 100))
St.solve_dispcontrol(LIST.tolist(), 0, Node, 0, tol=1e-3, dir_name=save_path, filename='results_Griffith_wall_12',
                     max_iter=100)
# St.solve_forcecontrol(100,tol=10)
St.plot_structure(scale=1, plot_cf=False, plot_forces=False, plot_supp=False,
                  save='Griffith.eps')  # structure in the deformed configuration

# %% Plotting

import matplotlib.pyplot as plt
from scipy import interpolate

# Lettura dati da simulazione
with h5py.File("results_Griffith_wall_12.h5", "r") as hf:
    U = hf["U_conv"][Node * 3 + 0]
    P = hf["P_r_conv"][Node * 3 + 0]

d_tw = U / t_w
F_W = P / W_tot

# Plot curva simulazione
plt.plot(d_tw, F_W, label='HybriDEFM Coulomb', color='tab:blue')

labels = ['rigid-body analysis', 'experimental', 'discrete elements']
colors = ['tab:green', 'tab:orange', 'tab:red']

for i in range(1, 4):
    data = np.loadtxt(f'curve_{i}_12.csv', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, label=labels[i - 1], color=colors[i - 1], linestyle='--')

# Grafico finale
plt.title("Pushover curve")
plt.xlabel("d/t_w [-]")
plt.ylabel("F/W [-]")
plt.ylim(0, 0.15)
plt.xlim(0, .5)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.savefig('response.eps')
plt.show()
