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
import Material as mat
import pandas as pd

N1 = np.array([0, 0], dtype=float)
N2 = np.array([1, 0], dtype=float)

E_c = 30e9
NU_c = 0.2
FY = 30e6
ALPHA = -.005

CPS = 1
BLOCKS = 5

Gc = 5e5
e_l = 1 / BLOCKS
gc = Gc / e_l

CONCR = mat.Bilinear_Mat(E_c, 0.2, FY, ALPHA)
CONCR = mat.concrete_EC_softening(E_c, FY, alpha=ALPHA, gt=gc)
CONCR.plot_stress_strain()

H = .2
B = .2

St = st.Structure_2D()
St.add_beam(N1, N2, BLOCKS, H, 0, B, material=CONCR)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

St.list_cfs[0].cps[0].sp1.A = 0.9 * B * H
St.list_cfs[0].cps[0].sp2.A = 0.9 * B * H

for i in range(1, BLOCKS - 1):
    St.fixNode(i, [1, 2])

F = 1e3
St.fixNode(0, [0, 1, 2])
St.fixNode(BLOCKS - 1, [0, 1, 2])

St.loadNode(BLOCKS - 1, 0, -F)

# St.solve_dispcontrol(1000, -5e-3, BLOCKS-1, 0, filename=f'Results_{BLOCKS}_R',tol=1e-2)


St.plot_stresses(tag=None)
St.plot_strains(tag=None)
# St.plot_stress_profile(save='stress_prof.eps')
St.plot_structure(scale=10, plot_cf=False, plot_forces=False, plot_supp=False)

# %% Plot Pushover
file1 = f'Results_{BLOCKS}_R.h5'
import matplotlib as mpl

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

with h5py.File(file1, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    P_c = -hf['P_r_conv'][3 * BLOCKS - 3, :last_conv] / 1000
    U_c = -hf['U_conv'][3 * BLOCKS - 3, :last_conv] * 1000

# file2 = 'Elastic_DispControl.h5'

# with h5py.File(file2, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     P_e = hf['P_r_conv'][3*Node,:last_conv] / W
#     U_e = hf['U_conv'][3*Node,:last_conv]*1000
import matplotlib.pyplot as plt

print(max(P_c))
plt.figure(figsize=(4.5, 4.5), dpi=600)
plt.grid(True)
plt.xlabel(r'Control displacement [mm]')
plt.ylabel(r'Load multiplier [-]')
plt.plot(U_c, P_c, color='black')
# plt.plot(U_e,P_e,label='Elastic')
# plt.legend()
# plt.xlim((0,40))
# plt.ylim((0, 0.08))

# plt.savefig('Dispcontrol_arch.eps')
