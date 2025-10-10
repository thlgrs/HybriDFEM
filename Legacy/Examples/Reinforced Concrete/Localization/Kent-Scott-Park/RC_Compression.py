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

CPS = 1
BLOCKS = 10

Gc = 5e5
e_l = 1 / BLOCKS
gc = Gc / e_l

ft = 2e6
et = 1e-4
etu = 10 * et
b = .01

fc = 1
ec = 1e-10
ecu = 10 * ec
n = 1.01

Gc = 2e5
gc = Gc * BLOCKS

Gt = 1e3
gt = Gt * BLOCKS

d_max = 1e-3
reg = True

if not reg:
    gc = None
    gt = None

file = f'{BLOCKS}Bl_{'C' if d_max < 0 else 'T'}{'_R' if reg else ''}'

CONCR = mat.KSP_concrete(fc, ec, 0.1 * fc, ecu, ft, et, etu, b, gc=None, gt=gt)
# CONCR.plot_stress_strain()

H = .2
B = .2

St = st.Structure_2D()
St.add_beam(N1, N2, BLOCKS, H, 0, B, material=CONCR)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

St.list_cfs[0].cps[0].sp1.A = 0.99 * B * H
St.list_cfs[0].cps[0].sp2.A = 0.99 * B * H

for i in range(1, BLOCKS - 1):
    St.fixNode(i, [1, 2])

F = 1e3
St.fixNode(0, [0, 1, 2])
St.fixNode(BLOCKS - 1, [0, 1, 2])

St.loadNode(BLOCKS - 1, 0, -F)

St.solve_dispcontrol(100, d_max, BLOCKS - 1, 0, filename=file, tol=1e-2, max_iter=100)

St.plot_stresses(tag=None)
# St.plot_strains(tag=None)
# St.plot_stress_profile(save='stress_prof.eps')
St.plot_structure(scale=10, plot_cf=False, plot_forces=False, plot_supp=False)

# %% Plot Pushover
file1 = file + '.h5'
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
