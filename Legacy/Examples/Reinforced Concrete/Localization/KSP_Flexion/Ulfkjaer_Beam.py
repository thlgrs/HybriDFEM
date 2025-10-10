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

# From Tanaka and Park 1990

N1 = np.array([0, 0], dtype=float)
N2 = np.array([7.2, 0], dtype=float)

CPS = 25
BLOCKS = 100

ft = 0
et = 1e-4
etu = 0
b = .1

fc = 22.75e6
ec = 2e-3
ecu = 10 * ec

Gc = 180e3
l_elem = 1.65 / BLOCKS
gc = Gc / l_elem

gt = None
# gc=None

d_max = -80e-3
reg = True

if not reg:
    gc = None
    gt = None

file = f'{BLOCKS}Bl_{'_R' if reg else ''}'

CONCR = mat.KSP_concrete(fc, ec, .2 * fc, ecu, ft, et, etu, b, gc=gc, gt=None)
CONCR.plot_stress_strain()

Es = 222e9
NU = 0.2
fy = 650e6
STEEL = mat.steel_EC(Es, fy, alpha=1e-2)

H = .4
B = .2
# P = 0 

St = st.Structure_2D()
St.add_beam(N1, N2, BLOCKS, H, 0, B, material=CONCR)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

r = 10e-3
A = np.pi * r ** 2
# print(A)
for cf in St.list_cfs:
    cf.add_reinforcement([-.8], 3 * A, material=STEEL, height=r)
    cf.add_reinforcement([-.6], 3 * A, material=STEEL, height=r)
    cf.add_reinforcement([-.4], 3 * A, material=STEEL, height=r)
    # cf.add_reinforcement([.8], 2*A, material=STEEL, height=r)

# for i in range(1, BLOCKS-1): 
#     St.fixNode(i, [1,2])

F = 1e3
St.fixNode(0, [0, 1])
St.fixNode(BLOCKS - 1, [0, 1])
# St.fixNode(BLOCKS-1, [0,1,2])

Node = int(3 * BLOCKS / 7.2)
St.loadNode(int(3 * BLOCKS / 7.2), 1, F)
St.loadNode(int(4.2 * BLOCKS / 7.2), 1, F)

# Node = int(BLOCKS/2)-1
St.solve_dispcontrol(200, d_max, Node, 1, filename=file, tol=10, max_iter=100)

St.plot_stresses(tag=None)
# St.plot_strains(tag=None)
# St.plot_stress_profile(save='stress_prof.eps')
St.plot_structure(scale=1, plot_cf=False, plot_forces=True, plot_supp=True)

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
    P_c = -hf['P_r_conv'][Node * 3 + 1, :last_conv] / 1000
    U_c = -hf['U_conv'][Node * 3 + 1, :last_conv] * 1000

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
