# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:34:18 2024

@author: ibouckaert
"""
# %% Libraries imports
import matplotlib as mpl

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

import matplotlib.pyplot as plt

import pickle
import h5py
import sys
import pathlib
import numpy as np

# Folder to access the HybriDFEM files
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

plt.figure(figsize=(5, 5), dpi=800)

plt.xlim([0, 65])
plt.ylim([0, 30])
plt.title(r'3m$\times$3m-frame with elastic-perfectly plastic material')
plt.xlabel(r'Horizontal displacement of top left corner [mm]')
plt.ylabel(r'Horizontal applied force [kN]')

# %% Import results from simulation

with open(f'Frame_BilinMat_Linear.pkl', 'rb') as file:
    Frame = pickle.load(file)

with open(f'Frame_BilinMat_Linear_Coupled.pkl', 'rb') as file:
    Frame2 = pickle.load(file)

N = np.array([0, 3], dtype=float)

# print(Frame2.list_nodes)

results = f'Frame_BilinMat_Linear.h5'

with h5py.File(results, 'r') as hf:
    # Import what you need
    D = hf['U_conv'][Frame.get_node_id(N) * 3]
    F = hf['P_r_conv'][Frame.get_node_id(N) * 3]

    last_conv1 = hf['Last_conv'][()]
    last_def1 = hf['U_conv'][:, last_conv1]

plt.plot(D * 1000, F / 1000, linewidth=.5, marker='*', markersize=2, label='HybriDFEM')

results = f'Frame_BilinMat_Linear_Coupled.h5'

with h5py.File(results, 'r') as hf:
    # Import what you need
    D = hf['U_conv'][Frame2.get_node_id(N) * 3]
    F = hf['P_r_conv'][Frame2.get_node_id(N) * 3]

    last_conv1 = hf['Last_conv'][()]
    last_def1 = hf['U_conv'][:, last_conv1]

plt.plot(D * 1000, F / 1000, linewidth=1, marker=None, markersize=2, label='Coupled', linestyle='-')

# %% Make the plot(s)

plt.legend()
plt.grid()

# Save figure under given name (ideally .eps)
plt.savefig('Force_disp_Bilinear_Frame.eps')
