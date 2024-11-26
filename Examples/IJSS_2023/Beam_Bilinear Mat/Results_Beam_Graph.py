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

# with open('Beam_Bilinear_Alpha.pkl', 'rb') as file:  # 'rb' means read in binary mode
#     St = pickle.load(file)
plt.figure(figsize=(5, 5), dpi=800)

plt.xlim([0, 15])
plt.ylim([0, 3])
plt.title(r'Cantilever beam with bilinear material')
plt.xlabel(r'Vertical displacement of beam end [mm]')
plt.ylabel(r'Vertical applied force')

# %% Import results from simulation
Alphas = [0.1, 0.05, 0.0, -0.05, -0.1]

K0 = 2 * 20e6 / (30e9 * .5)
M0 = .2 * .5 ** 2 * 20e6 / 6

for i, a in enumerate(Alphas):

    with open(f'Beam_Bilinear_Alpha={a}.pkl', 'rb') as file:
        St = pickle.load(file)
        nb_blocks = len(St.list_blocks)
        Lb = 3. / nb_blocks

    results = f'Beam_BilinearMat_Alpha={a}.h5'

    with h5py.File(results, 'r') as hf:

        # Import what you need
        K_star = -hf['U_conv'][5] / Lb
        M_star = -(3 - Lb / 2) * hf['P_r_conv'][-2]

        last_conv1 = hf['Last_conv'][()]
        last_def1 = hf['U_conv'][:, last_conv1]

    plt.plot(K_star / K0, M_star / M0, label=rf'$\alpha={a}$', linewidth=.5, marker='*', markersize=2)

    K_ = np.linspace(0, max(K_star / K0), 30)
    M_ = np.zeros(30)

    for i in range(30):
        if K_[i] <= 1:
            M_[i] = K_[i]
        else:
            M_[i] = .5 * (3 - (1 - a) / (K_[i] ** 2)) + a * (K_[i] - 1.5)

    if i == 0:
        label = 'Analytical'
    else:
        label = None
    plt.plot(K_, M_, label=label, linestyle='dotted', color='black')

# %% Make the plot(s)

plt.legend()
plt.grid()

# Save figure under given name (ideally .eps)
plt.savefig('Force_disp_Bilinear.eps')
