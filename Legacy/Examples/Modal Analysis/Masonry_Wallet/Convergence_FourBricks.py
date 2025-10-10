# -*- coding: utf-8 -*-
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
import matplotlib as mpl

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'


def reload_modules():
    importlib.reload(st)
    importlib.reload(surf)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Surface as surf

reload_modules()

N1 = np.array([0, 0])

blocks_h = 3  # head
blocks_b = 1  # bed
t = .01
H = 0.055
L = .25
B = .12

CPS = 10

E = 1e9
NU = 0.2
G = E / (2 * (1 + NU))
K = E / ((1 + NU) * (1 - 2 * NU))

kn = E / (t / 2)
ks = G / (t / 2)

# %%
RHO = 1800

List_CPs = [2, 5, 10, 50, 100]

import matplotlib.pyplot as plt

plt.figure(1, figsize=(5, 5), dpi=1200)

markers = ['o', 's', '^', 'D', '*']
modes = np.arange(4, 13)
plt.plot(modes, np.zeros(9), color='black', linewidth=.5, linestyle='dashed')

for i, CPS in enumerate(List_CPs):
    St = st.Structure_2D()

    vertices = np.array([[L, 0.], [L, H], [0., H], [0.0, 0.0]])
    St.add_block(vertices, RHO, b=B)

    vertices = np.array([[L / 2, H], [L / 2, 2 * H], [-L / 2, 2 * H], [-L / 2, H]])
    St.add_block(vertices, RHO, b=B, ref_point=np.array([0, 3 * H / 2]))

    vertices = np.array([[3 * L / 2, H], [3 * L / 2, 2 * H], [L / 2, 2 * H], [L / 2, H]])
    St.add_block(vertices, RHO, b=B, ref_point=np.array([L, 3 * H / 2]))

    vertices = np.array([[L, 2 * H], [L, 3 * H], [0.0, 3 * H], [0.0, 2 * H]])
    St.add_block(vertices, RHO, b=B)

    St.make_nodes()
    St.make_cfs(True, nb_cps=CPS, surface=surf.Surface(kn, ks))

    St.solve_modal()

    Expected_DE = np.array([2496, 3430, 4184, 4325, 4732, 5163, 5732, 8755, 9172]) * np.pi * 2
    Error_DE = np.around((St.eig_vals[3:] - Expected_DE) / Expected_DE * 100, 2)

    plt.plot(modes, Error_DE, label=f'{CPS} CPs', color='dimgrey', \
             marker=markers[i], linewidth=.75, markersize=3)
    plt.grid(True, color='gainsboro', linewidth=.5)
    plt.xlabel('Mode number')
    plt.ylabel('Relative error [\%]')
    plt.legend()
    plt.xlim([4, 12])
    plt.ylim([-10, .5])

plt.savefig('Convergence_FourBricks.eps')
