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
    importlib.reload(mat)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat

reload_modules()

N1 = np.array([0, 0], dtype=float)
N2 = np.array([3, 0], dtype=float)

H = .5
B = .2

BLOCKS = 100
CPS = 100

E = 30e9
NU = 0.0

RHO = 7000
A = B * H
I = B * H ** 3 / 12
L = 3

G = E / 2 / (1 + NU)
k = 5 * (1 + NU) / (6 + 5 * NU)
k2 = np.arange(1, 11, 1)
b_star = - G * k / RHO * (A / I + (k2 * np.pi / L) ** 2 * (1 + E / (G * k)))
c_star = E * G * k / (RHO ** 2) * (k2 * np.pi / L) ** 4

w2 = np.sqrt(1 / 2 * (-b_star - np.sqrt(b_star ** 2 - 4 * c_star)))
w1 = np.sqrt(1 / 2 * (-b_star + np.sqrt(b_star ** 2 - 4 * c_star)))
w_t = np.sqrt(G * k * A / (RHO * I))
w_T = np.sort(np.concatenate((w2, w1, np.array([w_t]))))
for i in np.arange(1, 11, 1):
    print(f'TImoshenko-Ehrenfest frequency for mode {i}: {np.around(w_T[i - 1], 3)}')

list_CPs = [5, 10, 25, 50, 100]
markers = ['o', 's', '^', 'D', '*']

# %% Plotting

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 13

plt.figure(1, figsize=(6, 6))

modes = np.arange(1, 11)
plt.plot(modes, np.zeros(10), color='black', linewidth=.5, linestyle='dashed')

plt.grid(True, color='gainsboro', linewidth=.5)
plt.xlabel('Mode number')
plt.ylabel('Relative error [\%]')
plt.xlim([1, 10])
plt.ylim([-5, .5])

for i, CPS in enumerate(list_CPs):

    St = st.Structure_2D()

    St.add_beam(N1, N2, BLOCKS, H, RHO, b=B, material=mat.Material(E, NU, corr_fact=6 / 5))
    St.make_nodes()
    St.make_cfs(True, nb_cps=CPS)

    St.fixNode(N1, [1])
    St.fixNode(N2, [1])

    for j in range(BLOCKS):
        St.fixNode(j, 0)

    St.solve_modal(10)

    error = (St.eig_vals - w_T[:10]) / w_T[:10]

    plt.plot(modes, error * 100, label=f'{CPS} CPs', color='black', \
             marker=markers[i], linewidth=.75, markersize=4)

plt.legend()
plt.xticks(np.arange(1, 11, 1))

plt.savefig('Convergence_CPs.eps')

# St.plot_modes(10, scale=3, save=True)

# w_tilde = np.sqrt(G*k*)


# %%
