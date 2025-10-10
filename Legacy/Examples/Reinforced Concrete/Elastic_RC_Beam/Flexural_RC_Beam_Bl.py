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
N2 = np.array([3, 0], dtype=float)

E_c = 30e9
E_s = 200e9
NU_c = 0.2
NU_s = 0.3
# RHO = 3000

BLOCKS = np.array(
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 150, 200])
# BLOCKS = np.array([2])
sig_s = np.zeros(len(BLOCKS))
sig_c = np.zeros(len(BLOCKS))

CPS = 100

CONCR = mat.NoTension_Mat(E_c, NU_c)
STEEL = mat.Material(E_s, NU_s)

H = .5
B = .2
r = 10e-3
A = 2 * np.pi * r ** 2
# print(A)

for i, bl in enumerate(BLOCKS):
    St = st.Structure_2D()
    St.add_beam(N1, N2, bl, H, 0, B, material=CONCR)

    St.make_nodes()
    St.make_cfs(True, nb_cps=CPS)

    for cf in St.list_cfs:
        cf.add_reinforcement([.8], A, material=STEEL, height=r)

    F = 25e3
    print(bl)
    St.fixNode(0, [0, 1, 2])
    print(bl - 1)
    St.loadNode(int(bl - 1), 1, -F)

    St.plot_structure(plot_forces=True, plot_cf=True)

    St.solve_forcecontrol(2, filename='Flexural', tol=10)
    St.plot_structure(scale=100)

    sigma, eps, x_s = St.get_stresses(angle=np.pi / 2)
    # print(sigma)
    sig_s[i] = np.max(sigma)
    sig_c[i] = np.min(sigma)

# print(sig_s)
# print(sig_c)  

# Create DataFrame with each array as a column
df = pd.DataFrame({'BLOCKS': BLOCKS, 'SIG_S': sig_s, 'SIG_C': sig_c})

# Save to Excel
df.to_excel('conv_blocks.xlsx', index=False, engine='openpyxl')

# %% Plotting

import matplotlib.pyplot as plt

file = 'Flexural.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    R = hf['U_conv'][-2] * 1000
    M = hf['P_r_conv'][-2]

plt.figure(None, figsize=(6, 6))
# plt.xlim(0,d_end*1000)
# plt.ylim(0,1)

plt.plot(R, M)

alph = E_s / E_c
D = H - .05
rho = A / B * D
x = alph * rho * (np.sqrt(1 + 2 / (alph * rho)) - 1)
print(x * 100)
M = 3 * F
s_max = M / ((1 - x / (3 * D)) * A * D)
print(s_max / 1e6)
c_max = 2 * M / ((1 - x / (3 * D)) * B * D ** 2 * x / D)
print(c_max / 1e6)

St.plot_structure(scale=10, plot_cf=True)
