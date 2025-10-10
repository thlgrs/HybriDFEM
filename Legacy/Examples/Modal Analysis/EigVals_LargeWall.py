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

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Surface as surf

N1 = np.array([0, 0])

L_wall = 6.24
H_wall = 5.2

Blocks_Bed = 6 * 4
Blocks_Head = 20 * 4

H = H_wall / Blocks_Head
L = L_wall / Blocks_Bed
B = .12

CPS = 6

E = .2e9
G = .01e9
t = .01 / 2

kn = E / t
ks = G / t

# %%
RHO = 1500.

Line1 = []
Line2 = []

for i in range(Blocks_Bed):
    Line1.append(1.)
    if i == 0:
        Line2.append(.5)
        Line2.append(1.)
    elif i == Blocks_Bed - 1:
        Line2.append(.5)
    else:
        Line2.append(1.)

PATTERN = []

for i in range(Blocks_Head):
    if i % 2 == 0:
        PATTERN.append(Line1)
    else:
        PATTERN.append(Line2)

St = st.Structure_2D()

St.add_wall(N1, L, H, PATTERN, RHO, b=B, material=None)
St.make_nodes()
St.make_cfs(True, nb_cps=CPS, surface=surf.Surface(kn, ks))

# %% Solving the eigenvalue problem
St.solve_modal(11)

# %% Plotting results
St.plot_modes(11, scale=10)

print(np.around(St.eig_vals, 3))
