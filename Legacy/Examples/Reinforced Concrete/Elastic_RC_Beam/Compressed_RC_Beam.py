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

N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, 3], dtype=float)

E_c = 35e9
E_s = 200e9
NU = 0.
# RHO = 3000

CPS = 10
BLOCKS = 2

CONCR = mat.Material(E_c, NU)
STEEL = mat.Material(E_s, NU)

H = .4
B = np.pi * H / 4
d = 16e-3
A = 8 * np.pi * d ** 2 / 4

St = st.Structure_2D()
St.add_beam(N1, N2, BLOCKS, H, 2500, B, material=CONCR)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

for cf in St.list_cfs:
    cf.add_reinforcement([-.9, .9], A / 2, material=STEEL, height=d)

St.plot_structure()

N = 2000e3
St.fixNode(0, [0, 1, 2])
St.fixNode(BLOCKS - 1, 2)
St.loadNode(BLOCKS - 1, 1, -N)

St.solve_forcecontrol(1)

print(St.U[-2] * 1000)
St.plot_structure(scale=1)
print(St.K[3, 3] / 1e6)
St.plot_stresses()

print(f'Max stress should be {(N / A) / 1e6} MPa')
