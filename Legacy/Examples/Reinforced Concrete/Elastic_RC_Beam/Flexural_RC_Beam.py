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

CPS = 35
BLOCKS = 50

CONCR = mat.NoTension_Mat(E_c, NU_c)
STEEL = mat.Material(E_s, NU_s)

H = .5
B = .2
r = 10e-3
A = 2 * np.pi * r ** 2
# print(A)


St = st.Structure_2D()
St.add_beam(N1, N2, BLOCKS, H, 0, B, material=CONCR)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

for cf in St.list_cfs:
    cf.add_reinforcement([.8], A, material=STEEL, height=r)

# St.plot_structure()

F = 25e3
St.fixNode(0, [0, 1, 2])
St.loadNode(BLOCKS - 1, 1, -F)

St.solve_forcecontrol(2, filename='Flexural', tol=10)
# St.plot_structure(scale=1)
print(St.list_cfs[0].cps[35].sp1.law.tag)

St.plot_stresses(angle=np.pi / 2, save='beam_stress_steel.eps', tag=None)
St.plot_stress_profile(save='stress_prof.eps')
St.plot_structure(scale=20, plot_cf=False, plot_forces=False, plot_supp=False, save='def_shape.eps')
# %% Plotting

alph = E_s / E_c
D = H - .05
rho = A / (B * D)
x = D * alph * rho * (np.sqrt(1 + 2 / (alph * rho)) - 1)
print(x * 100)
M = 3 * F
s_max = M / ((1 - x / (3 * D)) * A * D)
print(s_max / 1e6)
c_max = 2 * M / ((1 - x / (3 * D)) * B * D ** 2 * x / D)
print(c_max / 1e6)
