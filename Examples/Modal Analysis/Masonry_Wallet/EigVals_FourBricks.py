# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""

import importlib
import pathlib
import sys

import numpy as np


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

St = st.Structure_2D()

vertices = np.array([[L, 0.], [L, H], [0., H], [0.0, 0.0]])
St.add_block(vertices, RHO, b=B)

vertices = np.array([[L / 2, H], [L / 2, 2 * H], [0, 2 * H], [0, H]])
St.add_block(vertices, 2 * RHO, b=B, ref_point=np.array([0, 3 * H / 2]))

vertices = np.array([[L, H], [L, 2 * H], [L / 2, 2 * H], [L / 2, H]])
St.add_block(vertices, 2 * RHO, b=B, ref_point=np.array([L, 3 * H / 2]))

vertices = np.array([[L, 2 * H], [L, 3 * H], [0.0, 3 * H], [0.0, 2 * H]])
St.add_block(vertices, RHO, b=B)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS, surface=surf.Surface(kn, ks))

# for cf in St.list_cfs: 
#     if abs(cf.angle) < 1e-6: 
#         # print('Hello')
#         for cp in cf.cps: 

#             cp.sp1.law = surf.Surface(2*kn, 2*ks)
#             cp.sp2.law = surf.Surface(2*kn, 2*ks)

# %%

# St.plot_structure(plot_cf=False)
# St.fixNode(N1, [0,1,2])
# for i in range(4): 
#     St.fixNode(i, [2])

St.solve_modal()

St.plot_modes(12, scale=.1, save=True)

print(np.around(St.eig_vals / (2 * np.pi), 2))

# %% Analytical model

Expected_DE = np.array([2496, 3430, 4184, 4325, 4732, 5163, 5732, 8755, 9172]) * np.pi * 2
Expected_FE = np.array([2514, 3345, 4043, 4118, 4697, 4976, 5401, 8265, 8549]) * np.pi * 2

Error_DE = np.around((St.eig_vals[3:] - Expected_DE) / Expected_DE * 100, 2)
Error_FE = np.around((St.eig_vals[3:] - Expected_FE) / Expected_FE * 100, 2)

print(Error_DE)
print(Error_FE)
