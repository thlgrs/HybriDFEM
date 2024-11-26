# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""

import os

import numpy as np

import Material as mat
import Structure as st

save_path = os.path.dirname(os.path.abspath(__file__))

L = 4
N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, L], dtype=float)

H = .2
B = .2

BLOCKS = 31
CPS = 25

E = 30e9
NU = 0.0

# BC = 'FFr'
# BC = 'PP'
# BC = 'FP'
BC = 'FF'

if BC == 'FFr':
    Lf = 2 * L
elif BC == 'PP':
    Lf = L
elif BC == 'FP':
    Lf = .7 * L
elif BC == 'FF':
    Lf = .5 * L

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, 100., b=B, material=mat.Material(E, NU, shear_def=True))
St.make_nodes()
St.make_cfs(False, nb_cps=CPS)

L_F = 0.7 * 4
EULER = (np.pi / Lf) ** 2 * E * H ** 4 / 12
D_LIN = EULER * 4 / (E * H ** 2)

# STEPS = 2000
if BC == 'FF':
    D_MAX = -1
else:
    D_MAX = -2.5

LIST = np.linspace(0, -1.1 * D_LIN, 100)
LIST = np.append(LIST, np.linspace(-1.1 * D_LIN, D_MAX, 100))

F_h = 5000
F = 1000

St.loadNode(int(BLOCKS / 2), [0], F_h, fixed=True)
St.loadNode(N2, [1], -F)

if BC[0] == 'F':
    St.fixNode(N1, [0, 1, 2])
elif BC[0] == 'P':
    St.fixNode(N1, [0, 1])
if BC[1:] == 'F':
    St.fixNode(N2, [0, 2])
elif BC[1:] == 'P':
    St.fixNode(N2, [0])

St.plot_structure(plot_cf=False)

filename = f'Buckling Column_' + BC
St.solve_dispcontrol(LIST.tolist(), 0, St.get_node_id(N2), 1, filename=filename, dir_name=save_path, max_iter=100)

St.plot_structure(plot_cf=False, scale=1)
St.save_structure(filename)
# St.plot_stresses()
