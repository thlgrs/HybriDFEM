# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""

import numpy as np
import pathlib
import sys

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat

N1 = np.array([0, 0], dtype=float)
N2 = np.array([3, 0], dtype=float)

H = .5
B = .2

BLOCKS = 20
CPS = 10

E = 30e9
NU = 0.0

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, 100., b=B, material=mat.Material(E, NU, shear_def=True))
St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

F = -100e3

St.loadNode(N2, [1], F)
St.fixNode(N1, [0, 1, 2])

# St.solve_linear()
St.solve_forcecontrol(10)

St.get_P_r()

print(St.U[-2] * 1000)
print(np.around(St.P_r))

St.plot_structure(plot_cf=False, scale=10, save='linearBeam')

St.plot_stresses(save='linearBeamStress')
