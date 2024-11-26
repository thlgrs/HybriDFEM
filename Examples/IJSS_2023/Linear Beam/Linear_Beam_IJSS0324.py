# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""

import numpy as np

from Objects import Material as mat
from Objects import Structure as st

N1 = np.array([0, 0], dtype=float)
N2 = np.array([3, 0], dtype=float)

H = .5
B = .2

BLOCKS = 25
CPS = 25

E = 30e9
NU = 0.0

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, 100., b=B, material=mat.Material(E, NU, shear_def=False))
St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

F = -100e3

St.loadNode(N2, [1], F)
St.fixNode(N1, [0, 1, 2])

St.solve_linear()

print(St.U[-2] * 1000)

St.plot_structure(plot_cf=False, scale=1, save='linearBeam')

St.plot_stresses(save='linearBeamStress')
