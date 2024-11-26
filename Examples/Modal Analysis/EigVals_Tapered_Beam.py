# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""

import pathlib
import sys

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import numpy as np
import Structure as st
import Material as mat

N1 = np.array([0, 0], dtype=float)
N2 = np.array([5, 0], dtype=float)

C = 4
H1 = .4
H2 = H1 * (1 + C)
B = .2

BLOCKS = 100
CPS = 100

E = 210e9
NU = 0.3
RHO = 7850

St = st.Structure_2D()

St.add_tapered_beam(N1, N2, BLOCKS, H1, H2, RHO, b=B, material=mat.Material(E, NU, corr_fact=13 / 15, shear_def=False))
St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

St.plot_structure(plot_cf=False)
# St.fixNode(N1, [0,1,2])
St.fixNode(N2, [0, 1, 2])

St.solve_modal(no_inertia=True)

St.plot_modes(5, scale=5)

lbda = np.sqrt(RHO * H2 * B * 5 ** 4 / (E * B * H2 ** 3 / 12))

print(St.eig_vals[:5] * lbda)
