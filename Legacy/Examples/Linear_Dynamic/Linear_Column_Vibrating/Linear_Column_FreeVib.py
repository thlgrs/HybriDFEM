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
import Material as mat

save_path = os.path.dirname(os.path.abspath(__file__))

# Meth = ['WBZ', .1, .6, .3025]
# Meth = 'HHT'
Meth = ['GEN', 0.5]

N1 = np.array([0, 0], dtype=float)
N2 = np.array([4, 0], dtype=float)

H = .1
B = .1

BLOCKS = 6
CPS = 25

E = 4 * np.pi ** 2
NU = 0.0

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, 2, b=B, material=mat.Material(E, NU, shear_def=True))
St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

F = -E * 1e-3

St.loadNode(N2, [1], F, fixed=True)
St.fixNode(N1, [0, 1, 2])
# St.fixNode(N2, [1])

St.solve_linear()

St.plot_structure(scale=1, plot_cf=False)

U0 = St.U.copy()

St.set_damping_properties(xsi=0.0, damp_type='STIFF')

St.solve_dyn_linear(5, 1e-3, U0=U0, Meth=Meth)
