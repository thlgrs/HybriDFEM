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


N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, 4], dtype=float)
N3 = np.array([8, 4], dtype=float)
N4 = np.array([8, 0], dtype=float)
C = np.array([4, 4],dtype=float)

H = .5
B = 1

BLOCKS = 30
CPS = 50

E = 30e9
NU = 0.3
RHO = 2500

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, RHO, b=B, material=mat.Material(E, NU), end_2=False)
St.add_arch(C, 0, np.pi, 4, 3*BLOCKS, H, RHO, b=B, material=mat.Material(E, NU))
St.add_beam(N3, N4, BLOCKS, H, RHO, b=B, material=mat.Material(E, NU), end_1=False)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

St.fixNode(N1, [0,1,2])
St.fixNode(N4, [0,1,2])

St.plot_structure(plot_cf=False)

St.solve_modal(5)

St.plot_modes(5, scale=50)

