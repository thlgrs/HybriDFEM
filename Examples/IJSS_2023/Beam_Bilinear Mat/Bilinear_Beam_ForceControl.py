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

N1 = np.array([0, 0], dtype=float)
N2 = np.array([3, 0], dtype=float)

H = .5
B = .2

BLOCKS = 40
CPS = 40

E = 30e9
NU = 0.0
FY = 20e6
A = .001

filename = f'Beam_BilinearMat_Alpha={A}'

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, 100., b=B, material=mat.Bilin_Mat(E, NU, FY, A))
St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

F = -100e3

St.loadNode(N2, [1], F)
St.fixNode(N1, [0,1,2])
# St.fixNode(N2, [0])

St.solve_forcecontrol(4, dir_name=save_path, filename=filename)

St.save_structure(f'Beam_Bilinear_Alpha={A}')
