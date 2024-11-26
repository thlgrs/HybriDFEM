# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""

import os
import pathlib
import sys

import numpy as np

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat

save_path = os.path.dirname(os.path.abspath(__file__))

N1 = np.array([0, 0], dtype=float)
N2 = np.array([3, 0], dtype=float)

H = .5
B = .2

BLOCKS = 2
CPS = 2

E = 30e9
NU = 0.0
FY = 20e6
A = 0.1

K = np.linspace(0, 15, 30, dtype=float)
Rot = -K * (3 / (BLOCKS - 1)) * 2 * FY / (E * H)

filename = f'Beam_BilinearMat_Alpha={A}'

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, 100., b=B, material=mat.Bilin_Mat(E, NU, FY, A))
St.make_nodes()
St.make_cfs(False, nb_cps=CPS)

F = -100e3

St.loadNode(N2, [1], F)

St.fixNode(N1, [0, 1, 2])
# St.fixNode(N2, [0])

St.solve_dispcontrol(Rot.tolist(), 0, 1, 2, dir_name=save_path, filename=filename, tol=1, stiff='tan')

St.save_structure(f'Beam_Bilinear_Alpha={A}')

St.plot_structure(scale=10)
