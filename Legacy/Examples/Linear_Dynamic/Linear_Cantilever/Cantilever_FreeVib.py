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

BLOCKS = 5
CPS = 15

E = 30e9
NU = 0.0

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, 7000, b=B, material=mat.Material(E, NU, shear_def=True))
St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

St.plot_structure(scale=0, plot_cf=False, save='Beam_Undef.eps')

F = -100e3

St.loadNode(N2, [1], F, fixed=True)
St.fixNode(N1, [0, 1, 2])

St.solve_modal(filename=f'{BLOCKS}', dir_name='conv_blocks')
St.plot_modes(scale=1, save=True)

print(St.eig_vals[:4])

for i in range(4):
    print(St.eig_modes.T[i, -4:])

print(max(St.eig_vals))
St.solve_linear()

St.plot_structure(scale=20, plot_cf=False, save='Beam_Def.eps')

St.reset_loading()

print(St.U[-4:])
# U0 = St.U.copy()
# St.plot_structure(scale=100, plot_cf=False)
St.set_damping_properties(xsi=0.00, damp_type='STIFF')

Meth = 'LA'
dt = 1.61e-4

print(BLOCKS)
St.solve_dyn_linear(5, dt, Meth=Meth, dir_name='conv_blocks', filename=f'dt_')

St.plot_structure(scale=20, plot_cf=False)
