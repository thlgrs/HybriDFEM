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
N2 = np.array([1, 0], dtype=float)

H = .2
B = .2

BLOCKS = 20
CPS = 10

E = 10e9
NU = 0.0

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, 3500000, b=B, material=mat.Material(E, NU, shear_def=True))
St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

St.plot_structure(scale=0, plot_cf=False, save='Beam_Undef.eps')
St.save_structure('Beam_test')

St.fixNode(N1, [0, 1, 2])
# St.fixNode(N2, [0,1,2])

# St.solve_modal()
# print(max(St.eig_vals))

St.get_P_r()

St.set_damping_properties(xsi=0.01, damp_type='STIFF', stiff_type='TAN_LG')

dt = 1e-3
t_end = 15

time = np.arange(0, t_end, dt)
time = np.append(time, t_end)
w_s = 10
Amp = 2e-3
U_app = Amp * np.sin(w_s * time)

St.impose_dyn_excitation(0, 1, U_app, dt)
# St.impose_dyn_excitation(BLOCKS-1,1,-U_app,dt)

St.solve_dyn_linear(t_end, dt, Meth='CAA', filename=f'First_Test')

St.plot_structure(scale=20, plot_cf=False)
