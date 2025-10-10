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

E = 30e9
NU = 0.0

St = st.Structure_2D()

nb_fe = 50
for i in range(nb_fe):
    St.add_fe(N2 * i / nb_fe, N2 * (i + 1) / nb_fe, E, NU, H, b=B, lin_geom=True, rho=7000)

St.make_nodes()
St.make_cfs(True)

F = -100e3

St.loadNode(N2, [1], F, fixed=True)
St.fixNode(N1, [0, 1, 2])

# St.solve_modal()
# St.plot_modes(3)

# print(St.eig_vals[-1])
# print(max(St.eig_vals))
# # for i in range(2):
#     print(St.eig_modes.T[i,-2:])
# St.fixNode(N2, [1])
St.solve_linear()

St.plot_structure(scale=20, plot_cf=False)

St.reset_loading()

print(St.U[-2])
# U0 = St.U.copy()
# St.plot_structure(scale=100, plot_cf=False)
St.set_damping_properties(xsi=0.02, damp_type='RAYLEIGH')

Meth = 'LA'

St.solve_dyn_linear(.5, 2e-5, Meth=Meth, filename='_Timoshenko', dir_name='dt=2e-5')

St.plot_structure(scale=20, plot_cf=False)
