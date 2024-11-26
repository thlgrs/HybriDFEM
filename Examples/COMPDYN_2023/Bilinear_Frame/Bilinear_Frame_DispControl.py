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
N2 = np.array([0, 3], dtype=float)
N3 = np.array([3, 3], dtype=float)
N4 = np.array([3, 0], dtype=float)

H_B = .2
H_C = .2 * 2**(1/3) 
B = .2

BLOCKS = 40
CPS = 30

E = 30e9
NU = 0.0
FY = 20e6
A = 0.0

Lin_Geom = False
FEs = False

if Lin_Geom: text_lin_geom = 'Linear'
else: text_lin_geom = 'P-Delta'

if FEs: text_fes = '_Coupled'
else: text_fes = '_Full'

filename = f'Frame_BilinMat_' + text_lin_geom + text_fes

St = st.Structure_2D()

if FEs: 
    St.add_fe(N1, N2, E, NU, H_C, b=B, lin_geom=Lin_Geom)
    St.add_fe(N3, N4, E, NU, H_C, b=B, lin_geom=Lin_Geom)
else: 
    St.add_beam(N1, N2, BLOCKS, H_C, 100., b=B, material=mat.Bilin_Mat(E, NU, FY, A))
    St.add_beam(N3, N4, BLOCKS, H_C, 100., b=B, material=mat.Bilin_Mat(E, NU, FY, A))

St.add_beam(N2, N3, BLOCKS, H_B, 100., b=B, material=mat.Bilin_Mat(E, NU, FY, A))

St.make_nodes()
St.make_cfs(Lin_Geom, nb_cps=CPS)

F = 100e3

St.loadNode(N2, [0], F)
St.loadNode(N2, [1], -F, fixed=True)
St.loadNode(N3, [1], -F, fixed=True)
St.fixNode(N1, [0,1])
St.fixNode(N4, [0,1])

St.plot_structure(plot_cf=False, scale=0)

if FEs: control_node = St.get_node_id(N2)
else: control_node = BLOCKS-1  
    
# St.solve_dispcontrol(65, 65e-3, control_node, 0, dir_name=save_path, filename=filename, tol=1)

St.save_structure(f'Frame_BilinMat_' + text_lin_geom + text_fes)
St.plot_structure(scale=10, plot_cf=False)
