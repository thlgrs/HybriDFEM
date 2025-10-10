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

BLOCKS = 25
CPS = 25

E = 30e9
NU = 0.0
FY = 20e6
# A = .001

filename = f'Beam_ForceControl'

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, 100., b=B, material=mat.Bilinear_Mat(E, NU, FY, .1))
St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

F = -90e3

St.loadNode(N2, [1], F)
St.fixNode(N1, [0, 1, 2])
# St.fixNode(N2, [0])

St.solve_forcecontrol(25, max_iter=100, dir_name=save_path, filename=filename)

St.save_structure(filename)

import matplotlib.pyplot as plt
import pickle

with open(f'Beam_ForceControl.pkl', 'rb') as file:
    St = pickle.load(file)
    nb_blocks = len(St.list_blocks)
    Lb = 3. / nb_blocks

results = f'Beam_ForceControl.h5'

with h5py.File(results, 'r') as hf:
    # Import what you need
    U = hf['U_conv'][-2]
    F = hf['P_r_conv'][-2]

    last_conv1 = hf['Last_conv'][()]
    last_def1 = hf['U_conv'][:, last_conv1]

plt.plot(-U, -F, '*-')
