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
import Contact as cont

save_path = os.path.dirname(os.path.abspath(__file__))

N0 = np.array([0, 0], dtype=float)
N1 = np.array([3, 0], dtype=float)
N2 = np.array([0, 2], dtype=float)
N3 = np.array([2, 1], dtype=float)
N4 = np.array([0, 3], dtype=float)
N5 = np.array([3, 3], dtype=float)
N6 = np.array([2, 4], dtype=float)

St = st.Structure_2D()

vertices = np.array([N0, N1, N5, N3])
St.add_block(vertices, 100.)

vertices = np.array([N0, N3, N2])
St.add_block(vertices, 100.)

vertices = np.array([N2, N3, N5, N6, N4])
St.add_block(vertices, 100.)

St.make_nodes()
St.make_cfs(True, nb_cps=2, offset=0.0, contact=cont.NoTension(100, 100))

St.plot_structure(plot_cf=True, scale=0)
