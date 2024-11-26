import numpy as np
import os
import h5py
import sys
import pathlib

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat

N1 = np.array([0,0], dtype=float)
N2 = np.array([4,0], dtype=float)
N3 = np.array([4,4], dtype=float)
N4 = np.array([0,4], dtype=float)

surface = [N1, N2, N3, N4]

# List of 100 random points inside the surface delimited by the points N1, N2, N3, N4
points = np.random.rand(10,2)*4


St = st.Structure_2D()
St.add_voronoi_surface(surface, points, b=1, rho=100, material=mat.Material(60e9,0.0))
St.make_nodes()
# St.make_cfs(True, nb_cps=10)

St.plot_structure(plot_cf=False, scale=0)