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
import Surface as surf

save_path = os.path.dirname(os.path.abspath(__file__))

N0 = np.array([0, 0], dtype=float)
N1 = np.array([0, 1], dtype=float)
N2 = np.array([0, 3], dtype=float)

E = 30e9
NU = 0.
material = mat.Material(E, NU)

St = st.Structure_2D()

St.add_beam(N0, N1, 2, 1., 0., b=1., material=material)
St.add_fe(N1, N2, E, NU, 1., lin_geom=True)

St.make_nodes()
St.make_cfs(False, nb_cps=10)

St.plot_structure()

# First simulation

n0 = St.get_node_id(N0)
n1 = St.get_node_id(N1)
n2 = St.get_node_id(N2)

St.fixNode(n0, [0, 1, 2])
# St.fixNode(n1,[0,2])
St.loadNode(n2, 1, -1e3)

# St.solve_dispcontrol(10, 1e-1, 1, 1, max_iter=2)
St.solve_forcecontrol(10, max_iter=5)

filename = 'Results_ForceControl.h5'

with h5py.File(filename, 'r') as hf:
    U1 = hf['U_conv'][-2] * 1000
    P1 = hf['P_r_conv'][-2] / 1000

St.fixNode(n2, [1])

St.reset_loading()
St.loadNode(n1, 1, -1e3)

St.solve_forcecontrol(10, max_iter=5)
# St.U[7] += 3.33e-8
# St.U[4] += 3.33e-8
# St.get_P_r()
print(St.P_r)

with h5py.File(filename, 'r') as hf:
    U2 = hf['U_conv'][-2] * 1000
    P2 = hf['P_r_conv'][-2] / 1000

import matplotlib.pyplot as plt

plt.plot(np.append(U1, U2), np.append(P1, P2))

St.plot_structure(scale=1)
