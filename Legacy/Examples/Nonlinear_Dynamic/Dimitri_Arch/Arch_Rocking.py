import numpy as np
import os
import h5py
import sys
import pathlib
import matplotlib.pyplot as plt

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as cont

save_path = os.path.dirname(os.path.abspath(__file__))

# Meth = ['HHT', 0.3]
Meth = 'CDM'

kn = 40e9
ks = kn

rho = 262.

L_base = 40
H_base = 2
x = np.array([.5, 0])
y = np.array([0, .5])

St = st.Structure_2D()
# Base
N1 = np.array([0, -H_base / 2], dtype=float)
vertices = np.array([N1, N1, N1, N1])
vertices[0] += L_base * x - H_base * y
vertices[1] += L_base * x + H_base * y
vertices[2] += -L_base * x + H_base * y
vertices[3] += -L_base * x - H_base * y

St.add_block(vertices, rho)

vertices = np.array([[-10, 0],
                     [-10, 16.4],
                     [-10 + 10 * (1 - np.cos(np.pi / 12)), 16.4 + 10 * np.sin(np.pi / 12)],
                     [-10 + 10 * (1 - np.cos(np.pi / 6)), 16.4 + 10 * np.sin(np.pi / 6)],
                     [-10, 16.4 + 10 * np.tan(np.pi / 6)],
                     [-10, 30],
                     [-15, 30],
                     [-15, 0]], dtype=float)

St.add_block(vertices, rho)

vertices[:, 0] *= -1
vertices = np.flip(vertices, axis=0)
St.add_block(vertices, rho)

C = np.array([0, 16.4], dtype=float)
St.add_arch(C, np.pi / 6, 5 * np.pi / 6, 10 + .75, 12, 1.5, rho)

St.make_nodes()
St.make_cfs(False, nb_cps=2, offset=0.0, contact=cont.Coulomb(kn, ks, 1))

for i in range(1, len(St.list_blocks)):
    M = St.list_blocks[i].m
    W = 9.81 * M
    St.loadNode(i, [0], W)
    St.loadNode(i, [1], -W, fixed=True)

St.fixNode(0, [0, 1, 2])
St.plot_structure(scale=10)

# St.solve_modal()
# St.plot_modes()
# print(St.eig_vals)

# %% Excitation function and damping
t_p = .5
a = 7
lag = 0.0


def lmbda(x):
    if x < lag: return 0
    if x < t_p + lag:
        return a
    elif x < 3 * t_p + lag:
        return -a / 2
    else:
        return 0


St.set_damping_properties(xsi=0.01, damp_type='RAYLEIGH')

# %% Computation
St.solve_dyn_nonlinear(5, 2e-4, Meth=Meth, lmbda=lmbda, filename=f't_p={t_p}_a={a}')
St.plot_structure(scale=1, plot_forces=False, plot_cf=False)

St.save_structure(filename='Dimitri_Arch')
# #%% Debug

# # plt.plot()
