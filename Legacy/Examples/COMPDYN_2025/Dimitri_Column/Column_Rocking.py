import numpy as np
import os
import h5py
import sys
import pathlib
import matplotlib.pyplot as plt
import importlib


def reload_modules():
    importlib.reload(st)
    importlib.reload(ct)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as ct

reload_modules()

save_path = os.path.dirname(os.path.abspath(__file__))

Meth = ['HHT', 0.3]
# Meth = 'CDM'

kn = 1e8
ks = kn

B1 = 1.11
B2 = .9
H = 5.95
H_small = 0.21

dH = (H - H_small) / 7
dB = (B1 - B2) / H
B2 = B1 - dB * dH

rho = 2620

L_base = 2.5
H_base = .75

N1 = np.array([0, -H_base / 2], dtype=float)
N2 = np.array([0, dH / 2], dtype=float)
N3 = np.array([0, H - H_small / 2], dtype=float)
x = np.array([.5, 0])
y = np.array([0, .5])

St = st.Structure_2D()

vertices = np.array([N1, N1, N1, N1])
vertices[0] += L_base * x - H_base * y
vertices[1] += L_base * x + H_base * y
vertices[2] += -L_base * x + H_base * y
vertices[3] += -L_base * x - H_base * y

St.add_block(vertices, rho)

for i in range(7):
    vertices = np.array([N2, N2, N2, N2])
    vertices[0] += B1 * x - dH * y
    vertices[1] += B2 * x + dH * y
    vertices[2] += -B2 * x + dH * y
    vertices[3] += -B1 * x - dH * y
    B1 = B2
    B2 -= dB * dH
    N2 += 2 * dH * y

    St.add_block(vertices, rho)

B2 = B1 - dB * H_small
vertices = np.array([N3, N3, N3, N3])
vertices[0] += B1 * x - H_small * y
vertices[1] += B2 * x + H_small * y
vertices[2] += -B2 * x + H_small * y
vertices[3] += -B1 * x - H_small * y
St.add_block(vertices, rho)

St.make_nodes()

St.make_cfs(False, nb_cps=2, offset=0.0, contact=ct.NoTension(kn, ks))

for i in range(1, 9):
    M = St.list_blocks[i].m
    W = 9.81 * M
    St.loadNode(i, [0], W)
    St.loadNode(i, [1], -W, fixed=True)

St.fixNode(0, [0, 1, 2])
St.plot_structure(scale=1)

# St.solve_modal(2)
# St.plot_modes(2,scale=10)
# print(St.eig_vals)

# %% Excitation function and damping
t_p = .8
a = .2
lag = 0.1


def lmbda(x):
    if x < lag: return 0
    if x < t_p + lag:
        return a
    elif x < 3 * t_p + lag:
        return -a / 2
    else:
        return 0


St.set_damping_properties(xsi=0.00, damp_type='RAYLEIGH')

# %% Computation
St.solve_dyn_nonlinear(10 + lag, 1e-3, Meth=Meth, lmbda=lmbda, filename=f't_p={t_p}_a={a}')
St.plot_structure(scale=1, plot_forces=False, plot_cf=True)

St.save_structure(filename='Dimitri_Column')
# #%% Debug

# # plt.plot()
# %%
