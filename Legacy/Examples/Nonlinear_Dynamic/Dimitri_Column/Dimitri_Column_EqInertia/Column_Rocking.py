import numpy as np
import os
import h5py
import sys
import pathlib
import matplotlib.pyplot as plt


def reload():
    import importlib
    importlib.reload(st)
    importlib.reload(cont)
    importlib.reload(cp)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as cont
import ContactPair as cp
import Surface as surf

reload()

save_path = os.path.dirname(os.path.abspath(__file__))

# Meth =['HHT', 0.3]


kn = 5e10
ks = 5e10

B1 = 1.11
B2 = .9
H = 5.95
H_small = 0.21

nb_blocks = 7

dH = (H - H_small) / nb_blocks
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

for i in range(nb_blocks):
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

mu = 0.7
St.make_cfs(False, nb_cps=[-1, 1], offset=-1, surface=surf.NoTension_EP(kn, ks))
# St.make_cfs(False, nb_cps=[-1,1], offset=-1, surface=surf.Coulomb(kn,ks,mu))
St.fixNode(0, [0, 1, 2])

# St.solve_modal()
# print(max(St.eig_vals))

for i in range(1, nb_blocks + 2):
    M = St.list_blocks[i].m
    W = M * 9.81
    St.loadNode(i, [1], -W, fixed=True)

# Gravity loads applied statically
St.solve_forcecontrol(10)

for i in range(1, nb_blocks + 2):
    M = St.list_blocks[i].m
    W = M * 9.81
    St.loadNode(i, [0], -W)

St.plot_structure(scale=1)

# %% Excitation function and damping
t_p = .2
a = 1.4

lag = 0.0

print(f'Period is {t_p}s and amplitude is {a}g')


def lmbda(x):
    if x < lag: return 0
    if x < t_p + lag: return a
    if x < 3 * t_p + lag: return -a / 2
    return 0


St.set_damping_properties(xsi=0.0, damp_type='INIT')

# %% Computation
# Meth = 'CAA'
Meth = ['HHT', .3]
St.solve_dyn_nonlinear(2, 5e-4, Meth=Meth, lmbda=lmbda, filename=f't_p={t_p}_a={a}')
St.plot_structure(scale=1, plot_forces=False, plot_cf=True)

St.save_structure(filename='Dimitri_Column')

# # plt.plot()
