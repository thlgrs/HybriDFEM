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
import ContactPair as cp
import Surface as surf

save_path = os.path.dirname(os.path.abspath(__file__))

# Meth =['HHT', 0.3]
# Meth = 'CAA'
# Meth = 'CDM'

kn = 20e9
ks = 0.4 * kn
mu = .4

B = .5
H = 0.5
R = 7.5 / 2 + H / 2

nb_blocks = 17

rho = 2700

N1 = np.array([0, 0], dtype=float)
x = np.array([.5, 0])
y = np.array([0, 1])

St1 = st.Structure_2D()

H_base = .5
L_base = 1.1 * 2 * R
vertices = np.array([N1, N1, N1, N1])
vertices[0] += L_base * x - H_base * y
vertices[1] += L_base * x
vertices[2] += - L_base * x
vertices[3] += - L_base * x - H_base * y

St1.add_block(vertices, rho, B)

St1.add_arch(N1, 0, np.pi, R, nb_blocks, H, rho, B)
St1.make_nodes()

nb_cps = np.linspace(-1, 1, 3)
nb_cps = nb_cps.tolist()

St1.make_cfs(False, nb_cps=nb_cps, offset=-1, surface=surf.Coulomb(kn, ks, mu))
# St1.list_cfs[0].change_cps(nb_cp=[-1, 1], offset=-1,surface=surf.Coulomb(kn,ks,mu))
# St1.list_cfs[1].change_cps(nb_cp=[-1, 1], offset=-1,surface=surf.Coulomb(kn,ks,mu))

St1.plot_structure(scale=0, plot_forces=False, plot_cf=True)
St1.fixNode(0, [0, 1, 2])

for i in range(1, nb_blocks + 1):
    M = St1.list_blocks[i].m
    W = M * 10
    St1.loadNode(i, [1], -W)

Node = 9

St1.solve_forcecontrol(20, tol=1, max_iter=100)

St1.plot_structure(scale=1, plot_forces=False, plot_cf=True)

St1.reset_loading()

for i in range(nb_blocks):
    M = St1.list_blocks[i].m
    W = M * 10
    St1.loadNode(i, [1], -W, fixed=True)
    St1.loadNode(i, [0], -W)

# %% Excitation function and damping
t_p = .25
w_s = np.pi / t_p
a = -0.15
lag = 0


# print(f'Period is {t_p}s and amplitude is {a}g')

def lmbda(x):
    if x < lag: return 0
    if x < t_p + lag: return a * np.sin(w_s * (x - lag))
    return 0


damp = 0.05
stiff_type = 'TAN'
St1.set_damping_properties(xsi=damp, damp_type='STIFF', stiff_type=stiff_type)

# #%% Computation
Meth = 'CAA'
St1.solve_dyn_nonlinear(2.5, 2e-3, Meth=Meth, lmbda=lmbda, filename=stiff_type + f'F_NoTension_{-a}g_{damp}')
St1.plot_structure(scale=1, plot_forces=False, plot_cf=True)

St1.save_structure(filename='Lemos_Arch_Coulomb')

print(St1.damp_coeff)
