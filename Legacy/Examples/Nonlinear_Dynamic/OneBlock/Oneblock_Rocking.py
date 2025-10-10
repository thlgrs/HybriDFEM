import os
import pathlib
import sys

import numpy as np

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as cont
import Surface as surf

save_path = os.path.dirname(os.path.abspath(__file__))

Meth = 'NWK'
Meth = 'CDM'

kn = 1e8
ks = kn

H = .4
L = .4
B = 1

rho = 1000

L_base = 1
H_base = .2

N1 = np.array([0, -H_base / 2], dtype=float)
N2 = np.array([0, H / 2], dtype=float)
x = np.array([.5, 0])
y = np.array([0, .5])

St = st.Structure_2D()

vertices = np.array([N1, N1, N1, N1])
vertices[0] += L_base * x - H_base * y
vertices[1] += L_base * x + H_base * y
vertices[2] += -L_base * x + H_base * y
vertices[3] += -L_base * x - H_base * y

St.add_block(vertices, rho, b=B)

vertices = np.array([N2, N2, N2, N2])
vertices[0] += L * x - H * y
vertices[1] += L * x + H * y
vertices[2] += -L * x + H * y
vertices[3] += -L * x - H * y

St.add_block(vertices, rho, b=B)

St.make_nodes()
St.make_cfs(False, nb_cps=2, offset=0.0, contact=cont.Coulomb(kn, ks, 100))

M = St.list_blocks[1].m
W = 9.81 * M
print(f'Weight of upper block is {W}')
I = St.list_blocks[1].I
print(f'Rot ineritia is {I}')
St.loadNode(1, [1], -W, fixed=True)
St.fixNode(0, [0, 1, 2])

St.solve_forcecontrol(10)

St.loadNode(1, [0], W)

R = 2 * 0.141
period = 2 * np.sqrt(I / (W * R))

AMP = np.pi / 2 * 1.2
lag = 0


def lmbda(t):
    if t < period + lag:
        return AMP * np.sin((t - lag) * np.pi / period)
    return 0


import matplotlib.pyplot as plt

x = np.linspace(0, 2, 100)
y = [lmbda(x[i]) for i in range(100)]
plt.plot(x, y)

St.set_damping_properties(xsi=0.00, damp_type='RAYLEIGH')
# St.U = np.zeros(St.nb_dofs)
St.solve_dyn_nonlinear(2, 1e-4, Meth=Meth, lmbda=lmbda)
St.plot_structure(scale=1, plot_forces=False, plot_cf=True)

St.save_structure(filename='Rocking_block')
# %% Debug

# plt.plot()
