import os
import pathlib
import sys

import numpy as np

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as cont

save_path = os.path.dirname(os.path.abspath(__file__))

Meth = ['HHT', 0.01]
# Meth = 'CDM'

kn = 1e7
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
St.make_cfs(False, nb_cps=2, offset=0.0, contact=cont.NoTension(kn, ks, cheating=True))

M = St.list_blocks[1].m
W = 9.81 * M
# St.loadNode(1, [0], W)
St.loadNode(1, [1], -W, fixed=True)

St.fixNode(0, [0, 1, 2])

St.solve_modal()
# St.plot_modes()

print(f'dt should be {np.min(2 * np.pi / St.eig_vals) / np.pi}')

St.plot_structure(scale=1)


def lmbda(x):
    return 1


U0 = np.zeros(6)
U0[4] = 1
# U0[3:] = np.array([ 0.1, 0.06421626, -0.42358177])
# St.solve_dispcontrol(100,L/4,1,0)
# print(St.U[3:])
St.set_damping_properties(xsi=0.0, damp_type='RAYLEIGH')

St.solve_dyn_nonlinear(10, 1e-4, Meth=Meth, U0=U0)
St.plot_structure(scale=1, plot_forces=False, plot_cf=True)

St.save_structure(filename='Rocking_block')
# %% Debug

# plt.plot()
