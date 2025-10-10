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

kn = 20e9
ks = 0.4 * kn
mu = .88

B = 1.
H = 0.5
L = 0.5
nb_blocks = 1

rho = 2700

N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, L], dtype=float)
x = np.array([.5, 0])
y = np.array([0, .5])

St1 = st.Structure_2D()

vertices = np.array([N1, N1, N1, N1])
vertices[0] += L * x - H * y
vertices[1] += L * x + H * y
vertices[2] += - L * x + H * y
vertices[3] += - L * x - H * y

St1.add_block(vertices, rho)

vertices = np.array([N2, N2, N2, N2])
vertices[0] += L * x - H * y
vertices[1] += L * x + H * y
vertices[2] += - L * x + H * y
vertices[3] += - L * x - H * y

St1.add_block(vertices, rho)

St1.make_nodes()

St1.make_cfs(False, nb_cps=2, offset=0, contact=cont.Coulomb(kn, ks, 1))
St1.fixNode(0, [0, 1, 2])
# St1.fixNode(1,2)

M = St1.list_blocks[0].m
W = M * 9.81
St1.loadNode(1, [1], -W, fixed=True)
St1.loadNode(1, [0], W)

St1.plot_structure(scale=1, plot_forces=True)

# # Gravity loads applied statically
Node = 1
St1.solve_dispcontrol(2, 5e-4, Node, 0, tol=.1, max_iter=100, filename='OneBlock_DispControl')

St1.plot_structure(scale=1, plot_forces=False, plot_cf=True)

# print(St1.list_cfs[0].cps[-1].sp1.law.disps['n'], St1.list_cfs[0].cps[-1].sp1.law.disps['s'])
# print(St1.list_cfs[0].cps[-1].sp2.law.disps['n'], St1.list_cfs[0].cps[-1].sp2.law.disps['s'])
# print(St1.list_cfs[0].cps[-1].sp1.law.stiff['kn'], St1.list_cfs[0].cps[-1].sp1.law.stiff['ks'])
# print(St1.list_cfs[0].cps[-1].sp2.law.stiff['kn'], St1.list_cfs[0].cps[-1].sp2.law.stiff['ks'])
# print(St1.list_cfs[0].cps[-1].q_bsc)
# print(St1.list_cfs[0].cps[-2].sp1.law.disps['n'])
# print(St1.list_cfs[0].cps[-2].q_bsc)
# print(St1.list_cfs[0].cps[-2].sp1.law.disps['n'])


# %% Plot Pushover
file1 = 'OneBlock_DispControl.h5'

with h5py.File(file1, 'r') as hf:
    # Import what you need
    # last_conv = hf['Last_conv'][()]
    P = hf['P_r_conv'][3 * Node] / W
    U = hf['U_conv'][3 * Node] * 1000

# print(P,U)
plt.figure(None, (6, 6))
plt.grid(True)
plt.xlabel(r'Control displacement [mm]')
plt.ylabel(r'Load multiplier [-]')
plt.plot(U, P, '*-', label='Coulomb - No Tension')
plt.legend()
# plt.xlim((0,40))
# plt.ylim((0, 0.1))
