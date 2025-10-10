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
import Material as mat

save_path = os.path.dirname(os.path.abspath(__file__))

kn = 1e11
ks = 1e11
B = 400e-3
H = 0.4
R = 1.8

nb_blocks = 9

rho = 2000

N1 = np.array([0, 0], dtype=float)
x = np.array([.5, 0])
y = np.array([0, 1])

St1 = st.Structure_2D()

St1.add_arch(N1, 0.4014, np.pi - 0.4014, R, nb_blocks, H, rho, B)

vertices = np.array([[(R + H / 2) * np.cos(0.4014), (R + H / 2) * np.sin(0.4014)],
                     [(R - H / 2) * np.cos(0.4014), (R - H / 2) * np.sin(0.4014)],
                     [(R + H / 2) * np.cos(0.4014), (R - H / 2) * np.sin(0.4014)]])
St1.add_block(vertices, rho, b=B)

vertices = np.array([[-(R + H / 2) * np.cos(0.4014), (R + H / 2) * np.sin(0.4014)],
                     [-(R - H / 2) * np.cos(0.4014), (R - H / 2) * np.sin(0.4014)],
                     [-(R + H / 2) * np.cos(0.4014), (R - H / 2) * np.sin(0.4014)]])
St1.add_block(vertices, rho, b=B)

St1.make_nodes()

nb_cps = np.linspace(-1, 1, 20)
nb_cps = nb_cps.tolist()

# nb_cps
mu = np.tan(np.pi / 6)
fct = 80000
c = fct * mu

St1.make_cfs(False, nb_cps=nb_cps, offset=-1, surface=surf.Coulomb(kn, ks, mu, c))

Ef = 230e11
ft = 0.202e6
r = 400e-3
l_b = 366.87e-3
A = r * l_b
# print(A*ft)
fs = 9.89e3 / A
FRP = mat.frp_reinforcement(ft, fs, Ef)

for cf in St1.list_cfs:
    cf.add_reinforcement(-1, A, material=FRP, height=r)
    cf.add_reinforcement(1, A, material=FRP, height=r)

St1.fixNode(nb_blocks, [0, 1, 2])
St1.fixNode(nb_blocks + 1, [0, 1, 2])

W_tot = 0

for i in range(1, nb_blocks + 1):
    M = St1.list_blocks[i].m
    W = M * 9.81
    St1.loadNode(i, [1], -W, fixed=True)
    W_tot += W

# St1.solve_forcecontrol(20, tol=1, max_iter=100)

Node = nb_blocks - 2
F = 100e3
St1.loadNode(Node, 1, -F)

LIST = np.linspace(0, -5e-4, 200)
# LIST = np.append(LIST, np.linspace(-1e-4, -2e-4, 300))
# LIST = np.append(LIST, np.linspace(-1e-5, -5e-3, 100))

# St1.solve_forcecontrol(100, tol=1, max_iter=100)

LIST = LIST.tolist()
# Node = 4
St1.solve_dispcontrol(LIST, 0, Node, 1, tol=1, max_iter=100, filename='NoTension_DispControl')
# St2.solve_dispcontrol(80, 4e-2, Node, 0, tol=.1, max_iter=1000,filename='Elastic_DispControl')
# St1.solve_modal(4)
# St1.plot_modes(4, scale=50)

# for i in range(1,nb_blocks+2): 
#     M = St.list_blocks[i].m
#     W = M * 9.81
#     St.loadNode(i, [0], -W)

St1.plot_structure(scale=100, plot_forces=True, plot_cf=True, plot_supp=False, save='Arch_Def.eps')

# %% Plot Pushover
file1 = 'NoTension_DispControl.h5'
import matplotlib as mpl

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

with h5py.File(file1, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    P_c = -hf['P_r_conv'][3 * (nb_blocks - 2) + 1, :last_conv]
    U_c = -hf['U_conv'][3 * Node + 1, :last_conv] * 1000

# file2 = 'Elastic_DispControl.h5'

# with h5py.File(file2, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     P_e = hf['P_r_conv'][3*Node,:last_conv] / W
#     U_e = hf['U_conv'][3*Node,:last_conv]*1000

print(max(P_c))
plt.figure(figsize=(4.5, 4.5), dpi=600)
plt.grid(True)
plt.xlabel(r'Control displacement [mm]')
plt.ylabel(r'Load multiplier [-]')
plt.plot(U_c, P_c, color='black')
# plt.plot(U_e,P_e,label='Elastic')
# plt.legend()
# plt.xlim((0,40))
# plt.ylim((0, 0.08))

# plt.savefig('Dispcontrol_arch.eps')
