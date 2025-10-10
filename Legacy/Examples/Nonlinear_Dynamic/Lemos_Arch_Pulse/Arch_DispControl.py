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

B = 1.
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

St1.add_block(vertices, rho)

St1.add_arch(N1, 0, np.pi, R, nb_blocks, H, rho, B)
St1.make_nodes()

nb_cps = np.linspace(-1, 1, 3)
# print(nb_cps)
nb_cps = nb_cps.tolist()
# nb_cps=[-1,0,111]
mu = .7
# nb_cps
# St1.make_cfs(False, nb_cps=nb_cps, offset=-1, surface=surf.Surface(kn,ks))
St1.make_cfs(False, nb_cps=nb_cps, offset=-1, surface=surf.Coulomb(kn, ks, mu))
# St1.make_cfs(False, nb_cps=nb_cps, offset=-1, surface=surf.NoTension_CD(kn,ks))

# St1.list_cfs[0].change_cps(nb_cp=[-1, 1], offset=-1,surface=surf.Coulomb(kn,ks,mu))
# St1.list_cfs[1].change_cps(nb_cp=[-1, 1], offset=-1,surface=surf.Coulomb(kn,ks,mu))
St1.plot_structure(scale=0, plot_forces=False, plot_cf=True)
St1.fixNode(0, [0, 1, 2])

# St1.solve_modal(1)
# St1.plot_modes(1, scale=50, save=True)

for i in range(1, nb_blocks + 1):
    M = St1.list_blocks[i].m
    W = M * 9.81
    St1.loadNode(i, [1], -W)
#     # 
# St1.plot_structure(scale=10, plot_forces=False, plot_cf=False, plot_supp=False, save='Arch_Undef.eps')

St1.solve_forcecontrol(20, tol=1, max_iter=100)
St1.plot_structure(scale=1000, plot_forces=False, plot_cf=False, plot_supp=False, save='Arch_Undef.eps')

# # Gravity loads applied statically
Node = 9
# St1.fixNode(Node,0)
# St1.plot_structure(scale=10, plot_forces=False, plot_cf=True)
St1.reset_loading()

for i in range(1, nb_blocks + 1):
    M = St1.list_blocks[i].m
    W = M * 9.81
    St1.loadNode(i, [1], -W, fixed=True)
    St1.loadNode(i, [0], W)

St1.solve_dispcontrol(200, 4e-2, Node, 0, tol=10, max_iter=100, filename='NoTension_DispControl')
# St2.solve_dispcontrol(80, 4e-2, Node, 0, tol=.1, max_iter=1000,filename='Elastic_DispControl')
# St1.solve_modal(4)
# St1.plot_modes(4, scale=50)

# for i in range(1,nb_blocks+2): 
#     M = St.list_blocks[i].m
#     W = M * 9.81
#     St.loadNode(i, [0], -W)

St1.plot_structure(scale=1, plot_forces=False, plot_cf=True, plot_supp=False, save='Arch_Def.eps')

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
    P_c = hf['P_r_conv'][3 * Node, :last_conv] / W
    U_c = hf['U_conv'][3 * Node, :last_conv] * 1000

# file2 = 'Elastic_DispControl.h5'

# with h5py.File(file2, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     P_e = hf['P_r_conv'][3*Node,:last_conv] / W
#     U_e = hf['U_conv'][3*Node,:last_conv]*1000

# St1.solve_modal()
# St1.plot_modes(4, scale=-1)

print(np.linalg.cond(St1.K[np.ix_(St1.dof_free, St1.dof_free)]))

print(max(P_c))
plt.figure(figsize=(4.5, 4.5), dpi=600)
plt.grid(True)
plt.xlabel(r'Control displacement [mm]')
plt.ylabel(r'Load multiplier [-]')
plt.plot(U_c, P_c, color='black')
# plt.plot(U_e,P_e,label='Elastic')
# plt.legend()
plt.xlim((0, 40))
plt.ylim((0, 0.08))

# plt.savefig('Dispcontrol_arch.eps')
