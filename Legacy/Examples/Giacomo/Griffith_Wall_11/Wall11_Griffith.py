import numpy as np
import os
import h5py
import sys
import pathlib

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat
import Surface as surf
import Contact as ct
import ContactPair as cp

save_path = os.getcwd()

A = (110e-3) * (950e-3)  # m^2
phi_r = 35 * np.pi / 180  # rad

E_m = 48e6  # Pa
nu = 0.0

h_b = 86e-3  # m
t_w = 110e-3  # m

kn = E_m / h_b * ((t_w) / (t_w - 2 * 0.06 * t_w))  # *1.251 #Pa/m, check with Michele
ks = kn / (2 * (1 + nu))  # Pa/m

mu = np.tan(phi_r)

St = st.Structure_2D()

St.add_geometry(filepath=r"Griffith_wall_text_file_11.txt", rho=1800, material=None, gravity=False)

list = np.linspace(-0.88, 0.88, 10)
list = list.tolist()
St.make_cfs(lin_geom=True, nb_cps=list, offset=-1, surface=surf.Coulomb(2 * kn, 2 * ks, mu, c=2e6))
# St.make_cfs(lin_geom=True, nb_cps=list, offset=-1, surface=surf.NoTension_EP(2*kn,2*ks))


St.get_M_str()

for i in range(2, 20):
    M = St.list_blocks[i].m
    W = 9.81 * M
    St.loadNode(i, [1], -W, fixed=True)

M_tot = sum(blk.m for blk in St.list_blocks)
W_tot = 9.81 * M_tot

St.plot_structure(scale=0, plot_cf=True, plot_forces=True)  # undeformed configuration

# St.solve_modal(4)
# St.plot_modes(4)

LIST = np.linspace(0, 0.62 * t_w, 100)
St.solve_dispcontrol(LIST.tolist(), 0, 9, 0, dir_name=save_path, filename='results_Griffith_wall_11_static',
                     max_iter=100, tol=1)
St.save_structure('After_Static_W11')

St.plot_structure(scale=1, plot_cf=False, plot_forces=True)  # structure in the deformed configuration

import matplotlib.pyplot as plt
from scipy import interpolate

# Lettura dati da simulazione
with h5py.File("Results_Griffith_wall_11_static.h5", "r") as hf:
    U = hf["U_conv"][(9) * 3 + 0]
    P = hf["P_r_conv"][(9) * 3 + 0]

d_tw = U / t_w
F_W = P / W_tot

# Plot curva simulazione
plt.plot(d_tw, F_W, label='HybriDEFM Coulomb', color='tab:blue')

labels = ['rigid-body analysis', 'experimental', 'discrete elements']
colors = ['tab:green', 'tab:orange', 'tab:red']

# for i in range(1, 4):
#     data = np.loadtxt(f'curve_{i}_11_static.csv', delimiter=',')
#     x = data[:, 0] 
#     y = data[:, 1] 
#     plt.plot(x, y, label=labels[i - 1], color=colors[i - 1], linestyle='--')

# # horizontal line at the maximum of rigid-body analysis
# rigid_body = np.loadtxt('curve_1_11_static.csv', delimiter=',')
# max_rigid_body = np.max(rigid_body[:, 1])
# plt.axhline(y=max_rigid_body, color='black', linestyle='--', linewidth=1, label='limit analysis')


# Grafico finale
plt.title("Pushover curve specimen 11")
plt.xlabel("d/t_w [-]")
plt.ylabel("F/W [-]")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
# plt.savefig("pushover_test_11_lin_geometry.eps", format='eps', dpi=300)
plt.show()

St.reset_loading()

for i in range(2, 20):
    M = St.list_blocks[i].m
    W = 9.81 * M
    St.loadNode(i, [1], -W, fixed=True)

Meth = 'CDM'
damp = 0.005

St.set_damping_properties(xsi=damp, damp_type='STIFF', stiff_type='TAN_LG')  # 20% of damping
St.set_lin_geom(False)

St.solve_dyn_nonlinear(3.5, 1e-5, Meth=Meth, filename=f'W11_{damp * 100}', dir_name=save_path)
# duration of the simulation , time step

St.plot_structure(scale=1, plot_cf=False, plot_forces=True)  # structure in the deformed configuration

# %%
import matplotlib.pyplot as plt
from scipy import interpolate

# Lettura dati da simulazione
with h5py.File(f'W11_{damp * 100}_CDM_.h5', "r") as hf:
    U = hf["U_conv"][(9) * 3 + 0]
    V = hf["V_conv"][(9) * 3 + 0]
    time = hf['Time'][()]

d_tw = U / t_w
# time = np.linspace(0, 3.5, 350001)


# Plot curva simulazione
plt.plot(time, d_tw, label=f'xi={damp * 100}%', color='tab:orange')
plt.plot(time, V, label='velocity', color='tab:green')

labels = ['experimental', 'experimental', 'discrete elements']
colors = ['tab:gray', 'tab:grey', 'tab:blue']

# for i in range(1, 4):
#     data = np.loadtxt(f'curve_{i}_11_free.csv', delimiter=',')
#     x = data[:, 0] 
#     y = data[:, 1] 
#     plt.plot(x, y, label=labels[i - 1], color=colors[i - 1], linestyle='--')

# plt.yticks(np.arange(-1.0, 1.0 + 0.01, 0.5))  # y: -1.0, -0.5, 0.0, 0.5, 1.0
# plt.xticks(np.arange(0.00, 3.5 + 0.01, 0.7))   # x: 0.7, 1.4, 2.1, 2.8, 3.5

# Grafico finale
plt.title("Release test simulation")
plt.xlabel("time [s]")
plt.ylabel("d/t_w [-]")
plt.ylim((-1, 1))
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
# plt.savefig("release_test_11_damp1_step5_CDM.eps", format='eps', dpi=300)
plt.show()
