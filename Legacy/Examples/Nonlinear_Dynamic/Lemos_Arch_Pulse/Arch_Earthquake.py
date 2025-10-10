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
mu = .88

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
# L_base = 7.5
vertices = np.array([N1, N1, N1, N1])
vertices[0] += L_base * x - H_base * y
vertices[1] += L_base * x
vertices[2] += - L_base * x
vertices[3] += - L_base * x - H_base * y

St1.add_block(vertices, rho)

St1.add_arch(N1, 0, np.pi, R, nb_blocks, H, rho, B)
St1.make_nodes()

St1.make_cfs(False, nb_cps=[-1, 0, 1], offset=-1, surface=surf.NoTension_CD(kn, ks))
St1.list_cfs[0].change_cps(nb_cp=[-1, 1], offset=-1, surface=surf.NoTension_CD(kn, ks))
St1.list_cfs[1].change_cps(nb_cp=[-1, 1], offset=-1, surface=surf.NoTension_CD(kn, ks))
# St1.plot_structure(scale=0, plot_forces=False, plot_cf=True)
St1.fixNode(0, [0, 1, 2])

for i in range(1, nb_blocks + 1):
    M = St1.list_blocks[i].m
    W = M * 9.81
    St1.loadNode(i, [1], -W)

# # Gravity loads applied statically
Node = 9

# St1.plot_structure(scale=10, plot_forces=True, plot_cf=True)
St1.solve_forcecontrol(5, tol=1e-3, max_iter=100)

St1.plot_structure(scale=1, plot_forces=False, plot_cf=True)

St1.reset_loading()

for i in range(nb_blocks):
    M = St1.list_blocks[i].m
    W = M * 9.81
    St1.loadNode(i, [1], -W, fixed=True)
    St1.loadNode(i, [0], -W)

# %% Excitation function and damping
t_p = .25
w_s = np.pi / t_p
a = -0.15
lag = 0

vlocy = np.loadtxt("data-eq/eq23s1.txt", skiprows=2)

nb_points = 3005
dt = 5e-3
t_end = dt * nb_points
time = np.arange(nb_points) * dt
# print(data, time)
acc = np.gradient(vlocy, dt)

peak = .2
lmbda = acc * peak / np.max(abs(acc))

# if needed, interpolate earthquake record to match desired timestep
dt_new = 1e-4
new_time = np.arange(time[0], time[-1], dt_new)

from scipy.interpolate import interp1d

interpolator = interp1d(time, lmbda, kind="linear", fill_value="extrapolate")
new_lmbda = interpolator(new_time)

# lmbda = lmbda.tolist()

plt.figure(None, figsize=(8, 2))
# plt.plot(new_time, new_lmbda)
plt.plot(time, lmbda)
plt.plot(new_time, new_lmbda)

# %% NL Structure
St1.set_damping_properties(xsi=0.05, damp_type='STIFF')

# #%% Computation 
Meth = 'NWK'
St1.solve_dyn_nonlinear(16, dt_new, Meth=Meth, lmbda=new_lmbda.tolist(), filename=f'NoTension_EQ_{peak}g')
St1.plot_structure(scale=1, plot_forces=False, plot_cf=True)

St1.save_structure(filename='Lemos_Arch_Coulomb')
