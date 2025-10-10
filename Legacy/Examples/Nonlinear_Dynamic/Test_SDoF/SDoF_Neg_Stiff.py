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

kn = -100
ks = 100

nb_blocks = 2

rho = 1

L_base = 2.5
H_base = .75

N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, 1], dtype=float)

x = np.array([.5, 0])
y = np.array([0, .5])

St = st.Structure_2D()

L = 1
H = 1

vertices = np.array([N1, N1, N1, N1])
vertices[0] += L * x - H * y
vertices[1] += L * x + H * y
vertices[2] += -L * x + H * y
vertices[3] += -L * x - H * y

St.add_block(vertices, rho)

vertices = np.array([N2, N2, N2, N2])
vertices[0] += L * x - H * y
vertices[1] += L * x + H * y
vertices[2] += -L * x + H * y
vertices[3] += -L * x - H * y

St.add_block(vertices, rho)

St.make_nodes()

St.make_cfs(False, nb_cps=1, offset=-1, surface=surf.Surface(kn, ks))
St.fixNode(0, [0, 1, 2])
St.fixNode(1, [0, 2])

St.solve_modal()
print(St.eig_vals)

St.plot_structure(scale=1)

# %% Computation

St.set_damping_properties(xsi=0.00, damp_type='STIFF')
St.get_P_r()
# Meth=['HHT',.3]
St.U[4] = .1
St.plot_structure(scale=1)
t_end = .5
dt = 1e-2
Meth = ['HHT', 0.3]
# Meth = 'CAA'
St.solve_dyn_linear(t_end, dt, Meth=Meth, lmbda=None, filename=f'Test_Results')
# St.plot_structure(scale=1, plot_forces=False, plot_cf=True)

files = ['Test_Results_CDM_.h5',
         'Test_Results_NWK_g=0.5_b=0.25.h5',
         'Test_Results_NWK_g=0.5_b=0.17.h5',
         'Test_Results_GEN_am=0_af=0.3_g=0.8_b=0.42.h5']

plt.figure(None, figsize=(6, 6), dpi=400)

for i, file in enumerate(files):
    with h5py.File(file, "r") as hf:
        U_end = hf["U_conv"][4] * 1000
        time = hf['Time'][()]

    plt.plot(time, U_end, linewidth=1)
    # plt.plot(time, U_c, label=None,linewidth=.5, color=colors[i])

time_a = np.linspace(0, t_end, 1000)
w_r = 7.07106781
u0 = .1
u_a = u0 * (np.exp(-w_r * time_a) + np.exp(w_r * time_a)) / 2
plt.plot(time_a, u_a * 1000, linewidth=1)
plt.xlabel("Time [s]")
plt.ylabel("Displacement [mm]")
plt.yscale('log')
plt.ylim((99, 1000))
plt.xlim((0, .5))
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
# plt.legend()
