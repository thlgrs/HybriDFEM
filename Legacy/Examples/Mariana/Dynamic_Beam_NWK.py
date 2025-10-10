# -*- coding: utf-8 -*-
"""
CA2 Column – Force-Controlled Simulation with Sinusoidal Loading (HybridFEM)
"""

import numpy as np
import os
import h5py
import sys
import pathlib
import matplotlib.pyplot as plt

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat
import Surface as surf
import Contact as ct
import ContactPair as cp

# === Define folder paths
save_path = os.getcwd()
# results_folder = pathlib.Path(save_path) / "Results"
# results_folder.mkdir(exist_ok=True)

# === Geometry
h = 0.015
a = 8 * h
t = 0.0135
n_blocks = 20

N1 = np.array([0.0, 0.0])
N2 = np.array([0.0, a])

# === Material
E = 193333.33e6
FY = 580e6
H = 15425.53e6
r = 0.025
NU = 0.3
material = mat.Mixed_Hardening_Mat(E, NU, FY, H, r)

# === Max force based on section modulus and yield moment
Wb = t * h ** 2 / 6
My = Wb * FY
Pmax = 2 * My / a
# Pmax = 2500

# === Sinusoidal load parameters
T = 1.0  # Total time [s]
n_steps = 1000  # Number of time steps
omega = 10 * np.pi  # Frequency
time_vec = np.linspace(0, T, n_steps + 1)
lambda_list = list(np.sin(omega * time_vec))

# === Create and configure structure
S = st.Structure_2D()
S.add_beam(N1, N2, n_blocks, h, rho=7800, b=t, material=material)
S.make_nodes()
S.make_cfs(lin_geom=True, nb_cps=20)
S.fixNode(N1, [0, 1, 2])  # Fixed base

# === Apply base force ONCE
S.reset_loading()
S.loadNode(N2, [0], Pmax)  # Base force, will be scaled by lambda_list

# === Plot undeformed structure before solving
S.plot_structure(
    scale=0,  # No deformation
    plot_cf=True,  # Show contact pairs/interfaces
    plot_forces=True,  # Show applied forces
    plot_supp=True  # Show supports
)
plt.title("Undeformed Structure: CA2 Column")
plt.show()

# === Solve sinusoidal problem in a single call
filename = "CA2_Column_NEWMARK"
# start_time = time.time()  # <-- antes de resolver

dt = T / n_steps
S.set_damping_properties(xsi=0.0)  # No damping
S.solve_dyn_nonlinear(
    T=T,
    dt=dt,
    lmbda=lambda_list,  # your time-dependent load multiplier
    Meth='NWK',  # or 'GEN', 'CDM', etc. (see ask_method for options)
    filename=filename,
    dir_name=save_path
)

S.save_structure(filename)

# end_time = time.time()    # <-- después de resolver
# elapsed = end_time - start_time
# print(f"⏱️ Simulation time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
# %%

# === Read results and extract tip node data
results_file = os.path.join(save_path, filename + "_NWK_g=0.5_b=0.25.h5")
with h5py.File(results_file, 'r') as hf:
    U_all = hf['U_conv'][-3, :]  # X displacement at top node
    l_all = hf['Load_Multiplier'][()]

# === Plot force–displacement curve (publication style)
plt.figure(figsize=(6, 5))
plt.plot(U_all * 1e3, l_all * Pmax / 1e3, '-', color='black', linewidth=1.2)

plt.xlabel('Displacement [mm]', fontsize=14, fontname='Times New Roman')
plt.ylabel('Reaction Force [kN]', fontsize=14, fontname='Times New Roman')
plt.title('Reaction Force vs Displacement', fontsize=16, fontname='Times New Roman', pad=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.ylim(-5, 5)
plt.xticks(fontsize=12, fontname='Times New Roman')
plt.yticks(fontsize=12, fontname='Times New Roman')
plt.tight_layout()

# fig_path = results_folder / "CA2_column_sinusoidal_force_displacement.png"
# plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()
# print(f"✅ Hysteresis curve saved to:{fig_path}")
