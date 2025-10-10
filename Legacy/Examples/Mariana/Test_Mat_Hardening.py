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

# === Geometry and material
h = 0.02  # height [m]
L = 0.02  # width [m]
t = 0.01  # thickness [m]
rho = 7800  # density [kg/m³]

E = 200e9  # Young's modulus [Pa]
NU = 0.3
FY = 580e6  # Yield stress [Pa]
H = 15.4e9  # Hardening modulus [Pa]
r = 0.025  # Hardening ratio

G = E / (2 * (1 + NU))
kn = E / (t / 2)
ks = G / (t / 2)

# === Create nonlinear contact material
material = mat.Mixed_Hardening_Mat(E=E, nu=NU, fy=FY, H=H, r=r)

# material = mat.Bilinear_Mat(E, NU, FY, 0.1)

# === Create structure
S = st.Structure_2D()

# Add left block (fixed)
verts1 = np.array([[0, 0], [L, 0], [L, h], [0, h]])
S.add_block(vertices=verts1, rho=rho, b=t, material=material)

# Add right block (loaded)
verts2 = np.array([[L, 0], [2 * L, 0], [2 * L, h], [L, h]])
S.add_block(vertices=verts2, rho=rho, b=t, material=material)

# Setup mesh and nonlinear contact (one contact pair)
S.make_nodes()

S.make_cfs(lin_geom=True, nb_cps=1)

# === Apply boundary conditions
S.fixNode(0, [0, 1, 2])  # Fix all DOFs at node 0
S.fixNode(1, [1, 2])

# === Cyclic loading parameters
n_steps = 3
amplitude = 140e3  # N (adjust as needed to exceed yield)
load_hist = amplitude * np.sin(np.linspace(0, 4 * np.pi, n_steps + 1))
load_hist = list(load_hist)  # Convert to list!

plt.plot(np.arange(len(load_hist)), load_hist)

# === Apply load at node 1 in x-direction (will be scaled by load_hist)
# S.reset_loading()
S.loadNode(1, [0], amplitude)  # The actual load at each step is amplitude * load_hist[i]/amplitude = load_hist[i]

# === Plot undeformed structure before solving
S.plot_structure(
    scale=0,  # No deformation
    plot_cf=True,  # Show contact pairs
    plot_forces=True,  # Show applied forces
    plot_supp=True  # Show supports
)
plt.title("Undeformed Structure: 2 Blocks + 1 Contact Pair")
plt.show()

# === Solve with force control using the cyclic load history
filename = "CA2_Column_CyclicForceControl"
S.solve_forcecontrol(steps=n_steps, max_iter=100, dir_name=save_path, filename=filename, tol=1e-3)
S.save_structure(filename)

# === Extract and plot results
results_file = os.path.join(filename + ".h5")
with h5py.File(results_file, 'r') as hf:
    # Displacement and reaction force at loaded node (node 1, x-direction)
    U_all = hf['U_conv'][-3, :]  # x-displacement at node 1
    F_all = hf['P_r_conv'][-3, :]  # x-reaction at node 1

# === Plot force–displacement (hysteresis)
plt.figure()
plt.plot(U_all * 1e3, F_all / 1e3, 'o-', label='Node 1 (contact spring)', markersize=3)
plt.xlabel('Displacement [mm]')
plt.ylabel('Reaction Force [kN]')
plt.title('Cyclic Force–Displacement (Nonlinear Contact Spring)')
plt.grid(True)
plt.legend()

# === Save figure
# fig_path = results_folder / "CA2_column_cyclic_force_displacement.png"
# plt.savefig(fig_path, dpi=300)
# print(f"✅ Figure saved to: {fig_path}")

plt.show()
#
