# -*- coding: utf-8 -*-
"""
HybridFEM: Create and plot geometry of W10-2 wall (side profile) with a impactor block
Coded by Mariana Castro
"""

import numpy as np
import sys
import pathlib
import os
import matplotlib.pyplot as plt

# Set path to HybridFEM object definitions
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Surface as surf

# Wall geometry and material properties
H_total = 1.535
L_total = 0.245
depth = 0.775  # Depth of the wall (thickness)
n_rows = 21
h_block = H_total / n_rows
l_block = L_total
b = 0.062
rho = 1715  # [kg/m³]...because I am using a 3D wall with depth

pattern = []
for i in range(n_rows):
    pattern.append([1.0] if i % 2 == 0 else [0.5, 0.5])

S = st.Structure_2D()
corner = np.array([0.0, 0.0])
S.add_wall(corner, l_block, h_block, pattern, rho, b=depth, material=None)

# Contact law
kn_total = 90e9 * 2  # Normal stiffness
ks_total = 90e9 * 2  # Shear stiffness
mu = 0.8  # Friction coefficient
n_cp = 2  # Number of contact pairs

contact = surf.Coulomb(kn_total, ks_total, mu)
# contact = surf.NoTension_CD(kn=kn_total, ks=ks_total)


# No-tension contact properties (impactor interfaces)
knn = 10e8
kss = 10e8

# Impactor block 
row_index = 10
x0 = -l_block
y0 = row_index * h_block
impactor_vertices = np.array([
    [x0, y0],
    [x0 + l_block, y0],
    [x0 + l_block, y0 + h_block],
    [x0, y0 + h_block]
])

V_impactor = l_block * h_block * 1
rho_impactor = 116 / (depth * V_impactor)
print(f"impactor density set to: {rho_impactor:.2f} kg/m³")

S.add_block(impactor_vertices, rho=rho_impactor, b=depth, material=None)

# Add non-rigid TOP support (Timoshenko FE)
K_support = 31.4e6  # Stiffness of the support
nu = 0.3
h_spring = 0.01
b_spring = depth

# Convert total stiffness [N/m] to Young's modulus E [Pa]
# using: E = K * L / A = K_support * h_spring / (b * depth)
E_support = K_support / b_spring  # Young's modulus for the support

top_block = S.list_blocks[-2]
N1 = top_block.ref_point  # Nodo del bloque superior
N2 = N1 + np.array([0, h_spring])  # Nodo dummy por encima

S.list_nodes.append(N2)  #

S.add_fe(N1, N2, E=E_support, nu=nu, h=h_spring, b=b_spring,
         lin_geom=False, rho=1000)  # FE entre el nodo top y el dummy

S.make_nodes()
S.make_cfs(lin_geom=False, nb_cps=[-1, 1], surface=contact)

# Add non-tension contact to the impactor block
impactor_block = S.list_blocks[-1]
no_tension = surf.NoTension_CD(kn=knn, ks=kss)
for CF in S.list_cfs:
    if CF.bl_A == impactor_block or CF.bl_B == impactor_block:
        CF.change_cps(nb_cp=n_cp, offset=-1, surface=no_tension)

S.Udot = np.zeros(S.nb_dofs)
# Apply boundary conditions
n1_index = S.get_node_id(N1)
n2_index = len(S.list_nodes) - 1  # Último nodo agregado (dummy arriba)

S.fixNode(n2_index, [0, 1, 2])  # Nodo dummy totalmente fijo
S.fixNode(0, [0, 1, 2])  # Nodo base (primer bloque) totalmente fijo

# Fix top block and apply axial load
top_block_id = len(S.list_blocks) - 2
S.fixNode(top_block_id, [0, 2])

# Apply axial load to the top block
sigma_axial = 0.05e6
A_top = depth * L_total
F_axial = sigma_axial * A_top + 357 * 9.81
S.loadNode(top_block_id, 1, -F_axial, fixed=True)  # Apply axial load in Y direction
print("axial", F_axial)

# Add the self-weight load to the blocks
g = 9.81  # gravty [m/s²]

for i in range(len(S.list_blocks) - 1):  # not the impactor!
    block = S.list_blocks[i]
    W = block.m * g  # weight total [N]

    # DOF 1 = Uy
    S.loadNode(i, 1, -W, fixed=True)

# Assign initial velocity to impactor
dofs_impactor = impactor_block.dofs
S.fixNode(len(S.list_blocks) - 1, [1, 2])  # Fix the impactor in Y and Z directions
v0 = 4.43
S.Udot[dofs_impactor[0]] = v0
S.U[dofs_impactor[0]] = -0.001  # small initial displacement

# Plot undeformed structure
S.plot_structure(scale=0, plot_cf=True, plot_forces=True, plot_supp=True)

S.solve_forcecontrol(10)
# Mass diagnostics
S.get_M_str()

# --- SOLVER ---
T = 0.05
dt = 1e-5
S.set_damping_properties(xsi=0.02, damp_type='STIFF', stiff_type='TAN')
S.plot_structure(scale=100, plot_cf=False, plot_forces=False, plot_supp=True)

# Solve the dynamic problem using the CDM method
S.solve_dyn_nonlinear(
    T=T,
    dt=dt,
    V0=S.Udot,
    lmbda=None,
    Meth='CDM',
    filename='W11_1_Impact',
    # dir_name=r"C:\1.MARIANA\4. Research HybridFEM\HybriDFEM - Mariana - Research Intership\HybriDFEM---Mariana\HybriDFEM-3\Mariana_Examples\Impact loads - masonry wall\PAPER 1"
)

# Save for animation moviepy
S.save_structure(filename='Structure')
