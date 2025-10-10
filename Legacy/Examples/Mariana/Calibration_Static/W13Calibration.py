# -*- coding: utf-8 -*-
"""
# HybridFEM: W13 wall with prestress and four-point bending test
# Step 1: Static self-weight + axial prestress using Timoshenko FE
# Step 2: Displacement-controlled loading at two nodes
"""

import numpy as np
import sys
import pathlib
import os
import matplotlib.pyplot as plt
import pickle

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Surface as surf

# === Geometry and Material ===
H_total = 1.535  # Total wall height [m]
L_total = 0.245  # Wall width [m]
depth = 0.775  # Wall thickness [m]
n_rows = 21
h_block = H_total / n_rows
l_block = L_total
b = 0.062  # Block thickness (used for stiffness conversion)
rho = 1715  # Block density [kg/m³]
g = 9.81  # Gravity [m/s²]

# Masonry bond pattern: full block or two half-blocks per row
pattern = [[1.0] if i % 2 == 0 else [0.5, 0.5] for i in range(n_rows)]

# Create structure and add wall
S = st.Structure_2D()
corner = np.array([0.0, 0.0])
S.add_wall(corner, l_block, h_block, pattern, rho, depth)

# Add Timoshenko spring as finite element at the top
K_support = 19e6  # [N/m]
nu = 0.3
h_spring = .1
E_support = K_support / b
top_block = S.list_blocks[-1]
N1 = top_block.ref_point
N2 = N1 + np.array([0, h_spring])
S.add_fe(N1, N2, E=E_support, nu=nu, h=h_spring, b=b, lin_geom=True, rho=1000)

# Define contact law
kn_total = 1.2e9 * 2
ks_total = 0.4e9 * 2
mu = 0.85
contact = surf.Coulomb(kn_total, ks_total, mu, psi=.0)

# Finalize contact and mesh
S.make_nodes()
S.make_cfs(lin_geom=False, nb_cps=[-1, 1], surface=contact)

# Fix nodes
n1_index = S.get_node_id(N1)
n2_index = S.get_node_id(N2)
S.fixNode(0, [0, 1, 2])  # Bottom node
S.fixNode(n1_index, [0, 2])  # Top block Ux and rotation
S.fixNode(n2_index, [0, 2])  # Dummy node Ux and rotation

# Apply self-weight
for i in range(len(S.list_blocks[:-1])):
    W = S.list_blocks[i].m * g
    S.loadNode(i, 1, -W, fixed=True)

# Apply prestress force to dummy node
sigma_axial = 0.2e6
A_top = depth * L_total
F_axial = sigma_axial * A_top + (38 + 95) * g
S.loadNode(n2_index, 1, -F_axial, fixed=True)

# Plot deformed structure
S.plot_structure(scale=100, plot_cf=False, plot_forces=True, plot_supp=True)

# --- Step 1: Solve statically (self-weight + prestress)
print("Step 1: Solving prestress and self-weight...")
S.solve_forcecontrol(
    steps=25,
    max_iter=30,
    filename='W13_After_Prestress'
)

# STEP 2: SOLVE THE DISPLACEMENT-CONTROLLED LOADING AT TWO NODES

S.fixNode(n2_index, [1])  # Lock vertical DOF to simulate prestress lock
F = 100  # Whatever the value
S.loadNode(9, 0, F)
S.loadNode(21, 0, F)
# S.loadNode(15, 2, F*0.01 ) # Otherwise too symetric, and unrealsisitc
# Solve displacement control
S.solve_dispcontrol(
    steps=1000,
    disp=40e-3,
    node=15,  # Node ID for displacement control
    dof=0,  # 0 = Ux
    tol=1,
    max_iter=500,
    filename="W13_4Point_Test")

# %% Plotting results
S.plot_structure(scale=100, plot_cf=False, plot_forces=False, plot_supp=False)

import h5py

file = f'W13_4Point_Test.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U = hf['U_conv'][3 * 15] * 1000
    P = hf['P_r_conv'][3 * 9] / 1000

plt.plot(U, P)
