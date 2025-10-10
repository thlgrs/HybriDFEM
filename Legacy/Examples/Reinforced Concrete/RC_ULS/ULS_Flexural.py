# %% -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""
import numpy as np
import os
import h5py
import sys
import pathlib
import importlib

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as ct
import Surface as surf
import ContactPair as cp
import Material as mat

N1 = np.array([0, 0], dtype=float)
N2 = np.array([.1, 0], dtype=float)

E_c = 35e9
FCD = 22.67e6
E_s = 200e9
FYD = 435e6
NU = 0.
# RHO = 3000

CPS = 200
BLOCKS = 2

CONCR = mat.concrete_EC(E_c, FCD, fail_crit=True)
# CONCR.plot_stress_strain()
STEEL = mat.steel_EC(E_s, FYD, fail_crit=True)
# STEEL.plot_stress_strain()

H = .5
B = .2
D = 20e-3
A = 2 * np.pi * D ** 2 / 4
r = D / 2

St = st.Structure_2D()
St.add_beam(N1, N2, BLOCKS, H, 2500, B, material=CONCR)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

for cf in St.list_cfs:
    # cf.add_reinforcement(-.8, A, material=STEEL, height=D)
    cf.add_reinforcement([.8], A, material=STEEL, height=r)

St.save_structure('Undef_structure')

Ns = np.linspace(-2600, 400, 50) * 1e3
# print(Ns)
M = 200e3
import pickle
import pandas as pd
import os

for N in Ns:

    with open(f'Undef_structure.pkl', 'rb') as file:
        St = pickle.load(file)

    St.fixNode(0, [0, 1, 2])
    St.loadNode(1, 0, N, fixed=True)
    St.loadNode(1, 2, M)

    St.solve_dispcontrol(10000, -2e-2, 1, 2, tol=1)

    file = 'Results_DispControl.h5'

    with h5py.File(file, 'r') as hf:

        M_l = hf['P_r_conv'][5] / 1000

    file_path = "data_neg.xlsx"
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=["A", "B"])

    # %%
    nonzero = np.nonzero(M_l)
    M_l = M_l[nonzero]
    print(M_l)
    # df.loc[len(df)] = [-N/1000, M_l[-1]]
    if len(M_l) > 2:
        df.loc[len(df)] = [-N / 1000, M_l[-1]]
        df.to_excel(file_path, index=False)

# St.plot_stresses()

# %% Plotting

# import matplotlib.pyplot as plt

# file = 'Results_DispControl.h5'
# with h5py.File(file, 'r') as hf:

#         #Import what you need
#         print(file)
#         U_r = -hf['U_conv'][5]*1000
#         U_d =  -hf['U_conv'][3]*1000
#         N_l = -hf['P_r_conv'][3]/1000
#         M_l = -hf['P_r_conv'][5]/1000

# plt.figure(None, figsize=(6,6))
# plt.xlim(0,50)
# plt.ylim(0,90)
# if N_fixed:
#     plt.plot(U_r,M_l)
# else: 
#     plt.plot(U_d,N_l)
# plt.xlabel('Beam deflection [mm]')
# plt.ylabel('Applied force [kN]')
# plt.grid(True)

# import pandas as pd
# import os


# if N_fixed: 
#     N_app = N/1e3
#     M_app = max(M_l)
# else:
#     N_app = max(N_l*np.sign(N))*np.sign(N)
#     M_app = M/1e3
#     print(N_app)
#     print(M_app)


# # Load or create the DataFrame
# if os.path.exists(file_path):
#     df = pd.read_excel(file_path)
# else:
#     df = pd.DataFrame(columns=["A", "B"])

# # Find if either number1 or number2 already exists
# mask = (df["A"] == N_app) | (df["B"] == M_app)

# if mask.any():
#     # Update first matching row
#     idx = df[mask].index[0]
#     df.loc[idx] = [N_app, M_app]
# else:
#     # Append new row
#     df.loc[len(df)] = [N_app, M_app]

# # Save back to Excel
# df.to_excel(file_path, index=False)

# plt.plot(U, P)
