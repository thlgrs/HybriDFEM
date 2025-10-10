# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 21:12:20 2025

@author: ibouckaert
"""

import matplotlib.pyplot as plt
import h5py
import pandas as pd
import sys, math, copy, os
import numpy as np
import pathlib

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "serif",  # Use a serif font
    "font.size": 15  # Adjust font size
})

r_b = 0.02
psi = 0.65
W = 700

import pickle

with open(f'F&TL_Wall1.pkl', 'rb') as file:
    St = pickle.load(file)

file = f'Wall1_rb={r_b}_psi={psi}.h5'

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_final = hf['U_conv'][:, last_conv]

St.U = U_final
St.plot_structure(scale=1, plot_cf=False, plot_forces=False, plot_supp=False, save='Mech1.png')

# %% HybriDFEM results
file = f'Wall1_rb={r_b}_psi={psi}.h5'

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_h = hf['U_conv'][-3, :last_conv] * 1000
    P_h = hf['P_r_conv'][-3, :last_conv] / W

# %% LMGC90 Results
h = 0.175
b = 0.4
rho = 1019.
load = 9.81 * b * h * rho
total_load = (6 * 5) * load

outputfile = 'Plotting/Disps3.DAT'
df = pd.read_csv(outputfile, sep='\s+', header=None)
values = df.values
t_num = values[:, 0]
delta = values[:, 1]
delta_0 = values[0, 1]
rotation = values[:, 3]

U_l = (delta - delta_0) - (h / 2) * np.sin(rotation) - (b / 2) * (1 - np.cos(rotation))

# Calculate applied lambda_b
outputfile = 'Plotting/Forces3.DAT'
df = pd.read_csv(outputfile, sep='\s+', header=None)
values = df.values
P_l = values[:, 4] / (load)

# Clean first 10sec
for i in range(len(P_l)):
    if P_l[i] > 0:
        index = i
        break

P_l = P_l[index:]
U_l = U_l[index:]
U_l -= U_l[0]

# Import UDEC
inputfile = 'Plotting/Disp_UDEC.DAT'
df = pd.read_csv(inputfile, sep='\s+', header=None)

values = df.values
U_u = values[7:, 1]
U_u -= U_u[0]

inputfile = 'Plotting/lambdab_UDEC.DAT'
df = pd.read_csv(inputfile, sep='\s+', header=None)

values = df.values
P_u = values[7:, 1]

# %% Plotting

plt.figure(None, dpi=400, figsize=(6, 6))
plt.xlabel('Control displacement [mm]')
plt.ylabel('Load multiplier')

plt.plot(U_h, P_h, linewidth=1, color='red', marker=None, label='HybriDFEM')
plt.plot(U_l * 1000, P_l, linewidth=1, color='blue', marker=None, label='LMGC90')
plt.plot(U_u * 1000, P_u, linewidth=1, color='green', marker=None, label='UDEC')

plt.xlabel('Control displacement [mm]')
plt.ylabel('Load multiplier [-]')

plt.xlim((0, 200))
plt.ylim((0, 1))
plt.grid(True)
plt.legend()

# Load multipliers
