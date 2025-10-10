# %% Libraries imports

import matplotlib as mpl

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 15

import h5py
import os
import sys
import pathlib
import numpy as np
import pickle
import importlib

# Folder to access the HybriDFEM files
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

files = []

for file_name in os.listdir():

    if file_name.endswith('.h5'):
        files.append(file_name)

for i, file in enumerate(files):
    with h5py.File(file, 'r') as hf:
        # Import what you need
        print(file)
        U = hf['U_conv'][-3] * 1000
        P = hf['P_r_conv'][-3] / (0.175 * 0.4 * 1000 * 9.81)

    # plt.plot(Time, U, label='Newmark', linewidth=.75, color='black',linestyle='dashed')

# %% Force - displacement curve

plt.figure(None, figsize=(6, 6), dpi=600)

# plt.xlim([0, 300.])
# plt.ylim([0, 1.])
plt.ylabel(r'Load multiplier $\alpha$ [-]')
plt.xlabel(r'Horizontal displacement [mm]')
# plt.legend(fontsize=12)

plt.plot(U, P, linewidth=1, color='black')

plt.grid(True)
plt.savefig('Corner_disp.eps')

# %%Zoom corner displacement
