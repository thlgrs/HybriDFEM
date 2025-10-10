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

fig, ax1 = plt.subplots(figsize=(7, 6), dpi=600)

ax1.set_xlabel(r'Control displacement [mm]')
ax1.set_ylabel(r'Load multiplier $\alpha$ [-]')
ax1.set_title(f'Pushover curve of the mixed frame')

for file_name in os.listdir():

    if file_name.endswith('DispControl.h5'):
        with h5py.File(file_name, 'r') as hf:
            U = hf['U_conv'][0 * 3]
            P = hf['P_r_conv'][0] / (128.57)

ax1.plot(U * 1000, P, linewidth=1, color='black', marker='o')
ax1.tick_params(axis='y')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
# ax1.set_xlim((0,100))

ax1.legend(fontsize=12, loc='best')

fig.tight_layout()
plt.show()
# %%
fig.tight_layout()
plt.savefig(f'OOP_Pushover.png', dpi=1200)
# %%
