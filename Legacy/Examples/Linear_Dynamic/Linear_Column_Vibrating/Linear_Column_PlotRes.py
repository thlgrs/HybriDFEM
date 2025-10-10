# %% Libraries imports
import matplotlib as mpl

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

import matplotlib.pyplot as plt

import pickle
import h5py
import os
import sys
import pathlib
import numpy as np

# Folder to access the HybriDFEM files
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

plt.figure(figsize=(5, 5), dpi=800)

# plt.xlim([0, 5])
# plt.ylim([-1.2, 1.2])
plt.title(r'Cantilever beam in free vibration')
plt.xlabel(r'Time [s]')
plt.ylabel(r'Vertical displacement of free end [mm]')

styles = ['dashed', 'dotted', 'dashdot', ':']

files = []

for file_name in os.listdir():

    if file_name.endswith('.h5'):
        files.append(file_name)

for i, file in enumerate(files):
    with h5py.File(file, 'r') as hf:
        # Import what you need
        U = hf['U_conv'][-2] * 1000
        Time = hf['Time'][:]

    plt.plot(Time, U, label=None, linewidth=1, linestyle=styles[i])

plt.legend()
plt.grid(True)
