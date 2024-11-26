# %% Libraries imports
import matplotlib as mpl

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

import matplotlib.pyplot as plt

import h5py
import os
import sys
import pathlib

# Folder to access the HybriDFEM files
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

plt.figure(figsize=(5, 5), dpi=800)

plt.xlim([0, 10])
# plt.ylim([-250, 250])
plt.title(r'Crane in free vibration')
plt.xlabel(r'Time [s]')
plt.ylabel(r'Vertical displacement of free end [mm]')

# styles = ['dashed', 'dotted', 'dashdot', ':']

files = []

for file_name in os.listdir():

    if file_name.endswith('.h5'):
        files.append(file_name)

for i, file in enumerate(files):
    with h5py.File(file, 'r') as hf:
        # Import what you need
        U = hf['U_conv'][-2] * 1000
        Time = hf['Time'][:]

    plt.plot(Time, U, label=file[:-3], linewidth=1)

plt.legend()
plt.grid(True)
