# %% Libraries imports

import matplotlib as mpl

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 15

import pandas as pd
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

files = ['Response_NF13_NWK_g=0.5_b=0.25.h5',
         'Response_NF13_SSI_NWK_g=0.5_b=0.25.h5']
labels = ['Fixed base', 'Rocking base']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, ax = plt.subplots(2, 1, dpi=500, figsize=(8, 8))

ax[1].set_xlabel(r'Time [s]')
ax[1].set_ylabel(r'$\Delta_{oop}$ [mm]')

ax[0].set_ylabel(r'Ground acceleration [g]')

for i, file_name in enumerate(files):
    file_to_open = file_name

    with h5py.File(file_to_open, 'r') as hf:
        last_conv = hf['Last_conv'][()]
        U_top = hf['U_conv'][0, :last_conv]
        U_oop = hf['U_conv'][7 * 3, :last_conv]
        U_bot = hf['U_conv'][14 * 3, :last_conv]

        Time = hf['Time'][:last_conv]

    d_oop = U_oop - (U_top + U_bot) / 2

    ax[1].plot(Time, d_oop * 1000, linewidth=1, color=colors[i], label=labels[i])


def read_accelerogram(filename):
    df = pd.read_csv(filename, sep='\s+', header=1)
    values = df.to_numpy()

    a = values[:, :6]
    a = a.reshape(-1, 1)
    a = a[~np.isnan(a)]

    file = open(filename)
    # get the first line of the file
    line1 = file.readline()
    line2 = file.readline()
    items = line2.split(' ')
    items = np.asarray(items)
    items = items[items != '']
    dt = float(items[1])
    t = np.arange(len(a)) * dt

    return (t, a)


time, acc = read_accelerogram('Earthquakes/NF13')
pga = 1
acc = pga * acc / np.max(abs(acc))
acc = np.append(acc, np.zeros(10))
time = np.append(time, np.linspace(15, 20, 10))
ax[0].plot(time, acc, linewidth=1, color='black')

ax[1].grid(True, linestyle='--', linewidth=0.5)
ax[0].grid(True, linestyle='--', linewidth=0.5)
ax[1].legend()
ax[1].set_xlim((0, 20))
ax[0].set_xlim((0, 20))
plt.ylim((-100, 100))
