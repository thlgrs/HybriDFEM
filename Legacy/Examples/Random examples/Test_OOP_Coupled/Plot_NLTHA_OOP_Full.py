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

w_s = 10

if w_s == 10: Amp = [0.31, 0.3, 0.33, 0.35]
if w_s == 20: Amp = [0.21, 0.22]
if w_s == 30: Amp = [0.5]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, ax1 = plt.subplots(figsize=(7, 6), dpi=600)

ax1.set_xlabel(r'Time [s]')
ax1.set_ylabel(r'Out-of-plane displacement $\Delta_{oop}$ [mm]')
ax1.set_title(f'OOP Displacement with $\omega_s={w_s}$ rad/s')

ax2 = ax1.twinx()
ax2.set_ylabel(r'Ground acceleration $\ddot{u}_g(t)$ [g]')

MIN = 0
MAX = 0

for file_name in os.listdir():

    for i, a in enumerate(Amp):
        if file_name.endswith('.42.h5') and file_name.startswith(f'w={w_s}_a={a}_Full'):
            file_to_open = file_name

            with h5py.File(file_to_open, 'r') as hf:

                last_conv = hf['Last_conv'][()]
                U_top = hf['U_conv'][14 * 3, :last_conv]
                U_oop = hf['U_conv'][7 * 3, :last_conv]
                U_bot = hf['U_conv'][0 * 3, :last_conv]

                Time = hf['Time'][:last_conv]

            d_oop = U_oop - (U_top + U_bot) / 2

            ax1.plot(Time, d_oop * 1000, linewidth=1, color=colors[i], label=f'${a}g$')

            if i == len(Amp) - 1:
                ax1.plot(Time[-1], d_oop[-1] * 1000, '*', color='black', label='Collapse')
            else:
                ax1.plot(Time[-1], d_oop[-1] * 1000, '*', color='black')

            MIN = min(MIN, min(d_oop * 1000))
            MAX = max(MAX, max(d_oop * 1000))

            acc = np.zeros(len(Time))
            lag = 0.1

            for j, t in enumerate(Time):
                if t < lag:
                    acc[j] = 0
                else:
                    acc[j] = a * np.sin(w_s * (t - lag))
            if i == 0:
                ax2.plot(Time, acc, linewidth=.5, color='black', label=r'$\ddot{u}_g(t)$', linestyle='dashed')

ax1.tick_params(axis='y')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.set_xlim((0, 10))

ax2.tick_params(axis='y')

y_lim_ax1 = [MIN * 1.1, MAX * 1.1]
ax1.set_ylim(y_lim_ax1)

# Smallest scaling factor between min and max
scale_factor = min(abs(y_lim_ax1[0] / min(Amp)), abs(y_lim_ax1[1] / max(Amp)))
y_lim_ax2 = y_lim_ax1 / scale_factor
ax2.set_ylim([y_lim_ax2[0] * 1.1, y_lim_ax2[1] * 1.1])
ax1.legend(fontsize=12, loc='best')
ax2.legend(fontsize=12, loc='lower right')
fig.tight_layout()
plt.show()
# %%
fig.tight_layout()
plt.savefig(f'OOP_Displacement_w_s={w_s}.png', dpi=1200)
# %%
