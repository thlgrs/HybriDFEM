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

plt.figure(figsize=(4.5, 4.5), dpi=800)

obj = np.array([-7.14285714e-03, -1.46196429e-02, -7.23214286e-03])
sol = np.array([[1.95139418e-02, 4.29737860e-02, 1.95769347e-02],
                [-6.12574119e-02, -3.88708575e-02, -6.34674687e-02],
                [-8.04794689e-02, -3.35899506e-02, -9.43536219e-02]])

q = np.linalg.solve(sol.T, obj)

print(q)

phi1 = sol.T[1, 0]
phi2 = sol.T[1, 1]
phi3 = sol.T[1, 2]

w_1 = 114.0289
w_2 = 649.266
w_3 = 1622.00

x = np.linspace(0, 0.5, 1000)
y = q[0] * phi1 * np.cos(w_1 * x)
plt.plot(x, y * 1000, label='Mode 1', linewidth=.75, color='red')

y = q[1] * phi2 * np.cos(w_2 * x)
plt.plot(x, y * 1000, label='Mode 2', linewidth=.75, color='blue')

y = q[2] * phi3 * np.cos(w_3 * x)
plt.plot(x, y * 1000, label='Mode 3', linewidth=.75, color='green')

y = q[0] * phi1 * np.cos(w_1 * x) + q[1] * phi2 * np.cos(w_2 * x) + q[2] * phi3 * np.cos(w_3 * x)

plt.plot(x, y * 1000, label='Modal sup.', linewidth=.75, color='black', linestyle='--')

plt.legend(fontsize=10)
plt.grid(linewidth=.25)
plt.xlim((0, .5))
plt.ylim((-15, 15))
plt.ylabel('Deflection [mm]', fontsize=15)
plt.xlabel('Time [s]', fontsize=15)
plt.yticks([-14.4, 14.4], [r'$-\Delta_0$', r'$\Delta_0$'], fontsize=14)

plt.savefig('modal_sup.eps')
