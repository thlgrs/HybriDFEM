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
from scipy.optimize import curve_fit

# Folder to access the HybriDFEM files
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

plt.figure(figsize=(4.5, 4.5), dpi=600)

# plt.xlim([0, 5])
# plt.ylim([-1.2, 1.2])
# plt.title(r'Cantilever beam in free vibration')
plt.xlabel(r'Normalized time $t / T_1$')
plt.ylabel(r'Normalized deflection [-]')

styles = []

files = []
methods = []
colors = []

meth = 'CAA'
# meth= 'LA'
# meth='CDM'
dir_name = meth + '_conv_blocks'

styles = [':', '--', '-.', (0, (1, 2, 1, 2, 4, 2))]
colors = ['b', 'r', 'g', 'orange']

# frequencies = [111.715, 114.232, 114.719, 114.838]
# amplitudes = [-14.3865, -14.5673, -14.6017, -14.61]
frequencies = [80.597, 111.715, 114.719, 114.838]
amplitudes = [-11, -14.3865, -14.6017, -14.61]

files = [2, 5, 20, 50]
if meth == 'CDM': files = [2, 5, 20]

for i, file in enumerate(files):

    if meth == 'CAA': filename = f'{file}' + '_NWK_g=0.5_b=0.25.h5'
    if meth == 'LA': filename = f'{file}' + '_NWK_g=0.5_b=0.17.h5'
    if meth == 'CDM': filename = f'{file}' + '_CDM_.h5'

    with h5py.File(dir_name + '/' + filename, 'r') as hf:

        # Import what you need
        U = hf['U_conv'][-2] * 1000
        Time = hf['Time'][:]
    period = 2 * np.pi / frequencies[i]


    def model(x, A, B, C):
        return A * np.exp(-B * x) * np.cos(C * x)


    params, _ = curve_fit(model, Time, U, p0=[1, 1, 1])

    A, B, C = params
    print(f"U(t) = {A} e^(-{B}*t) * cos({C}t)  for {file} blocks")


    def damping_ratio(delta):
        return delta / np.sqrt(4 * np.pi ** 2 + delta ** 2)


    def logarithmic_decrement(A, B):
        return 2 * np.pi * A / B


    delta = logarithmic_decrement(B, C)
    xi = damping_ratio(delta)

    print(f'Damping ratio is {xi}')

    period = 1
    x = np.linspace(0, 0.5, 1000)
    y = model(x, A, B, C)
    plt.plot(x, y, linewidth=2)
    plt.plot(Time, U, linewidth=.75, linestyle=styles[i], label=f'{file} Blocks', color=colors[i])

obj = np.array([-1.46102641e-02, -7.20588235e-03])
sol = np.array([[0.04320525, -0.04056667],
                [0.01953577, -0.06290884]])

q = np.linalg.solve(sol, obj)

q01 = q[0]
q02 = q[1]
phi1 = sol[0, 0]
phi2 = sol[0, 1]
w_1 = 114.838
w_2 = 660.979
xi = 0.02
w_d1 = w_1 * np.sqrt(1 - xi ** 2)
w_d2 = w_2 * np.sqrt(1 - xi ** 2)
x = np.linspace(0, 0.5, 1000)
y = np.exp(-xi * w_1 * x) * q01 * phi1 * np.cos(w_d1 * x) \
    + np.exp(-xi * w_2 * x) * q02 * phi2 * np.cos(w_d2 * x)

plt.plot(x, y * 1000, label='Modal sup.', linewidth=.75, color='black')

plt.legend(fontsize=8)
plt.grid(linewidth=.25)
plt.xlim((0, .5))
# plt.ylim((-1,1))
plt.gca().set_yticklabels([])

plt.savefig(dir_name + '_comp.eps')
