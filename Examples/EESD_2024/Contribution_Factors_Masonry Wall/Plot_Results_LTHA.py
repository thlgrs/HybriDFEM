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
import numpy as np
import pickle

# Folder to access the HybriDFEM files
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

plt.figure(1, figsize=(5, 5), dpi=1200)

lim = 10
plt.xlim([0, lim])
# plt.ylim([-250, 250])
# plt.title(r'Wall with sinusoidal excitation')
plt.xlabel(r'Time [s]')
plt.ylabel(r'Horizontal displacement of top right brick [mm]')

# styles = ['dashed', 'dotted', 'dashdot', ':']

files = []

for file_name in os.listdir():

    if file_name.endswith('0.25.h5'):
        files.append(file_name)

for i, file in enumerate(files):
    with h5py.File(file, 'r') as hf:
        # Import what you need
        print(file)
        U = hf['U_conv'][-3] * 1000
        Time = hf['Time'][:]

    # plt.plot(Time, U, label='Newmark', linewidth=.75, color='black',linestyle='dashed')

list_freqs = [14.038, 28.253, 52.188, 66.357, 68.295, 83.283, 98.91, \
              104.094, 106.06, 108.989, 112.439, 127.301, 128.586, 136.657, 138.642, 149.877]
list_r_U = np.array([100.775]) * 1e-2
#  , 1.119, -2.59]) * 1e-2
# , 0.281, 0.121, 0.321, 0.311]) * 1e-2
list_r_V = np.array([58.197, 26.353, 3.833, 3.528, 1.308, 0.878, 0.455, 0.67]) * 1e-2

w_s = 20
xi = [0.05, 0.054, 0.08, 0.097, 0.1, 0.119, 0.14, \
      0.147, 0.149, 0.153, 0.158, 0.178, 0.18, 0.19, 0.193, 0.208]

U_ref = 5.8565e-3
V_ref = - 70000
l_0 = 0.2 * 9.81


def get_response(w_s, list_xis, r_ref, l_0, list_freqs, list_r, t):
    def response(w, w_s, xi, t):
        r = w_s / w
        w_d = w * np.sqrt(1 - xi ** 2)
        C = (1 - r ** 2) / (((1 - r ** 2) ** 2) + (2 * xi * r) ** 2)
        D = (-2 * xi * r) / (((1 - r ** 2) ** 2) + (2 * xi * r) ** 2)
        A = -D
        B = A * xi * w / w_d - C * w_s / w_d
        return np.e ** (-xi * w * t) * (A * np.cos(w_d * t) + B * np.sin(w_d * t)) + C * np.sin(w_s * t) + D * np.cos(
            w_s * t)

    R = np.zeros(len(t))

    for i in range(len(list_r)):
        r_t = response(list_freqs[i], w_s, list_xis[i], t)
        R += l_0 * r_ref * list_r[i] * r_t

    return R


def get_amplitude_response(w_s, list_xis, r_ref, l_0, list_freqs, list_r):
    def response_factor(r, xi): return np.sqrt(1 / (((1 - r ** 2) ** 2) + (2 * xi * r) ** 2))

    Amp = 0

    for i in range(len(list_r)):
        ratio = w_s / list_freqs[i]
        xi = list_xis[i]
        print(response_factor(ratio, xi))
        Amp += l_0 * r_ref * list_r[i] * response_factor(ratio, xi)

    return Amp


Amp = get_amplitude_response(w_s, xi, U_ref, l_0, list_freqs, list_r_U)
Resp = get_response(w_s, xi, U_ref, l_0, list_freqs, list_r_U, Time)

max_U = max(abs(U))
diff_U = abs(U - Resp * 1000)
err_max_U = max(diff_U) / max_U

print(f'Max. Error in corner displacement: {err_max_U * 100}%')

plt.plot(Time, Resp * 1000, label='Modal superp.', linewidth=1, color='grey')
plt.plot(Time, U, label='Newmark', linewidth=.75, color='black', linestyle='dotted')
plt.legend()
plt.grid(True)
plt.savefig('Corner_disp.eps')

# %% Base shear

plt.figure(2, figsize=(5, 5), dpi=1200)

plt.xlim([0, lim])
# plt.title(r'Wall with sinusoidal excitation')
plt.xlabel(r'Time [s]')
plt.ylabel(r'Base Shear [kN]')

with open(f'F&TL_Wall1.pkl', 'rb') as file:
    St = pickle.load(file)

for i, file in enumerate(files):

    with h5py.File(file, 'r') as hf:

        # Import what you need
        U_conv = hf['U_conv'][:]
        Time = hf['Time'][:]

    P_base = np.zeros(len(Time))
    # St.get_K_str()

    for i in range(len(Time)):
        St.U = U_conv[:, i]
        # St.get_P_r()
        P_r = St.K[np.ix_(St.dof_fix, St.dof_free)] @ St.U[St.dof_free]
        P_base[i] = P_r[0]
        # P_base[i] = np.sum(P_r)

    # plt.plot(Time, P_base, label='Newmark', linewidth=.75, color='black',linestyle='dotted')

Amp = get_amplitude_response(w_s, xi, V_ref, l_0, list_freqs, list_r_V)
Resp = get_response(w_s, xi, V_ref, l_0, list_freqs, list_r_V, Time)

plt.plot(Time, Resp / 1000, label='Modal superp.', linewidth=1, color='grey')
plt.plot(Time, P_base / 1000, label='Newmark', linewidth=.75, color='black', linestyle='dotted')

max_V = max(abs(P_base[8000:]))
diff_V = max(abs(P_base[8000:])) - max(abs(Resp[8000:]))
err_max_V = diff_V / max_V

print(f'Max. Error in base shear: {err_max_V * 100}%')

plt.legend()
plt.grid(True)

plt.savefig('Base_shear.eps')
