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

    if file_name.endswith('0.25.h5'):
        files.append(file_name)

for i, file in enumerate(files):
    with h5py.File(file, 'r') as hf:
        # Import what you need
        print(file)
        U = hf['U_conv'][-3] * 1000
        Time = hf['Time'][:]

    # plt.plot(Time, U, label='Newmark', linewidth=.75, color='black',linestyle='dashed')

list_freqs = [14.038, 22.462, 28.253, 37.506, 52.188, 53.94, 66.357, 67.355, 68.295,
              70.693, 75.107, 83.283, 85.197, 98.91, 101.412, 104.094, 106.06, 108.989,
              112.14, 112.439, 113.564, 115.702, 124.465, 127.301, 128.586, 132.711, 136.657,
              138.642, 149.649, 149.877, 150.623, 154.278, 154.395, 156.799, 156.931, 157.508,
              158.574, 163.873, 165.35, 169.001, 172.669, 173.777, 178.371, 180.409, 185.052,
              187.444, 188.65, 190.361, 191.272, 198.988, 199.333, 200.874, 201.12, 202.556,
              203.335, 204.759, 204.835, 207.756, 210.148, 213.934, 214.852, 217.298, 218.268,
              222.8, 225.28, 226.155, 229.506, 229.96, 235.386, 235.718, 238.951, 239.683,
              240.245, 242.405, 242.888, 244.141, 245.065, 245.09, 249.981, 250.783, 252.423,
              255.885, 257.636, 258.226, 258.683, 261.197, 263.888, 264.434, 265.793, 266.809,
              270.199, 272.325, 274.75, 274.931, 275.395, 277.933, 281.47, 281.534, 286.02,
              286.838]

list_r_U = np.array([100.775, 0, 1.119, 0, -2.59]) * 1e-2
# , 0.281, 0.121, 0.321, 0.311]) * 1e-2
list_r_V = np.array([5.81968e-01, 0.00000e+00, 2.63527e-01, 0.00000e+00, 3.83270e-02, 0.00000e+00,
                     3.52840e-02, 0.00000e+00, 1.30770e-02, 0.00000e+00, 0.00000e+00, 8.78400e-03,
                     0.00000e+00, 4.55400e-03, 0.00000e+00, 6.69500e-03, 4.14900e-03, 5.79800e-03,
                     0.00000e+00, 3.49700e-03, 0.00000e+00, 0.00000e+00, 0.00000e+00, 8.00000e-05,
                     9.39000e-04, 0.00000e+00, 2.85000e-04, 6.48400e-03, 0.00000e+00, 4.35900e-03,
                     0.00000e+00, 4.90000e-05, 0.00000e+00, 0.00000e+00, 1.31200e-03, 0.00000e+00,
                     0.00000e+00, 0.00000e+00, 1.55000e-04, 4.48000e-04, 2.47300e-03, 0.00000e+00,
                     0.00000e+00, 2.97000e-04, 0.00000e+00, 2.57600e-03, 0.00000e+00, 0.00000e+00,
                     5.75000e-04, 0.00000e+00, 8.89000e-04, 1.19000e-04, 0.00000e+00, 3.17000e-04,
                     1.55000e-04, 0.00000e+00, 9.20000e-05, 0.00000e+00, 7.06000e-04, 0.00000e+00,
                     1.54000e-04, 5.02000e-04, 0.00000e+00, 0.00000e+00, 1.34700e-03, 0.00000e+00,
                     0.00000e+00, 3.00000e-06, 1.77000e-04, 3.22000e-04, 0.00000e+00, 9.12000e-04,
                     0.00000e+00, 0.00000e+00, 1.90000e-04, 0.00000e+00, 0.00000e+00, 8.80000e-05,
                     4.24000e-04, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 6.90000e-05,
                     0.00000e+00, 2.20000e-05, 8.56000e-04, 0.00000e+00, 0.00000e+00, 4.00000e-05,
                     2.93000e-04, 0.00000e+00, 0.00000e+00, 2.22000e-04, 0.00000e+00, 4.60000e-05,
                     0.00000e+00, 5.50000e-04, 0.00000e+00, 2.89000e-04])

w_s = 20
xi = [0.05, 0.05, 0.054, 0.0629, 0.0798, 0.0819, 0.0974, 0.0987, 0.0999, 0.103,
      0.1086, 0.1193, 0.1218, 0.1399, 0.1432, 0.1467, 0.1494, 0.1533, 0.1575, 0.1579,
      0.1594, 0.1622, 0.174, 0.1778, 0.1795, 0.1851, 0.1904, 0.193, 0.2079, 0.2082,
      0.2092, 0.2141, 0.2143, 0.2176, 0.2177, 0.2185, 0.22, 0.2271, 0.2291, 0.2341,
      0.239, 0.2405, 0.2468, 0.2495, 0.2558, 0.2591, 0.2607, 0.263, 0.2643, 0.2748,
      0.2752, 0.2773, 0.2777, 0.2796, 0.2807, 0.2826, 0.2827, 0.2867, 0.2899, 0.2951,
      0.2963, 0.2997, 0.301, 0.3071, 0.3105, 0.3117, 0.3163, 0.3169, 0.3243, 0.3247,
      0.3291, 0.3301, 0.3309, 0.3338, 0.3345, 0.3362, 0.3375, 0.3375, 0.3442, 0.3453,
      0.3475, 0.3522, 0.3546, 0.3554, 0.356, 0.3595, 0.3631, 0.3639, 0.3657, 0.3671,
      0.3717, 0.3746, 0.3779, 0.3782, 0.3788, 0.3823, 0.3871, 0.3872, 0.3933, 0.3944]

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


# %% Corner displacement
plt.figure(1, figsize=(6, 6), dpi=600)

lim = 10
plt.xlim([0, lim])
plt.xlabel(r'Time [s]')
plt.ylabel(r'Horizontal displacement of top right brick [mm]')
plt.legend(fontsize=16)
Resp1 = get_response(w_s, xi, U_ref, l_0, list_freqs, list_r_U, Time)
Resp2 = get_response(w_s, xi, U_ref, l_0, list_freqs, list_r_U[:1], Time)

max_U = max(abs(U))
diff_U = abs(U - Resp1 * 1000)
err_max_U = max(diff_U) / max_U

print(f'Max. Error in corner displacement: {err_max_U * 100}%')

plt.plot(Time, Resp1 * 1000, label='3 modes', linewidth=1, color='grey')
plt.plot(Time, Resp1 * 1000, label='1 mode', linewidth=.75, color='black', linestyle='dashed')
plt.plot(Time, U, label='Newmark', linewidth=.75, color='black', linestyle='dotted')
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('Corner_disp.eps')

# %%Zoom corner displacement
plt.figure(None, figsize=(6, 6), dpi=600)

plt.xlim([0, 4])
plt.xlabel(r'Time [s]')
plt.ylabel(r'Horizontal displacement of top right brick [mm]')

plt.plot(Time, Resp1 * 1000, label='3 modes', linewidth=1, color='grey')
plt.plot(Time, Resp2 * 1000, label='1 mode', linewidth=.75, color='black', linestyle='dashed')
plt.plot(Time, U, label='Newmark', linewidth=.75, color='black', linestyle='dotted')
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('Corner_disp_zoom.eps')

# %% Base shear

plt.figure(None, figsize=(6, 6))

plt.xlim([0, lim])
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

    for i in range(len(Time)):
        St.U = U_conv[:, i]
        P_r = St.K[np.ix_(St.dof_fix, St.dof_free)] @ St.U[St.dof_free]
        P_base[i] = P_r[0]

Resp1 = get_response(w_s, xi, V_ref, l_0, list_freqs, list_r_V[:16], Time)
Resp2 = get_response(w_s, xi, V_ref, l_0, list_freqs, list_r_V[:3], Time)

plt.plot(Time, Resp1 / 1000, label='8 modes', linewidth=1, color='grey')
plt.plot(Time, Resp2 / 1000, label='3 modes', linewidth=.75, color='black', linestyle='dashed')
plt.plot(Time, P_base / 1000, label='Newmark', linewidth=.5, color='black', linestyle='dotted')

max_V = max(abs(P_base[8000:]))
diff_V = max(abs(P_base[8000:])) - max(abs(Resp1[8000:]))
err_max_V = diff_V / max_V

print(f'Max. Error in base shear: {err_max_V * 100}%')

plt.legend(fontsize=16)
plt.grid(True)

plt.savefig('Base_shear.eps')
# %% Zoom base shear

plt.figure(None, figsize=(6, 6))

plt.xlim([0, 4])
plt.xlabel(r'Time [s]')
plt.ylabel(r'Base Shear [kN]')

plt.plot(Time, Resp1 / 1000, label='8 modes', linewidth=1, color='grey')
plt.plot(Time, Resp2 / 1000, label='3 modes', linewidth=.75, color='black', linestyle='dashed')
plt.plot(Time, P_base / 1000, label='Newmark', linewidth=.5, color='black', linestyle='dotted')

plt.legend(fontsize=16)
plt.grid(True)

plt.savefig('Base_shear_zoom.eps')

# %% Dynamic contribution factor


list_freqs = [14.038, 22.462, 28.253, 37.506, 52.188, \
              53.940, 66.357, 67.355, 68.295, 70.693, \
              75.107, 83.283, 85.197, 98.910, 101.412, \
              104.094]

list_xis = [0.05, 0.05, 0.054, 0.063, 0.08, 0.082, 0.097, 0.099, 0.1, 0.103, \
            0.109, 0.119, 0.122, 0.14, 0.143, 0.147, 0.149, 0.153, 0.157, 0.158, 0.159, 0.162, \
            0.174, 0.178, 0.18, 0.185, 0.19, 0.193, 0.208, 0.208]

R_i = []
w_s = 20

import numpy as np

for i, w in enumerate(list_freqs):
    r_w = w_s / w

    R = 1 / np.sqrt((1 - r_w ** 2) ** 2 + (2 * list_xis[i] * r_w) ** 2)

    R_i.append(R)

print(np.around(R_i, 2))
# %%
