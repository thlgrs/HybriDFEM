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
import re

# Folder to access the HybriDFEM files
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

dir_name = 'conv_timestep'

styles = [':', '--', '-.', (0, (1, 2, 1, 2, 4, 2))]
colors = ['b', 'r', 'g', 'orange']

p_CAA = []
x_CAA = []
dt_CAA = []

p_LA = []
x_LA = []
dt_LA = []

p_CDM = []
x_CDM = []
dt_CDM = []


def damping_ratio(delta):
    return delta / np.sqrt(4 * np.pi ** 2 + delta ** 2)


def logarithmic_decrement(A, B):
    return 2 * np.pi * A / B


# def model(x, A, B, C):
#     return A * np.exp(-B * x) * np.cos(C * x)
for i, file in enumerate(os.listdir(dir_name)):

    with h5py.File(dir_name + '/' + file, 'r') as hf:

        # Import what you need
        U = hf['U_conv'][-2]
        Time = hf['Time'][:]

    if file.endswith('_NWK_g=0.5_b=0.25.h5'):  meth = 1
    if file.endswith('_NWK_g=0.5_b=0.17.h5'):  meth = 2
    if file.endswith('_CDM_.h5'):  meth = 3

    fft_result = np.fft.fft(U)
    frequencies = np.fft.fftfreq(len(Time), d=(Time[1] - Time[0]))

    from scipy.signal import find_peaks

    magnitude = np.abs(fft_result)
    peaks, _ = find_peaks(magnitude, height=0)
    w1 = frequencies[peaks[0]] * np.pi * 2
    w2 = frequencies[peaks[1]] * np.pi * 2
    w3 = frequencies[peaks[2]] * np.pi * 2

    print(f"Peak frequencies: {meth}", w1, w2, w3)

    if meth == 1:
        p_CAA.append([np.pi * 2 / w1, np.pi * 2 / w2, np.pi * 2 / w3])
    elif meth == 2:
        p_LA.append([np.pi * 2 / w1, np.pi * 2 / w2, np.pi * 2 / w3])
    elif meth == 3:
        p_CDM.append([np.pi * 2 / w1, np.pi * 2 / w2, np.pi * 2 / w3])

    # def model(x, A, B, C, D, E, F): 
    #     return A * np.exp(B * x) * np.cos(w1 * x) + C * np.exp(D * x) * np.cos(w2 * x) + E * np.exp(F * x) * np.cos(w3 * x)
    # conv=True
    # try: 
    #     params, _ = curve_fit(model, Time, U, p0=[1, 1, 1, 1, 1, 1])
    #     A, B, C, D, E, F = params
    #     d1 = logarithmic_decrement(B, w1)
    #     d2 = logarithmic_decrement(D, w2)
    #     d3 = logarithmic_decrement(F, w3)
    #     xi1 = damping_ratio(d1)
    #     xi2 = damping_ratio(d2)
    #     xi3 = damping_ratio(d3)

    #     print(xi1, xi2, xi3)
    #     conv=True
    # except: 
    #     print('Did not find a match for ' + file)
    #     conv = False

    # if conv: 
    #     if meth==1: x_CAA.append([xi1, xi2, xi3])
    #     elif meth == 2:  x_LA.append([xi1, xi2, xi3]) 

    if meth == 1:
        dt_CAA.append(float(re.match(r'^(.*?)_NWK', file).group(1)))
    elif meth == 2:
        dt_LA.append(float(re.match(r'^(.*?)_NWK', file).group(1)))
    elif meth == 3:
        dt_CDM.append(float(re.match(r'^(.*?)_CDM', file).group(1)))

    # if meth == 1 and dt_CAA[-1]==8e-5: plt.plot(Time, U, label='CAA')
    # if meth == 2 and dt_LA[-1]==8e-5: plt.plot(Time, U, label='LA')
    # if meth == 3 and dt_CDM[-1]==8e-5: plt.plot(Time, U, label='CDM')

# %% Plotting

T1 = 2 * np.pi / 113.72
T2 = 2 * np.pi / 649.266
T3 = 2 * np.pi / 1622

T1_list = [100 * (sub[0] - T1) / T1 for sub in p_CAA]
dt1 = [dt * 1000 for dt in dt_CAA]
sorted_i = np.argsort(dt_CAA)
T2_list = [100 * (sub[1] - T2) / T2 for sub in p_CAA]
dt2 = [dt * 1000 for dt in dt_CAA]
T3_list = [100 * (sub[2] - T3) / T3 for sub in p_CAA]
dt3 = [dt * 1000 for dt in dt_CAA]

# print(p_CAA)

plt.figure(None, figsize=(4.5, 4.5), dpi=800)
# plt.xscale('log')
plt.grid(True)
# plt.legend()
plt.ylim((0.0, 15))
plt.xlim((0, .3))
plt.plot([dt1[i] for i in sorted_i], [T1_list[i] for i in sorted_i], linewidth=1.5, linestyle='-.',
         label='Mode 1 - CAA', color='red')
plt.plot([dt2[i] for i in sorted_i], [T2_list[i] for i in sorted_i], linewidth=1.5, linestyle='-', label='Mode 2 - CAA',
         color='red')
plt.plot([dt3[i] for i in sorted_i], [T3_list[i] for i in sorted_i], linewidth=1.5, linestyle=':', label='Mode 3 - CAA',
         color='red')

plt.xlabel('$\Delta t$ [ms]', fontsize=15)
plt.ylabel('$(T_{TI,i} - T_i)/T_i$ [\%]', fontsize=15)

T1_list = [100 * (sub[0] - T1) / T1 for sub in p_LA]
dt1 = [dt * 1000 for dt in dt_LA]
sorted_i = np.argsort(dt_LA)
T2_list = [100 * (sub[1] - T2) / T2 for sub in p_LA]
dt2 = [dt * 1000 for dt in dt_LA]
T3_list = [100 * (sub[2] - T3) / T3 for sub in p_LA]
dt3 = [dt * 1000 for dt in dt_LA]
plt.plot([dt1[i] for i in sorted_i], [T1_list[i] for i in sorted_i], linewidth=1.5, linestyle='-.', label='Mode 1 - LA',
         color='blue')
plt.plot([dt2[i] for i in sorted_i], [T2_list[i] for i in sorted_i], linewidth=1.5, linestyle='-', label='Mode 2 - LA',
         color='blue')
plt.plot([dt3[i] for i in sorted_i], [T3_list[i] for i in sorted_i], linewidth=1.5, linestyle=':', label='Mode 3 - LA',
         color='blue')

T1_list = [100 * (sub[0] - T1) / T1 for sub in p_CDM]
dt1 = [dt * 1000 for dt in dt_CDM]
sorted_i = np.argsort(dt_CDM)
T2_list = [100 * (sub[1] - T2) / T2 for sub in p_CDM]
dt2 = [dt * 1000 for dt in dt_CDM]
T3_list = [100 * (sub[2] - T3) / T3 for sub in p_CDM]
dt3 = [dt * 1000 for dt in dt_CDM]
plt.plot([dt1[i] for i in sorted_i], [T1_list[i] for i in sorted_i], linewidth=1.5, linestyle='-.',
         label='Mode 1 - CDM', color='green')
plt.plot([dt2[i] for i in sorted_i], [T2_list[i] for i in sorted_i], linewidth=1.5, linestyle='-', label='Mode 2 - CDM',
         color='green')
plt.plot([dt3[i] for i in sorted_i], [T3_list[i] for i in sorted_i], linewidth=1.5, linestyle=':', label='Mode 3 - CDM',
         color='green')
plt.legend()

# x1_list = [sub[0] for sub in x_CAA]
# x2_list = [sub[1] for sub in x_CAA]
# x3_list = [sub[2] for sub in x_CAA]

# plt.figure(None)
# plt.xscale('log')
# plt.plot([dt1[i] for i in sorted_i], [x1_list[i] for i in sorted_i], linewidth=.75, marker='*', label='Mode 1')
# plt.plot([dt2[i] for i in sorted_i], [x2_list[i] for i in sorted_i], linewidth=.75, marker='*', label='Mode 1')
# plt.plot([dt3[i] for i in sorted_i], [x3_list[i] for i in sorted_i], linewidth=.75, marker='*', label='Mode 1')

# # # plt.legend(fontsize=8)
# # # plt.grid(linewidth=.25)
# # # plt.xlim((0,.5))
# # # # plt.ylim((-1,1))
# # # plt.gca().set_yticklabels([])

plt.savefig('beam_comp_dt.eps')
