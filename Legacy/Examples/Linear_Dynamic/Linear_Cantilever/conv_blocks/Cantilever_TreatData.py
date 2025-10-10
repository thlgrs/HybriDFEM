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
from scipy.signal import find_peaks

# Folder to access the HybriDFEM files
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

styles = [':', '--', '-.', (0, (1, 2, 1, 2, 4, 2))]
colors = ['b', 'r', 'g', 'orange']

p_CAA = []
bl_CAA = []

p_LA = []
bl_LA = []

p_CDM = []
bl_CDM = []

p_real = []
bl_real = []

for i, file in enumerate(os.listdir()):

    meth = 0
    period = False

    if file.endswith('_NWK_g=0.5_b=0.25.h5'):
        meth = 1
    elif file.endswith('_NWK_g=0.5_b=0.17.h5'):
        meth = 2
    elif file.endswith('_CDM_.h5'):
        meth = 3
    elif file.endswith('Blocks.h5'):
        pass
    elif file.endswith('.h5'):
        period = True

    if meth > 0:

        print(file)

        with h5py.File(file, 'r') as hf:

            # Import what you need
            U = hf['U_conv'][-2]
            Time = hf['Time'][:]

        fft_result = np.fft.fft(U)
        frequencies = np.fft.fftfreq(len(Time), d=(Time[1] - Time[0]))

        # def exp_decay(t, A, B):
        #     return A * np.exp(-B * t)

        # analytic_signal = np.abs(np.fft.ifft(np.fft.fft(U) * 2))
        # envelope = np.abs(analytic_signal)

        # # Fit the envelope to extract B
        # t = np.arange(len(U))
        # popt, _ = curve_fit(exp_decay, t, envelope)

        # B = popt[1]

        magnitude = np.abs(fft_result)
        peaks, _ = find_peaks(magnitude, height=0)

        # xi = B / (2*np.pi*frequencies[peaks[0]])
        # print(xi)

        try:
            f1 = frequencies[peaks[0]]
            f2 = frequencies[peaks[1]]
            f3 = frequencies[peaks[2]]
        except:
            print(Time, U)
            # plt.plot(Time, U)
        # print(f"Peak frequencies: {meth}", w1, w2, w3)

        if meth == 1:
            p_CAA.append([1 / f1, 1 / f2, 1 / f3])
        elif meth == 2:
            p_LA.append([1 / f1, 1 / f2, 1 / f3])
        elif meth == 3:
            p_CDM.append([1 / f1, 1 / f2, 1 / f3])

        if meth == 1:
            bl_CAA.append(float(re.match(r'^(.*?)_NWK', file).group(1)))
        elif meth == 2:
            bl_LA.append(float(re.match(r'^(.*?)_NWK', file).group(1)))
        elif meth == 3:
            bl_CDM.append(float(re.match(r'^(.*?)_CDM', file).group(1)))

    elif period:

        with h5py.File(file, 'r') as hf:

            eig_vals = hf['eig_vals'][:4]

        bl_real.append(float(re.match(r'^(.*?).h5', file).group(1)))
        # print(bl_real[-1])
        p_real.append(2 * np.pi / eig_vals)
        print(eig_vals)

# %% Plotting
# record sur 60s 
# caa ok sauf moins que 9


sorted_caa = np.argsort(bl_CAA)
sorted_la = np.argsort(bl_LA)
sorted_cdm = np.argsort(bl_CDM)
sorted_real = np.argsort(bl_real)

CAA = [p_CAA[i] for i in sorted_caa]
LA = [p_LA[i] for i in sorted_la]
CDM = [p_CDM[i] for i in sorted_cdm]
REAL = [p_real[i] for i in sorted_real]
BL = [bl_real[i] for i in sorted_real]

T1_caa = [(CAA[i][0] - REAL[i][0]) / REAL[i][0] for i, sub in enumerate(CAA)]
T2_caa = [(CAA[i][1] - REAL[i][1]) / REAL[i][1] for i, sub in enumerate(CAA)]
T3_caa = [(CAA[i][2] - REAL[i][3]) / REAL[i][3] for i, sub in enumerate(CAA)]

T1_la = [(LA[i][0] - REAL[i][0]) / REAL[i][0] for i, sub in enumerate(LA)]
T2_la = [(LA[i][1] - REAL[i][1]) / REAL[i][1] for i, sub in enumerate(LA)]
T3_la = [(LA[i][2] - REAL[i][3]) / REAL[i][3] for i, sub in enumerate(LA)]

T1_cdm = [(CDM[i][0] - REAL[i][0]) / REAL[i][0] for i, sub in enumerate(CDM)]
T2_cdm = [(CDM[i][1] - REAL[i][1]) / REAL[i][1] for i, sub in enumerate(CDM)]
T3_cdm = [(CDM[i][2] - REAL[i][3]) / REAL[i][3] for i, sub in enumerate(CDM)]

save_data = 'Data_Conv_Blocks.h5'

with h5py.File(save_data, 'w') as f:
    f.create_dataset('CAA', data=[T1_caa, T2_caa, T3_caa])
    f.create_dataset('LA', data=[T1_la, T2_la, T3_la])
    f.create_dataset('CDM', data=[T1_cdm, T2_cdm, T3_cdm])
    f.create_dataset('BLOCKS', data=BL)
