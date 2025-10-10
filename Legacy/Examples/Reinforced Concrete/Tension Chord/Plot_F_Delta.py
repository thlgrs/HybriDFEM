import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import pickle
import numpy as np
import pathlib
import sys
import os
import pandas as pd

# To have a nice LaTeX rendering (import LaTeX)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 15

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st

D = 14e-3
Node = 149

file = f'Structure_d={D * 1000}mm.pkl'

with open(file, 'rb') as f:
    St = pickle.load(f)

file = f'Danilo_TC_d={D * 1000}mm.h5'
with h5py.File(file, 'r') as hf:
    U = hf['U_conv'][()]
    P = hf['P_r_conv'][()]

    # max_P = np.max(P)
    # print(max_P)


def plot_F_Delta(U, P, save=None):
    plt.figure(None, figsize=(5, 5), dpi=400)
    try:
        x_max = max(U)
        max_P = max(P)
    except:
        x_max = 1e-5
        max_P = 1e-5

    # max_P = np.max(P)
    # print(max(U))
    plt.xlim(0, 35)
    plt.ylim(0, 400)
    plt.xlabel('Elongation $\Delta$ [mm]')
    plt.ylabel('Applied force $N$ [kN]')
    plt.grid(True)

    df = pd.read_excel('plot-data.xlsx', engine='openpyxl')

    # Extract arrays from columns
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    plt.plot(x, y, color='grey', linestyle='--', linewidth=1, label='Tension chord')

    plt.plot(U * 1000, P / 1000, linewidth=1, color='black', label='HybriDFEM')
    # plt.legend()

    plt.savefig(saveto)
    # plt.close()


U_new = U[3 * Node]
P_new = P[3 * Node]
saveto = 'F_delta_tc_z.eps'

plot_F_Delta(U_new, P_new, save=saveto)
