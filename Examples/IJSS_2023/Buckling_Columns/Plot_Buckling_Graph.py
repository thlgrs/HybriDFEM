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
import sys
import pathlib
import numpy as np

# Folder to access the HybriDFEM files
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

plt.figure(figsize=(5, 5), dpi=800)

plt.xlim([0, 3])
plt.ylim([0, 1.5])
plt.title(r'Buckling and post-buckling of columns')
plt.xlabel(r'Displacement [mm]')
plt.ylabel(r'Vertical applied force')

# %% Import results from simulation
BCs = ['FFr', 'PP', 'FP', 'FF']
BCs = ['FF']
Colors = ['red', 'green', 'blue', 'orange']
L = 4
E = 30e9
H = .2

for i, bc in enumerate(BCs):

    with open(f'Buckling Column_' + bc + '.pkl', 'rb') as file:
        St = pickle.load(file)
        nb_blocks = len(St.list_blocks)

    results = f'Buckling Column_' + bc + '.h5'

    with h5py.File(results, 'r') as hf:

        # Import what you need
        F_v = -hf['P_r_conv'][-2]
        U_v = -hf['U_conv'][-2]
        U_h = hf['U_conv'][int(nb_blocks / 2)]

        last_conv = hf['Last_conv'][()]
        last_def = hf['U_conv'][:, last_conv]

    if bc == 'FFr':
        Lf = 2 * L
    elif bc == 'PP':
        Lf = L
    elif bc == 'FP':
        Lf = .7 * L
    elif bc == 'FF':
        Lf = .5 * L

    EULER = (np.pi / Lf) ** 2 * E * H ** 4 / 12

    plt.plot(U_v, F_v / EULER, label=bc + r'$-y$', linewidth=1, color=Colors[i])
    plt.plot(U_h, F_v / EULER, label=bc + r'$-x$', linewidth=1, linestyle='dashed', color=Colors[i])

# %% Make the plot(s)

plt.legend()
plt.grid()

# Save figure under given name (ideally .eps)
plt.savefig('Force_Disp_Buckling.eps')
