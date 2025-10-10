import h5py
import matplotlib.pyplot as plt
import numpy as np
import importlib
import pathlib
import sys
import pickle
import matplotlib as mpl


def reload_modules():
    importlib.reload(st)


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'Calibri'
# mpl.rcParams['font.serif'] = ['Calibri'] 
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st

filename = 'Results_Modal_Deg.h5'

with h5py.File(filename, 'r') as hf:
    U = hf['U'][:]
    P = hf['P'][:]
    w = hf['w'][:]

# %%

import matplotlib.pyplot as plt

plt.figure(None, figsize=(6, 6), dpi=1000)
plt.xlabel(r'Control displacement [mm]')
plt.ylabel(r'Load multiplier $\alpha$')
plt.grid()
n_incr = 10
Node = 11
RHO = 1000

for i in range(1, n_incr + 1):

    results = f'Wall/Step_{i}_DispControl.h5'

    with h5py.File(results, 'r') as hf:
        if i == 1:
            label = 'Loading'
        else:
            label = None

        # Import what you need
        P = hf['P_r_conv'][3 * Node] / (.2 * .2 * 9.81 * RHO)
        U = hf['U_conv'][3 * Node] * 1000
        plt.plot(U, P, color='blue', linewidth=.75, label=label)

    results = f'Wall/Step_{i}_ForceControl.h5'

    with h5py.File(results, 'r') as hf:
        if i == 1:
            label = 'Unloading'
        # Import what you need
        P = hf['P_r_conv'][3 * Node] / (.2 * .2 * 9.81 * RHO)
        U = hf['U_conv'][3 * Node] * 1000
        plt.plot(U, P, color='red', linewidth=.75, label=label)
plt.xlim([0, 0.16])
plt.ylim([0, 0.6])
# plt.xticks([0,.0.05, 0.075, 0.1, 0.125, 0.15])
plt.savefig('Force-displacement.png')

# %%

plt.figure(None, figsize=(6, 6), dpi=1000)
nb_steps = w.shape[1]
steps = np.arange(0, nb_steps)

color = ['blue', 'red', 'green', 'orange', 'purple']

for i in range(5):
    plt.plot(steps, w[i, :] / w[i, 0], label=f'$\omega_{{{i + 1}}}$', color=color[i])
    # plt.plot(steps,w[i,:], label=f'$\omega_{{{i+1}}}$')

plt.ylabel('Normalized frequency [rad/s]')
plt.xlabel('Loading/Unloading cycle')
plt.xlim([0, nb_steps - 1])
plt.ylim([0, 1.05])
plt.grid()
plt.legend()
plt.xticks(steps)
plt.savefig('Evolution_Freqs.png')

# %% Plotting modes of interest

steps = [0, 10]

nb_modes = 1

for i in steps:
    with open(f'Wall/Plastic_Frame.pkl', 'rb') as file:
        St = pickle.load(file)

    # St.plot_modes(2, scale=1)

    filename = f'Wall/Step_{i}_Modal_C.h5'

    with h5py.File(filename, 'r') as hf:
        St.eig_vals = hf['eig_vals'][()]
        St.eig_modes = hf['eig_modes'][()]

    St.plot_modes(nb_modes, scale=-1, save=True, folder=f'Wall/Step_{i}', show=True)

# %%
