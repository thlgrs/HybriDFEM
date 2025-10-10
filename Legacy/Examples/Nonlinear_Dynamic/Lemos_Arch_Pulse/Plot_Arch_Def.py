# -*- coding: utf-8 -*-
"""
Created on Fri May  2 14:02:15 2025

@author: ibouckaert
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import h5py
import pickle
import pathlib
import sys

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "serif",  # Use a serif font
    "font.size": 12  # Adjust font size
})

file = f'Lemos_Arch_Coulomb.pkl'

with open(file, 'rb') as f:
    St = pickle.load(f)

file = 'NoTension_0.17g_0.05_CDM_.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    # print(file)
    U = hf['U_conv'][()]
    Time = hf['Time'][()]

Node = 9
Def_index_max = np.argmax(U[3 * Node])
Def_index_min = np.argmin(U[3 * Node])
print(Time[Def_index_max])
print(Time[Def_index_min])

St.U = U[:, -1]

St.plot_structure(scale=1, plot_forces=False, plot_cf=False, plot_supp=False, save='Ult_Def.eps')
