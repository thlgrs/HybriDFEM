import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import pickle
import numpy as np
import pathlib
import sys
import os

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st

D = 14e-3

file = f'Structure_d={D * 1000}mm.pkl'

with open(file, 'rb') as f:
    St = pickle.load(f)

file = f'Danilo_TC_d={D * 1000}mm.h5'
with h5py.File(file, 'r') as hf:
    U = hf['U_conv'][()]
    P = hf['P_r_conv'][()]

St.U = U[:, 19999]

saveto = 'def_beam_tc.eps'
St.plot_structure(scale=5, plot_cf=False, plot_supp=False, plot_forces=True, lims=[[-.1, 2], [-D, D + .3]], show=True,
                  save=saveto)
