# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:34:18 2024

@author: ibouckaert
"""
#%% Libraries imports
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

# Folder to access the HybriDFEM files
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))


# with open('Beam_Bilinear_Alpha.pkl', 'rb') as file:  # 'rb' means read in binary mode
#     St = pickle.load(file)
plt.figure(figsize=(5,5),dpi=800)
   
# plt.xlim([0, 15])
# plt.ylim([0, 3])
plt.title(r'Cantilever beam with bilinear material')
plt.xlabel(r'Vertical displacement of block [mm]')
plt.ylabel(r'Vertical applied force [kN]')

#%% Import results from simulation 

    
with open(f'Rocking_block.pkl', 'rb') as file: 
    St = pickle.load(file)
    nb_blocks = len(St.list_blocks)
    Lb = 3. / nb_blocks 
    
results = f'Disp_control_Horiz.h5'

with h5py.File(results, 'r') as hf:
    
    #Import what you need
    U = hf['U_conv'][-2]*1000
    P = hf['P_r_conv'][-2]/1000
    
    last_conv1 = hf['Last_conv'][()]
    last_def1 = hf['U_conv'][:,last_conv1]
    
plt.plot(U, P, label=None,linewidth=.5, marker='*', markersize=2)
  
    
#%% Make the plot(s)

plt.legend()
plt.grid()

# Save figure under given name (ideally .eps)
plt.savefig('Force_disp_Bilinear.eps')