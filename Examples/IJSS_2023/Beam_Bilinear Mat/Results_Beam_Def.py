# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:18:19 2024

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

# Folder to access the HybriDFEM files
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

#%% Import the structure

with open('Beam_Bilinear_Alpha=0.01.pkl', 'rb') as file:  # 'rb' means read in binary mode
    St1 = pickle.load(file)
    
with open('Beam_Bilinear_Alpha=0.05.pkl', 'rb') as file:  # 'rb' means read in binary mode
    St2 = pickle.load(file)

with open('Beam_Bilinear_Alpha=0.001.pkl', 'rb') as file:  # 'rb' means read in binary mode
    St3 = pickle.load(file)
    
#%% Plot structure at last converged step

St1.plot_structure(scale=10, plot_cf=False)
St2.plot_structure(scale=10, plot_cf=False)
St3.plot_structure(scale=1, plot_cf=False)

#%% Plot stresses at last converged step
St1.plot_stresses()
St2.plot_stresses()
St3.plot_stresses()
