# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:09:25 2025

@author: ibouckaert
"""

import numpy as np
import os
import h5py
import sys
import pathlib
import matplotlib.pyplot as plt
from scipy import interpolate

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "serif",  # Use a serif font
    "font.size": 12  # Adjust font size
})

xis = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
t_w = 110e-3

plt.figure(None, figsize=(6, 6), dpi=400)

for xi in xis:
    with h5py.File(f'W11_{xi * 100}_CDM_.h5', "r") as hf:
        U = hf["U_conv"][(9) * 3 + 0]
        time = hf['Time'][()]

    plt.plot(time, U / t_w, label=rf'$\xi = {xi * 100}$ \%', linewidth=.75)

plt.title("Wall 11 - Free-release test")
plt.xlabel("Time [s]")
plt.ylabel("d/t_w [-]")
plt.ylim((-.25, .65))
plt.xlim((0, 3.5))
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()

plt.figure(None, figsize=(6, 6), dpi=400)

for xi in xis:
    with h5py.File(f'W11_{xi * 100}_CDM_.h5', "r") as hf:
        V = hf["V_conv"][(9) * 3 + 0]
        time = hf['Time'][()]

    plt.plot(time, V, label=rf'$\xi = {xi * 100}$ \%', linewidth=.75)

plt.title("Wall 11 - Free-release test")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.ylim((-.5, .5))
plt.xlim((0, 3.5))
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
