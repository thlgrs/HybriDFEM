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

files = [
    'First_Test_NWK_g=0.5_b=0.25.h5',
    'First_Test_GEN_am=0_af=0.1_g=0.6_b=0.3.h5']
# 'First_Test_CDM_.h5']

BLOCKS = 20

names = ['LA', 'CAA', 'HHT', 'CDM']
colors = ['red', 'blue', 'green', 'orange']

plt.figure(None, figsize=(6, 6), dpi=400)

for i, file in enumerate(files):
    with h5py.File(file, "r") as hf:
        U_c = hf["U_conv"][1] * 1000
        U_end = hf["U_conv"][(BLOCKS - 1) * 3 + 1] * 1000
        time = hf['Time'][()]

    plt.plot(time, U_end, label=names[i], linewidth=1, color=colors[i])
    plt.plot(time, U_c, label=None, linewidth=.5, color=colors[i])

# plt.title("Wall 11 - Free-release test")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [mm]")
plt.ylim((-60, 60))
plt.xlim((0, 15))
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
