# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 11:03:29 2025

@author: ibouckaert
"""

import numpy as np
import os
import h5py
import sys
import pathlib
import matplotlib.pyplot as plt

blocks = [50, 60, 70, 80, 90, 100, 110]

plt.figure(None, figsize=(6, 6))

for bl in blocks:
    file = f'TC_Danilo_{bl}BL.h5'
    Node = bl - 1
    with h5py.File(file, 'r') as hf:
        # Import what you need
        # print(file)
        last_conv = hf['Last_conv'][()]
        print(last_conv)
        U = hf['U_conv'][3 * Node, :last_conv] * 1000
        P = hf['P_r_conv'][3 * Node, :last_conv] / 1000

    plt.plot(U, P, label=f'{bl} Blocks')

plt.legend()
plt.grid(True)
plt.xlim((0, 35))
plt.ylim((0, 400))
