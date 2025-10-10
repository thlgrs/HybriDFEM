# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:46:10 2025

@author: ibouckaert
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py

# %% Rotations

acc = [.66]

plt.figure(None, (6, 6))
plt.xlim((0, 10))
plt.ylim((-1, 1))
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Rotation [rad]')

for i, a in enumerate(acc):
    file = f'TwoBlocks_{a}g_NWK_g=0.5_b=0.25.h5'

    with h5py.File(file, 'r') as hf:
        # Import what you need
        last_conv = hf['Last_conv'][()]
        Th_conv = hf['U_conv'][-1, :last_conv]
        Th_p_conv = hf['U_conv'][-1, :last_conv]
        Time = hf['Time'][:last_conv]

        plt.plot(Time, Th_conv, label=f'{a}g')
plt.legend()

# %% Velocities

plt.figure(None, (6, 6))
plt.xlim((0, 2))
# plt.ylim((-.1,.1))
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Angular velocity [rad/s]')

for i, a in enumerate(acc):
    file = f'TwoBlocks_{a}g_NWK_g=0.5_b=0.25.h5'

    with h5py.File(file, 'r') as hf:
        # Import what you need
        last_conv = hf['Last_conv'][()]
        Th_p_conv = hf['V_conv'][-1, :last_conv]
        Time = hf['Time'][:last_conv]

        plt.plot(Time, Th_p_conv, label=f'{a}g')
plt.legend()

# %% Accelerations

plt.figure(None, (6, 4))
plt.xlim((0, 3))
# plt.ylim((-.1,.1))
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Angular velocity [rad/s]')

for i, a in enumerate(acc):
    file = f'TwoBlocks_{a}g_NWK_g=0.5_b=0.25.h5'

    with h5py.File(file, 'r') as hf:
        # Import what you need
        last_conv = hf['Last_conv'][()]
        Th_pp_conv = hf['A_conv'][-1, :last_conv]
        Time = hf['Time'][:last_conv]

        plt.plot(Time, Th_pp_conv, label=f'{a}g')
plt.legend()
