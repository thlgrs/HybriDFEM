# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 15:49:37 2025

@author: ibouckaert
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py

discs = [10, 9, 8, 7]

plt.figure(None, figsize=(6, 6), dpi=400)
for disc in discs:
    filename = f'{disc}Bl_.h5'

    with h5py.File(filename, 'r') as hf:
        # Import what you need
        last_conv = hf['Last_conv'][()]
        P_c = hf['P_r_conv'][3 * disc - 3, :last_conv] / 1000
        U_c = hf['U_conv'][3 * disc - 3, :last_conv] * 1000

    plt.plot(U_c, P_c, label=f'{disc} Blocks')

plt.title('Non-regularized')
plt.grid(True)
plt.legend()
# plt.xlim((-5,1))
# plt.ylim(-800,100)

# %% Regularized plotplt.figure(None,figsize=(6,6),dpi=400)

# plt.figure(None,figsize=(6,6),dpi=400)

# for disc in discs: 

#     filename = f'Results_{disc}_R.h5'

#     with h5py.File(filename, 'r') as hf:

#         #Import what you need
#         last_conv = hf['Last_conv'][()]
#         P_c = -hf['P_r_conv'][3*disc-3,:last_conv] / 1000
#         U_c = -hf['U_conv'][3*disc-3,:last_conv] * 1000

#         plt.plot(U_c, P_c, label=f'{disc} Blocks')

# plt.title('Regularized with fracture energy')
# plt.grid(True)
# plt.legend()
# plt.xlim((0,5))
# plt.ylim(0,1300)
