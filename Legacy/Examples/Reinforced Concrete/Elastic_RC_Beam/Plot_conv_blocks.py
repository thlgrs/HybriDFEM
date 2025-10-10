# -*- coding: utf-8 -*-
"""
Created on Thu May 15 19:29:45 2025

@author: ibouckaert
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('conv_blocks.xlsx', engine='openpyxl')

# Extract arrays from columns
cps = df['BLOCKS'].to_numpy()
sig_s = df['SIG_S'].to_numpy() / 1e6
sig_c = -df['SIG_C'].to_numpy() / 1e6

print(sig_s)
print(sig_c)

sig_s_ref = 290.64
sig_c_ref = 15.48

error_s = (sig_s - sig_s_ref) / sig_s_ref
error_c = (sig_c - sig_c_ref) / sig_c_ref

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 15

plt.figure(1, figsize=(5, 5))

plt.plot(cps, error_s * 100, color='black', linewidth=1, marker='x', label='$\sigma_s$')
plt.plot(cps, error_c * 100, color='grey', linewidth=.5, marker='.', label='$\sigma_c$')
plt.plot(np.arange(0, 200, 10), np.zeros(20), color='black', linewidth=.5, linestyle='dashed', label=None)

plt.grid(True, color='gainsboro', linewidth=.5)
plt.xlabel('Blocks')
plt.ylabel('Relative error [\%]')
# plt.xscale('log')
plt.xlim([2, 200])
plt.ylim([-20, 2])

plt.legend(fontsize=17)

# plt.savefig('rc_conv_blocks.eps')
