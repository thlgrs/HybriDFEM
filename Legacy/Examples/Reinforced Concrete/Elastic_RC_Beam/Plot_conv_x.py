# -*- coding: utf-8 -*-
"""
Created on Thu May 15 19:29:45 2025

@author: ibouckaert
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('conv_cps.xlsx', engine='openpyxl')

# Extract arrays from columns
cps = df['CP'].to_numpy()
x = df['X'].to_numpy() * 100

x_ref = 11.8

error_x = (x - x_ref) / x_ref

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 15

plt.figure(1, figsize=(5, 5))

plt.plot(cps, error_x * 100, color='black', linewidth=1, marker='.', label='$x$')
# plt.plot(cps, error_c*100, color='blue', linewidth=.5,marker='.',label='$\sigma_c$')
plt.plot(np.arange(0, 210, 10), np.zeros(21), color='black', linewidth=.5, linestyle='dashed', label=None)

plt.grid(True, color='gainsboro', linewidth=.5)
plt.xlabel('Contact Pairs')
plt.ylabel('Relative error [\%]')
plt.xscale('log')
plt.xlim([2, 200])
plt.ylim([-30, 40])
# plt.legend(fontsize=17)

plt.savefig('rc_conv_x.eps')
