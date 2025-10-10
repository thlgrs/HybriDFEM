# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 10:00:21 2025

@author: ibouckaert
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'Calibri'
# mpl.rcParams['font.serif'] = ['Calibri'] 
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

t_p_h = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.31, 0.32, 0.33, 0.34, .35, 0.36, 0.37, 0.38, 0.39, \
                  0.4, 0.5, 0.6, 0.8, 1.0])
t_p_l = np.array([0.1, 0.15, 0.2, 0.3, 0.31, 0.32, 0.33, 0.34, .35, 0.36, 0.37, 0.38, 0.39, \
                  0.4, 0.5, 0.6, 0.8, 1.0])
# a_udec = 

x_udec = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.32, 0.33, 0.34, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0])
y_udec = np.array([3.6, 2.05, 1.46, 1.09, 0.87, 0.8, 0.78, 0.46, 0.38, 0.31, 0.28, 0.27, 0.25, 0.22, 0.20])

a_hd = np.array([4.16, 2.14, 1.46, 1.09, 0.88, 0.84, 0.81, 0.78, 0.39, 0.37, 0.34, 0.33, 0.31, 0.29, \
                 0.29, 0.25, 0.23, 0.22, 0.21])

a_lm = np.array([4.3, 2.2, 1.48, 0.88, 0.85, 0.82, 0.79, 0.44, 0.43, 0.38, 0.36, 0.33, 0.31, \
                 0.3, 0.25, 0.24, 0.22, 0.21])

plt.figure(None, (6, 6))

plt.plot(t_p_l, a_lm, label='LMGC90', color='grey', marker='x', linewidth=.75)
plt.plot(x_udec, y_udec, label='UDEC', color='grey', marker='^', linewidth=.75)
plt.plot(t_p_h, a_hd, label='HybriDFEM', color='black', marker='.', linewidth=1, markersize=9)
plt.axhline(y=0.19, color='grey', linestyle='--', label='Limit analysis')

plt.grid(True)
plt.xlim([0, 1])
plt.ylim([0, 5])
plt.xlabel('Impulse period $t_p$ [s]')
plt.ylabel('Impulse amplitude $a$ [g]')
plt.legend()

plt.savefig('Failuredomain.eps')
