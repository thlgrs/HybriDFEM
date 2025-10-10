# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 18:56:53 2025

@author: ibouckaert
"""

import numpy as np
import matplotlib.pyplot as plt

plt.figure(None, (5, 5))

x = np.linspace(0, 2.5, 1000)
y = np.zeros(1000)

for i, t in enumerate(x):
    if t < .25:
        y[i] = 0.15 * np.sin(2 * np.pi * t / 0.5)

plt.plot(x, y)
plt.savefig('impulse.eps')
