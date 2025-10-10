# -*- coding: utf-8 -*-
"""
Created on Sun May 11 11:01:19 2025

@author: ibouckaert
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 15

plt.figure(None, (6, 6), dpi=400)

df1 = pd.read_excel('plot-data.xlsx', sheet_name='positive')
N_th_pos = df1['N'].to_numpy()
M_th_pos = df1['M'].to_numpy()

df1 = pd.read_excel('plot-data.xlsx', sheet_name='negative')
N_th_neg = df1['N'].to_numpy()
M_th_neg = df1['M'].to_numpy()

# Load the Excel file
df = pd.read_excel("data_pos.xlsx")

# Extract columns as NumPy arrays
N_h_pos = df["A"].to_numpy()
M_h_pos = df["B"].to_numpy()

df = pd.read_excel("data_neg.xlsx")

# Extract columns as NumPy arrays
N_h_neg = df["A"].to_numpy()
M_h_neg = df["B"].to_numpy()

# indices = np.argsort(N_h)
# M_h = M_h[indices]
# N_h = N_h[indices]

# %%
# plt.title('Interaction diagram for RC Beam')
plt.ylabel(r'Bending Moment $M$ [kNm]')
plt.xlabel(r'Axial force $N$ [kN]')
plt.grid()
plt.plot(N_th_pos, -M_th_pos, label=r'$\theta>0$ - Stressblock', color='red', linestyle='--', marker='.', markersize=7,
         markerfacecolor='white', linewidth=.5)
plt.plot(N_th_neg, -M_th_neg, label=r'$\theta<0$ - Stressblock', color='blue', linestyle='--', marker='.', markersize=7,
         markerfacecolor='white', linewidth=.5)
plt.plot(N_h_pos, M_h_pos, color='red', label='$\theta>0$ - HybriDFEM')
plt.plot(N_h_neg, M_h_neg, color='blue', label='$\theta<0$ - HybriDFEM')
plt.legend()

plt.savefig('interact_rc.eps')

# %%

D = 20e-3
Ac = 0.2 * 0.2
As = np.pi * D ** 2 / 4
Ab = np.pi * D
r = As / (Ac + As)

f = 5e3
t = 1e3

lb1 = D * f * (1 - r) / (4 * r * t)
lb2 = Ac * f / (Ab * t)

print(lb1, lb2)
