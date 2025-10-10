# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 15:39:46 2025

@author: ibouckaert
"""

import numpy as np
import os
import h5py
import sys
import pathlib
import matplotlib.pyplot as plt

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as cont
import ContactPair as cp
import Surface as surf
import Material as mat

save_path = os.path.dirname(os.path.abspath(__file__))

# %% Steel material
FY = 510e6
FU = 635e6
ES = 204e9
ESH = 1430e6

ey = FY / ES
eu = 9e-2

STEEL = mat.steel_tensionchord(FY, FU, ey, eu, fail_crit=True)
STEEL.plot_stress_strain()

# %% Concrete material
FC = 32
FCT = 0.3 * FC ** (2 / 3) * 1e6
EC = 33e9

CONC = mat.concrete_tensionchord(FCT, EC)
CONC.plot_stress_strain()

# %% Check tensile strength ?
N1 = np.array([0, 0], dtype=float)
N2 = np.array([1, 0], dtype=float)

H = .2
B = .2

BLOCKS = 20
CPS = 10

As = np.pi * (50e-3) ** 2 / 4

St = st.Structure_2D()
St.add_beam(N1, N2, BLOCKS, H, 0, b=B, material=CONC)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

for cf in St.list_cfs:
    cf.add_reinforcement(0, As, material=STEEL, height=20e-3)

for i in range(BLOCKS):
    St.fixNode(i, [1, 2])

St.fixNode(0, 0)
F = 100e3
St.loadNode(BLOCKS - 1, 0, F)

St.plot_structure(scale=1, plot_cf=True)

LIST = np.linspace(0, 5e-4, 200)
LIST = np.append(LIST, np.linspace(5e-4, 1e-2, 100))
LIST = LIST.tolist()

St.solve_dispcontrol(LIST, 0, BLOCKS - 1, 0, tol=1e-4, filename='Simplified_TC')
St.plot_structure(scale=100, plot_cf=False)

# %% Plot results

import matplotlib.pyplot as plt

file = 'Simplified_TC.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U = hf['U_conv'][-3] * 1000
    P = hf['P_r_conv'][-3] / 1000

EA_eq = (EC * H * B + ES * As)
NFC = EA_eq * FCT / EC
d_c = NFC / EA_eq

EA_s = ES * As
NFs = d_c * EA_s

NFy = As * FY
d_y = NFy / EA_s

plt.figure(None, figsize=(6, 6), dpi=400)
plt.xlim(0, 10)
plt.ylim(0, 1200)
plt.xlabel('Elongation [mm]')
plt.ylabel('Applied force [kN]')
plt.grid(True)

plt.plot(U, P, linewidth=.75, color='black')
plt.scatter(d_c * 1000, NFC / 1000, color='red', marker='x', label='Concrete fails')
plt.scatter(d_c * 1000, NFs / 1000, color='green', marker='x', label='Only steel')
plt.scatter(d_y * 1000, NFy / 1000, color='orange', marker='x', label='Steel yields')

plt.legend()
