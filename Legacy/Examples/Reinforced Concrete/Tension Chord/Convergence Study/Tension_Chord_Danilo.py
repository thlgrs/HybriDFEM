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
# STEEL.plot_stress_strain()

# %% Concrete material
FC = 32
FCT = 0.3 * FC ** (2 / 3) * 1e6
EC = 33e9

CONC = mat.concrete_tensionchord(FCT, EC)
# CONC.plot_stress_strain()


# %% Bond slip model
TB0 = 2 * FCT
# print(TB0, TB0*np.pi/2/1000)
TB1 = FCT

BSTC = surf.bond_slip_tc(TB0, TB1)

# %% Check tensile strength ?
H = .2
B = .2
D = 14e-3
As = 4 * np.pi * (D) ** 2 / 4
L = 1.26

Ac = H * B
r = As / Ac

if r > 0.04: print('Trop d armatures dans la section')
print(f'Reinforcement ratio is {r}')

N1 = np.array([0, 0], dtype=float)
N2 = np.array([L, 0], dtype=float)

N3 = np.array([0, (D + H) / 2], dtype=float)
N4 = np.array([L, (D + H) / 2], dtype=float)

# N5 = np.array([0,-(D+H/2)/2],dtype=float)
# N6 = np.array([L,-(D+H/2)/2],dtype=float)

BLOCKS = 60
CPS = 1

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, D, 0, b=As / D, material=STEEL)
St.add_beam(N3, N4, BLOCKS, H, 0, b=B, material=CONC)
# St.add_beam(N5, N6, BLOCKS, H/2, 0, b=B, material=CONC)


St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

for cf in St.list_cfs:
    if not cf.bl_A.material.tag == cf.bl_B.material.tag:
        cf.b = 4 * np.pi * D
        cf.change_cps(nb_cp=1, offset=-1, surface=BSTC)

# %%
for i in range(1, 2 * BLOCKS):
    St.fixNode(i, [1, 2])

# St.fixNode(1,[1,2])
St.fixNode(0, [0, 1, 2])
# St.fixNode(BLOCKS,[0])
# St.fixNode(2*BLOCKS,[0])
F = 100e3
Node = BLOCKS - 1
# St.fixNode(Node,[0])
St.loadNode(Node, 0, F)

# St.plot_structure(scale=1,plot_cf=True, plot_supp=True, plot_forces=True)

EA_eq = (EC * Ac + ES * As)
NFC = EA_eq * FCT / EC
d_c = NFC * L / EA_eq

LIST = np.linspace(0, 35e-3, 50000)
# LIST = np.append(LIST, np.linspace(4e-3, 35e-3, 10000))
# LIST = np.append(LIST, np.linspace(1e-3, 2e-3, 100))
LIST = LIST.tolist()

St.solve_dispcontrol(LIST, 0, Node, 0, tol=1e-2, filename=f'TC_Danilo_{BLOCKS}BL', max_iter=1000)
St.plot_structure(scale=100, plot_cf=False, plot_supp=False, plot_forces=True)

St.save_structure(f'TC_Danilo_{BLOCKS}BL')

# %% Plot results

import matplotlib.pyplot as plt

file = f'TC_Danilo_{BLOCKS}BL.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U = hf['U_conv'][3 * Node] * 1000
    P = hf['P_r_conv'][3 * Node] / 1000

As = np.pi * D ** 2 / 4
Ac = H * B

EA_eq = (EC * Ac + ES * As)
NFC = EA_eq * FCT / EC
d_c = NFC * L / EA_eq

Fc = TB0 * np.pi * D * (L / BLOCKS)

EA_s = ES * As
NFs = d_c * EA_s

NFy = As * FY
d_y = NFy * L / EA_s

r = As / Ac
l_b = D * FCT * (1 - r) / (4 * r * TB0)
l_b = np.around(l_b, 5)
print(f'Dev length is {l_b * 1000}mm, Crack spacing should be between {l_b * 1000}mm and {2 * l_b * 1000}mm,')

plt.figure(None, figsize=(6, 6), dpi=400)
# plt.xlim(0,3.5)
# plt.ylim(0,175)
plt.xlabel('Elongation [mm]')
plt.ylabel('Applied force [kN]')
plt.grid(True)

plt.plot(U, P, linewidth=.75, color='black')
plt.scatter(d_c * 1000, NFC / 1000, color='red', marker='x', label='Concrete fails')
plt.scatter(d_c * 1000, (NFs) / 1000, color='green', marker='x', label='Only steel')
plt.scatter(d_y * 1000, NFy / 1000, color='orange', marker='x', label='Steel yields')

plt.legend()

# %% Plotting stress profile

# sigma_c, eps_c, x = St.get_stresses(angle=np.pi/2, tag='CTC')
# sigma_s, eps_s, x = St.get_stresses(angle=np.pi/2, tag='STC')

# plt.figure(None, figsize=(6,2),dpi=400)
# plt.xlabel('Position [mm]')
# plt.ylabel('Strain [\%]')

# eps_c_max = FCT/EC
# # eps_c[eps_c > eps_c_max] = 0

# plt.plot(x, eps_c*100, color='red',label=r'$\varepsilon_c$')
# plt.plot(x, eps_s*100, color='blue', label=r'$\varepsilon_s$')

# plt.legend()

sigma_bs, tau_bs, x = St.get_stresses()
St.plot_stresses(angle=0)
