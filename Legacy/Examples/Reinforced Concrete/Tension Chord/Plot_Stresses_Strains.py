import matplotlib.pyplot as plt
import h5py
import pickle
import numpy as np
import pathlib
import sys

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st

D = 45e-3
H = .2
B = .2

BLOCKS = 100
Node = BLOCKS - 1

EC = 33e9
ES = 204e9
FC = 32
FCT = 0.3 * FC ** (2 / 3) * 1e6
TB0 = 2 * FCT
FY = 510e9

L = 3

file = f'Structure_d={D * 1000}mm.pkl'

with open(file, 'rb') as f:
    St = pickle.load(f)

file = f'Full_TC_d={D * 1000}mm.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U = hf['U_conv'][()]
    P = hf['P_r_conv'][()]

# St.plot_structure(scale=500, plot_cf=False, plot_supp=False)
St.U = U[:, 1000]
St.plot_structure(scale=500, plot_cf=False, plot_supp=False)
St.get_P_r()

FC = 32
FCT = 0.3 * FC ** (2 / 3) * 1e6
EC = 33e9

FY = 510e6
FU = 635e6
ES = 204e9
eps_y = FY / ES
eps_u = 9e-2
eps_c_max = FCT / EC

sigma_c, eps_c, x_c = St.get_stresses(angle=np.pi / 2, tag='CTC')
sigma_s, eps_s, x_s = St.get_stresses(angle=np.pi / 2, tag='STC')

# x_c = x_c[eps_c <= eps_c_max*1.01]
# sigma_c = sigma_c[eps_c <= eps_c_max*1.01]
# eps_c = eps_c[eps_c <= eps_c_max*1.01]

fig, axs = plt.subplots(5, 1, figsize=(10, 6))

axs[0].set_ylabel(r'Steel Strain $[\%]$')
axs[0].set_xlim(0, 1000)
axs[0].set_ylim(0, eps_y * 300)

axs[0].plot(x_c * 1000, eps_c * 100, color='red', label=r'$\varepsilon_c$', linewidth=0.1, marker='.', markersize=1)
axs[0].plot(x_s * 1000, eps_s * 100, color='blue', label=r'$\varepsilon_s$', linewidth=0.1, marker='.', markersize=1)
axs[0].axhline(y=eps_c_max * 100, color='grey', label=r'$\varepsilon_{c, max}$', linewidth=.5, marker=None)
axs[0].axhline(y=eps_y * 100, color='grey', label=r'$\varepsilon_{y}$', linewidth=.5, marker=None)

axs[0].legend(loc='lower left')

axs[1].set_ylabel(r'Concrete Strain $[\%]$')
axs[1].set_xlim(0, 1000)
axs[1].set_ylim(0, eps_c_max * 110)

axs[1].plot(x_c * 1000, eps_c * 100, color='red', label=r'$\varepsilon_c$', linewidth=0.1, marker='.', markersize=1)
axs[1].plot(x_s * 1000, eps_s * 100, color='blue', label=r'$\varepsilon_s$', linewidth=0.1, marker='.', markersize=1)
axs[1].axhline(y=eps_c_max * 100, color='grey', label=r'$\varepsilon_{c, max}$', linewidth=.5, marker=None)

axs[1].legend(loc='lower left')

axs[2].set_ylabel('Concrete Stress [MPa]')
axs[2].set_xlim(0, 1000)
axs[2].set_ylim(-.1, 3.5)

axs[2].plot(x_c * 1000, sigma_c / 1e6, color='red', label=r'$\sigma_c$', linewidth=0.1, marker='.', markersize=1)
axs[2].axhline(y=FCT / 1e6, color='grey', label=r'$f_{c,t}$', linewidth=.5, marker=None)
axs[2].legend(loc='upper left')

axs[3].set_xlabel('Position [mm]')
axs[3].set_ylabel('Steel Stress [MPa]')
axs[3].set_xlim(0, 1000)
axs[3].set_ylim(0, FU / 1e6)

axs[3].plot(x_s * 1000, sigma_s / 1e6, color='blue', label=r'$\sigma_s$', linewidth=0.1, marker='.', markersize=1)
axs[3].axhline(y=FY / 1e6, color='grey', label=r'$f_{y}$', linewidth=.5, marker=None)
axs[3].legend(loc='lower left')

axs[4].set_xlabel('Position [mm]')
axs[4].set_ylabel('Crack opening [mm]')
axs[4].set_xlim(0, 1000)
# axs[4].set_ylim(0, FU/1e6)
mask = eps_c > eps_c_max
print(mask)
print(x_c[mask], eps_c[mask] * 1e-2)
axs[4].bar(x_c[mask] * 1e3, eps_c[mask] * 1e-2 * 1e3, width=1)  #

# plt.savefig(saveto)
# plt.close()
