import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import pickle
import numpy as np
import pathlib
import sys
import os

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st

D = 14e-3

file = f'Structure_d={D * 1000}mm.pkl'

with open(file, 'rb') as f:
    St = pickle.load(f)

file = f'Danilo_TC_d={D * 1000}mm.h5'
with h5py.File(file, 'r') as hf:
    U = hf['U_conv'][()]
    P = hf['P_r_conv'][()]

FC = 32
FCT = 0.3 * FC ** (2 / 3) * 1e6
EC = 33e9

FY = 510e6
FU = 635e6
ES = 204e9
eps_y = FY / ES
eps_u = 9e-2
eps_c_max = FCT / EC

L = 1260


def plot_stresses_and_strains(save=None, color='black'):
    sigma_c, eps_c, x_c = St.get_stresses(angle=np.pi / 2, tag='CTC')
    sigma_s, eps_s, x_s = St.get_stresses(angle=np.pi / 2, tag='STC')
    mask = eps_c > eps_c_max
    print(x_c[mask])
    # print(eps_c)
    x = x_c[mask].tolist()
    x_end = x + [1.26]
    x_start = [0] + x
    print(np.array(x_end) - np.array(x_start))
    print(np.mean(np.array(x_end) - np.array(x_start)))
    # x_c = x_c[eps_c <= eps_c_max*1.01]
    # sigma_c = sigma_c[eps_c <= eps_c_max*1.01]
    # eps_c = eps_c[eps_c <= eps_c_max*1.01]
    eps_c[mask] *= 0

    # axs[1].set_ylabel(r'Steel Strain $[\%]$')
    axs[1].set_xlim(0, L)
    lim_eps_s = max(eps_y, np.max(eps_s))
    axs[1].set_ylim(0, 5.5)

    # axs[1].plot(x_c*1000, eps_c*100, color=color,label=r'$\varepsilon_c$', linewidth =0.1, marker='.',markersize=1)
    axs[1].plot(x_s * 1000, eps_s * 100, color=color, label=r'$\varepsilon_s$', linewidth=0.1, marker='.', markersize=1)
    # axs[1].axhline(y=eps_c_max*100, color='grey', label=r'$\varepsilon_{c, max}$', linewidth=.5, marker=None)
    axs[1].axhline(y=eps_y * 100, color='grey', label=r'$\varepsilon_{y}$', linewidth=.5, marker=None, linestyle=':')
    axs[1].set_xticks([])
    # axs[1].legend(loc='lower left')

    axs[0].set_ylabel(r'Concrete Strain $[\%]$')
    axs[0].set_xlim(0, L)
    axs[0].set_ylim(0, 0.01)

    axs[0].plot(x_c * 1000, eps_c * 100, color=color, label=r'$\varepsilon_c$', linewidth=0.1, marker='.', markersize=1)
    # axs[0].plot(x_s*1000, eps_s*100, color=color, label=r'$\varepsilon_s$', linewidth =0.1, marker='.',markersize=1)
    axs[0].axhline(y=eps_c_max * 100, color='grey', label=r'$\varepsilon_{c, max}$', linewidth=.5, marker=None,
                   linestyle=':')
    axs[0].set_xticks([])
    # axs[0].legend(loc='lower left')

    # axs[2].set_ylabel('Concrete Stress [MPa]')
    axs[2].set_xlim(0, L)
    axs[2].set_ylim(-.01, 3.5)

    axs[2].plot(x_c * 1000, sigma_c / 1e6, color=color, label=r'$\sigma_c$', linewidth=0.1, marker='.', markersize=1)
    axs[2].axhline(y=FCT / 1e6, color='grey', label=r'$f_{c,t}$', linewidth=.5, marker=None, linestyle=':')
    # axs[2].legend(loc='upper left')
    axs[2].set_xticks([])
    # axs[3].set_xlabel('Position [mm]')
    # axs[3].set_ylabel('Steel Stress [MPa]')
    axs[3].set_xlim(0, L)

    axs[3].set_ylim(300, 635)

    axs[3].plot(x_s * 1000, sigma_s / 1e6, color=color, label=r'$\sigma_s$', linewidth=0.1, marker='.', markersize=1)
    axs[3].axhline(y=FY / 1e6, color='grey', label=r'$f_{y}$', linewidth=.5, marker=None, linestyle=':')
    # axs[3].legend(loc='lower left')
    axs[3].set_xticks([])
    axs[4].set_xlabel('Position [mm]')
    # axs[4].set_ylabel('Crack opening [mm]')
    axs[4].set_xlim(0, L)

    sigma_c, eps_c, x_c = St.get_stresses(angle=np.pi / 2, tag='CTC')
    # sigma_s, eps_s, x_s = St.get_stresses(angle=np.pi/2, tag='STC')
    mask = eps_c > eps_c_max

    try:
        max_eps = max(1e3 * eps_c[mask] * 1e-2)
    except:
        max_eps = 0
    axs[4].set_ylim(0, 6)
    axs[4].grid(True)
    # print(mask)
    print(color)
    print(1e3 * eps_c[mask] * 1e-2)
    # print(x_c[mask],eps_c[mask]*1e-2)
    axs[4].bar(x_c[mask] * 1e3, 1e3 * eps_c[mask] * 1e-2, width=10, color=color)  #

    # plt.close()


for cf in St.list_cfs:
    if cf.bl_A.material.tag == 'CTC' and cf.bl_B.material.tag == 'CTC':
        for cp in cf.cps:
            cp.sp1.law.cracked = False
            cp.sp2.law.cracked = False

print(U.shape)
# ratios=[0,500]
ratios = [0, 1714, 8571, 19999]
ratios = [0, 19999, 8571, 1714]

colors = [None, 'red', 'blue', 'green']

fig, axs = plt.subplots(5, 1, figsize=(6, 6))
for i, ratio in enumerate(ratios):
    print(ratio)
    St.U = U[:, ratio]
    # St.plot_structure(scale=10, plot_cf=False, plot_supp=False)
    St.get_P_r()
    saveto = None

    plot_stresses_and_strains(save=saveto, color=colors[i])

plt.savefig('stress_strain_tc.eps')
