# %% Libraries imports
import matplotlib as mpl
import h5py

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

import matplotlib.pyplot as plt

save_data = 'Data_Conv_Blocks.h5'

with h5py.File(save_data, 'r') as f:
    CAA = f['CAA'][:]
    LA = f['LA'][:]
    CDM = f['CDM'][:]
    BL = f['BLOCKS'][:]

plt.figure(None, figsize=(4.5, 4.5), dpi=800)
# # plt.xscale('log')
plt.grid(True)
plt.legend()
plt.ylim((-0.01, 5))
plt.xlim((4, 50))
plt.plot(BL, CAA[0] * 100, linewidth=1.5, linestyle='-.', label='$T_{1,CAA}$', color='red')
plt.plot(BL, CAA[1] * 100, linewidth=1.5, linestyle='-', label='$T_{2,CAA}$', color='red')
plt.plot(BL, CAA[2] * 100, linewidth=1.5, linestyle=':', label='$T_{3,CAA}$', color='red')

plt.plot(BL, LA[0] * 100, linewidth=1.5, linestyle='-.', label='$T_{1,LA}$', color='blue')
plt.plot(BL, LA[1] * 100, linewidth=1.5, linestyle='-', label='$T_{2,LA}$', color='blue')
plt.plot(BL, LA[2] * 100, linewidth=1.5, linestyle=':', label='$T_{3,LA}$', color='blue')

plt.plot(BL[:len(CDM[0])], CDM[0] * 100, linewidth=1.5, linestyle='-.', label='$T_{1,CDM}$', color='green')
plt.plot(BL[:len(CDM[0])], CDM[1] * 100, linewidth=1.5, linestyle='-', label='$T_{2,CDM}$', color='green')
plt.plot(BL[:len(CDM[0])], CDM[2] * 100, linewidth=1.5, linestyle=':', label='$T_{3,CDM}$', color='green')
# plt.plot([dt2[i] for i in sorted_i], [T2_list[i] for i in sorted_i], linewidth=1.5, linestyle='-', label='Mode 2 - CAA', color='red')
# plt.plot([dt3[i] for i in sorted_i], [T3_list[i] for i in sorted_i], linewidth=1.5, linestyle=':', label='Mode 3 - CAA', color='red')
plt.legend()
plt.xlabel('Blocks', fontsize=15)
plt.ylabel('$(T_{FFT} - T_i)/T_i$ [\%]', fontsize=15)

plt.savefig('conv_blocks.eps')
