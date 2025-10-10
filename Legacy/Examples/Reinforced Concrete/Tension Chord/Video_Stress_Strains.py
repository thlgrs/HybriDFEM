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
    U = hf['U_conv'][:, 1000:]
    P = hf['P_r_conv'][:, 1000:]

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


def plot_stresses_and_strains(save=None):
    sigma_c, eps_c, x_c = St.get_stresses(angle=np.pi / 2, tag='CTC')
    sigma_s, eps_s, x_s = St.get_stresses(angle=np.pi / 2, tag='STC')
    mask = eps_c > eps_c_max

    x = x_c[mask].tolist()
    x_end = x + [1.26]
    x_start = [0] + x
    eps_c[mask] *= 0

    # x_c = x_c[eps_c <= eps_c_max*1.01]
    # sigma_c = sigma_c[eps_c <= eps_c_max*1.01]
    # eps_c = eps_c[eps_c <= eps_c_max*1.01]

    fig, axs = plt.subplots(5, 1, figsize=(6, 6), dpi=400)

    # axs[1].set_ylabel(r'Steel Strain $[\%]$')
    axs[1].set_xlim(0, L)
    lim_eps_s = max(eps_y, np.max(eps_s))
    axs[1].set_ylim(0, 110 * lim_eps_s)

    axs[1].plot(x_c * 1000, eps_c * 100, color='red', label=r'$\varepsilon_c$', linewidth=0.1, marker='.', markersize=1)
    axs[1].plot(x_s * 1000, eps_s * 100, color='blue', label=r'$\varepsilon_s$', linewidth=0.1, marker='.',
                markersize=1)
    # axs[1].axhline(y=eps_c_max*100, color='grey', label=r'$\varepsilon_{c, max}$', linewidth=.5, marker=None)
    axs[1].axhline(y=eps_y * 100, color='grey', label=r'$\varepsilon_{y}$', linewidth=.5, marker=None, linestyle=':')
    axs[1].set_xticks([])
    # axs[1].legend(loc='center left',bbox_to_anchor=(1, 0.5))

    # axs[0].set_ylabel(r'Concrete Strain $[\%]$')
    axs[0].set_xlim(0, L)
    axs[0].set_ylim(0, eps_c_max * 110)
    # axs[0].set_ylabel(r'Concrete strain [\%]',fontsize=8, labelpad=20)
    # axs[1].set_ylabel(r'Steel strain [\%]',fontsize=8, labelpad=20)

    axs[0].plot(x_c * 1000, eps_c * 100, color='red', label=r'$\varepsilon_c$', linewidth=0.1, marker='.', markersize=1)
    axs[0].plot(x_s * 1000, eps_s * 100, color='blue', label=r'$\varepsilon_s$', linewidth=0.1, marker='.',
                markersize=1)
    axs[0].axhline(y=eps_c_max * 100, color='grey', label=r'$\varepsilon_{c, max}$', linewidth=.5, marker=None,
                   linestyle=':')
    axs[0].set_xticks([])
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # axs[2].set_ylabel('Concrete Stress [MPa]')
    axs[2].set_xlim(0, L)
    axs[2].set_ylim(-.01, 3.5)
    # axs[2].set_ylabel('Concrete stress [MPa]',fontsize=8)

    axs[0].set_xticks([200, 400, 600, 800, 1000, 1200], [''] * 6)
    axs[1].set_xticks([200, 400, 600, 800, 1000, 1200], [''] * 6)
    axs[2].set_xticks([200, 400, 600, 800, 1000, 1200], [''] * 6)
    axs[3].set_xticks([200, 400, 600, 800, 1000, 1200], [''] * 6)
    axs[0].grid(True, linewidth=.3, linestyle='dotted')
    axs[1].grid(True, linewidth=.3, linestyle='dotted')
    axs[2].grid(True, linewidth=.3, linestyle='dotted')
    axs[3].grid(True, linewidth=.3, linestyle='dotted')
    axs[2].plot(x_c * 1000, sigma_c / 1e6, color='red', label=r'$\sigma_c$', linewidth=0.1, marker='.', markersize=1)
    axs[2].axhline(y=FCT / 1e6, color='grey', label=r'$f_{c,t}$', linewidth=.5, marker=None, linestyle=':')
    axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # axs[3].set_xlabel('Position [mm]')
    # axs[3].set_ylabel('Steel Stress [MPa]')
    axs[3].set_xlim(0, L)
    # axs[3].set_ylabel('Steel stress [MPa]',fontsize=8, labelpad=20)
    sig_max = np.max(sigma_s)
    sig_min = np.min(sigma_s)
    axs[3].set_ylim(sig_min * 0.9 / 1e6, sig_max * 1.1 / 1e6)

    axs[3].plot(x_s * 1000, sigma_s / 1e6, color='blue', label=r'$\sigma_s$', linewidth=0.1, marker='.', markersize=1)
    axs[3].axhline(y=FY / 1e6, color='grey', label=r'$f_{y}$', linewidth=.5, marker=None, linestyle=':')
    axs[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[4].set_xlabel('Position [mm]')
    # axs[4].set_ylabel('Crack opening [mm]',fontsize=8)
    axs[4].set_xlim(0, L)
    sigma_c, eps_c, x_c = St.get_stresses(angle=np.pi / 2, tag='CTC')
    mask = eps_c > eps_c_max
    try:
        max_eps = max(1e3 * eps_c[mask] * 1e-2)
    except:
        max_eps = 0
    axs[4].set_ylim(0, max(1, max_eps * 1.1))
    axs[4].grid(True, linewidth=.3, linestyle='dotted')
    # print(mask)
    # print(x_c[mask],eps_c[mask]*1e-2)
    axs[4].bar(x_c[mask] * 1e3, 1e3 * eps_c[mask] * 1e-2, width=10, color='red')  #
    plt.tight_layout()
    plt.savefig(saveto)
    plt.close()


# St.plot_structure(scale=500, plot_cf=False, plot_supp=False)

save_path = os.path.dirname(os.path.abspath(__file__)) + f'/Frames_{D}'

if not pathlib.Path(save_path).exists():
    pathlib.Path(save_path).mkdir(exist_ok=True)
else:
    print('Deleting Frames folder content...')
    for filename in os.listdir(save_path):
        file_path = os.path.join(save_path, filename)
        os.unlink(file_path)

fps = 50
time = 15

frames_needed = int(time * fps)
ratio = int(len(U[0]) / frames_needed)

print(f'Making {frames_needed} frames for movie...')

for cf in St.list_cfs:
    if cf.bl_A.material.tag == 'CTC' and cf.bl_B.material.tag == 'CTC':
        for cp in cf.cps:
            cp.sp1.law.cracked = False
            cp.sp2.law.cracked = False

for i in range(frames_needed):
    St.U = U[:, i * ratio]
    St.get_P_r()
    saveto = save_path + f'/Frame{i:04d}.png'

    plot_stresses_and_strains(save=saveto)

    if i % fps == 0:
        print(f'I created {i} frames already.')

# %% Make movie

from moviepy.editor import ImageSequenceClip

frame_pattern = f'Frames_{D}/Frame%04d.png'
frame_filenames = [frame_pattern % i for i in range(frames_needed)]
# Create a video clip from the images
clip = ImageSequenceClip(frame_filenames, fps=fps)  # Set the frames per second (fps) for the video

# Write the video file
clip.write_videofile(f'Video_d={D * 1000}mm_new_part2.mp4', codec='libx264')

# %%Remove Frame files from folder
print('Video done... Deleting frames')
for filename in os.listdir(save_path):
    file_path = os.path.join(save_path, filename)
    os.unlink(file_path)
