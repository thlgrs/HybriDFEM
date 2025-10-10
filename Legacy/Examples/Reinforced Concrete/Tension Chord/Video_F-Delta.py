import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import pickle
import numpy as np
import pathlib
import sys
import os
import pandas as pd

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st

D = 14e-3
Node = 149

file = f'Structure_d={D * 1000}mm.pkl'

with open(file, 'rb') as f:
    St = pickle.load(f)

file = f'Danilo_TC_d={D * 1000}mm.h5'
with h5py.File(file, 'r') as hf:
    U = hf['U_conv'][:, 1000:]
    P = hf['P_r_conv'][:, 1000:]
    P_o = hf['P_r_conv'][:, :1000]
    U_o = hf['U_conv'][:, :1000]
    # max_P = np.max(P)
    # print(max_P)


def plot_F_Delta(U, P, save=None, compare=False):
    plt.figure(None, figsize=(5, 5), dpi=400)
    try:
        x_max = max(U)
        max_P = max(P)
    except:
        x_max = 1e-5
        max_P = 1e-5

    if compare:
        df = pd.read_excel('plot-data.xlsx', engine='openpyxl')
        print('Coucou')
        # Extract arrays from columns
        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        plt.plot(x - x[0], y, color='grey', linestyle='--', linewidth=1, label='Tension chord')

        # plt.plot(U*1000, P/1000, linewidth=1, color='black',label='HybriDFEM')

    # max_P = np.max(P)
    # print(max(U))
    plt.xlim(0, x_max * 1000)
    plt.ylim(0, 1.1 * max_P / 1000)
    plt.xlabel('Elongation [mm]')
    plt.ylabel('Applied force [kN]')
    plt.grid(True)

    plt.plot(U * 1000, P / 1000, linewidth=.75, color='black')

    plt.savefig(saveto)
    plt.close()


# St.plot_structure(scale=500, plot_cf=False, plot_supp=False)

save_path = os.path.dirname(os.path.abspath(__file__)) + f'/Frames_F_Delta_{D}'

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

for i in range(frames_needed):
    U_new = np.append(U_o[3 * Node], U[3 * Node, :i * ratio])
    P_new = np.append(P_o[3 * Node], P[3 * Node, :i * ratio])
    saveto = save_path + f'/Frame{i:04d}.png'
    compare = i >= frames_needed - 1
    plot_F_Delta(U_new, P_new, save=saveto, compare=compare)

    if i % fps == 0:
        print(f'I created {i} frames already.')

# %% Make movie

from moviepy.editor import ImageSequenceClip

frame_pattern = f'Frames_F_Delta_{D}/Frame%04d.png'
frame_filenames = [frame_pattern % i for i in range(frames_needed)]
# Create a video clip from the images
clip = ImageSequenceClip(frame_filenames, fps=fps)  # Set the frames per second (fps) for the video

# Write the video file
clip.write_videofile(f'F_Delta_d={D * 1000}mm_Part2.mp4', codec='libx264')

# Remove Frame files from folder
print('Video done... Deleting frames')
for filename in os.listdir(save_path):
    file_path = os.path.join(save_path, filename)
    os.unlink(file_path)
