# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:52:48 2024

@author: ibouckaert
"""

# %% Libraries imports
import matplotlib as mpl

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

import matplotlib.pyplot as plt
from scipy import linalg
import pickle

import h5py
import os
import sys
import pathlib
import numpy as np

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

with open(f'Wallet.pkl', 'rb') as file:
    St = pickle.load(file)

file = 'Wall6_rb=0.0_psi=0.01.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][-3, :last_conv] * 1000
    P_conv = hf['P_r_conv'][-3, :last_conv]

save_path = os.path.dirname(os.path.abspath(__file__)) + '/Frames_FD'

if not pathlib.Path(save_path).exists():
    pathlib.Path(save_path).mkdir(exist_ok=True)
else:
    print('Deleting Frames folder content...')
    for filename in os.listdir(save_path):
        file_path = os.path.join(save_path, filename)
        os.unlink(file_path)

T_tot = 15
fps = 50

frames_needed = T_tot * fps
ratio = int(U_conv.shape[0] / frames_needed)

power = int(np.ceil(np.log10(frames_needed)))

print(f'Making {frames_needed} frames for movie...')
for i in range(frames_needed):

    plt.figure(None, (5, 5), dpi=400)
    plt.plot(U_conv[:i * ratio], P_conv[:i * ratio] / 350, linewidth=1, color='black')
    if i == 0:
        x_max = 1e-9
        y_max = 1e-9
    else:
        x_max = max(U_conv[i * ratio], 1e-9)
        y_max = max(P_conv[i * ratio] / 350, 1e-9)

    plt.xlabel('Control displacement [mm]')
    plt.ylabel('Load multiplier [-]')
    plt.xlim((0, 1.1 * x_max))
    plt.ylim((0, 1.1 * y_max))
    plt.grid(True)
    saveto = save_path + f'/Frame{i:0{power}d}.png'
    plt.savefig(saveto)
    plt.close()

# %% Make movie

from moviepy.editor import ImageSequenceClip

frame_pattern = f'Frames_FD/Frame%0{power}d.png'
frame_filenames = [frame_pattern % i for i in range(frames_needed)]
# Create a video clip from the images
clip = ImageSequenceClip(frame_filenames, fps=fps)  # Set the frames per second (fps) for the video

# Write the video file
clip.write_videofile(f'Wall6_Video_FD.mp4', codec='libx264')

# Remove Frame files from folder
print('Video done... Deleting frames')
for filename in os.listdir(save_path):
    file_path = os.path.join(save_path, filename)
    os.unlink(file_path)
