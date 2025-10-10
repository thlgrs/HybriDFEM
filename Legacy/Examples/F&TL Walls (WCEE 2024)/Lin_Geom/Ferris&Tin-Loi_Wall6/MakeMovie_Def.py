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
    U_conv = hf['U_conv'][:last_conv]

save_path = os.path.dirname(os.path.abspath(__file__)) + '/Frames_Def'

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
ratio = int(U_conv.shape[1] / frames_needed)

power = int(np.ceil(np.log10(frames_needed)))

print(f'Making {frames_needed} frames for movie...')
for i in range(frames_needed):
    St.U = U_conv[:, i * ratio]
    saveto = save_path + f'/Frame{i:0{power}d}.png'
    St.plot_structure(scale=200, plot_cf=False, plot_forces=False, plot_supp=False, show=False, save=saveto,
                      lims=[[-.1, 3.5], [-.2, 4]])

# %% Make movie

from moviepy.editor import ImageSequenceClip

frame_pattern = f'Frames_Def/Frame%0{power}d.png'
frame_filenames = [frame_pattern % i for i in range(frames_needed)]
# Create a video clip from the images
clip = ImageSequenceClip(frame_filenames, fps=fps)  # Set the frames per second (fps) for the video

# Write the video file
clip.write_videofile(f'Wall6_Video_Def.mp4', codec='libx264')

# Remove Frame files from folder
print('Video done... Deleting frames')
for filename in os.listdir(save_path):
    file_path = os.path.join(save_path, filename)
    os.unlink(file_path)
