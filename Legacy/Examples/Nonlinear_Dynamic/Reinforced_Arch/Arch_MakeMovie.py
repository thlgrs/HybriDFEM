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

files = []

t_p = 1
a = 0.16

with open(f'Dimitri_Arch.pkl', 'rb') as file:
    St = pickle.load(file)

for file_name in os.listdir():

    if file_name.endswith('0.42.h5') and file_name.startswith(f't_p={t_p}_a={a}'):
        files.append(file_name)

for i, file in enumerate(files):
    with h5py.File(file, 'r') as hf:
        # Import what you need
        last_conv = hf['Last_conv'][()]
        U_conv = hf['U_conv'][:last_conv]
        Time = hf['Time'][:last_conv]

save_path = os.path.dirname(os.path.abspath(__file__)) + '/Frames'

if not pathlib.Path(save_path).exists():
    pathlib.Path(save_path).mkdir(exist_ok=True)
else:
    print('Deleting Frames folder content...')
    for filename in os.listdir(save_path):
        file_path = os.path.join(save_path, filename)
        os.unlink(file_path)

dt = Time[1] - Time[0]
fps = 100

frames_needed = int(Time[-1] * fps)
ratio = int(len(Time) / frames_needed)

power = int(np.ceil(np.log10(frames_needed)))

print(f'Making {frames_needed} frames for movie...')
for i in range(frames_needed):
    St.U = U_conv[:, i * ratio]
    saveto = save_path + f'/Frame{i:0{power}d}.png'
    St.plot_structure(scale=1, plot_cf=False, plot_forces=False, show=False, save=saveto,
                      lims=[[-20., 20.], [-2.5, 35]])

# %% Make movie

from moviepy.editor import ImageSequenceClip

frame_pattern = f'Frames/Frame%0{power}d.png'
frame_filenames = [frame_pattern % i for i in range(frames_needed)]
# Create a video clip from the images
clip = ImageSequenceClip(frame_filenames, fps=fps)  # Set the frames per second (fps) for the video

# Write the video file
clip.write_videofile(f'Video_Rocking_{t_p}_{a}.mp4', codec='libx264')

# Remove Frame files from folder
print('Video done... Deleting frames')
for filename in os.listdir(save_path):
    file_path = os.path.join(save_path, filename)
    os.unlink(file_path)
