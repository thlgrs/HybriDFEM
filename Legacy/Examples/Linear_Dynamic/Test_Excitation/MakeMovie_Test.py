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
import cv2

import h5py
import os
import sys
import pathlib
import numpy as np

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

file = 'First_Test_NWK_g=0.5_b=0.25.h5'

with open(f'Beam_Test.pkl', 'rb') as structure:
    St = pickle.load(structure)

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
fps = 50

frames_needed = int(Time[-1] * fps)
ratio = int(len(Time) / frames_needed)

speed = 1

power = int(np.ceil(np.log10(frames_needed)))

print(f'Making {frames_needed} frames for movie...')
for i in range(frames_needed):
    St.U = U_conv[:, i * ratio]
    saveto = save_path + f'/Frame{i:0{power}d}.png'
    St.plot_structure(scale=2, plot_cf=False, plot_forces=False, show=False, save=saveto, lims=[[-.05, 1.1], [-.5, .5]])

# %% Make movie


frame_pattern = f'Frames/Frame%0{power}d.png'
frame_filenames = [frame_pattern % i for i in range(frames_needed)]

# Read the first frame to get the size
first_frame = cv2.imread(frame_filenames[0])
height, width, layers = first_frame.shape

video_filename = 'Video_Test.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_filename, fourcc, fps * speed, (width, height))

for fname in frame_filenames:
    img = cv2.imread(fname)
    video.write(img)

video.release()

# Remove Frame files from folder
print('Video done... Deleting frames')
for filename in os.listdir(save_path):
    file_path = os.path.join(save_path, filename)
    os.unlink(file_path)
