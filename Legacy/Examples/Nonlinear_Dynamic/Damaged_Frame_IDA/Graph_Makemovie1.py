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
import pandas as pd

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

files = []

a = 5
namedir = 'Dynamic'

for file_name in os.listdir(namedir):

    if file_name.endswith('0.25.h5') and file_name.startswith(f'PGA={a}.0g'):
        files.append(file_name)

Node = 3
for i, file in enumerate(files):
    with h5py.File(namedir + '/' + file, 'r') as hf:
        # Import what you need
        last_conv = hf['Last_conv'][()]
        U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
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
# %%
# %%
power = int(np.ceil(np.log10(frames_needed)))
# print(fps*speed)

print(f'Making {frames_needed} frames for movie...')


def read_accelerogram(filename):
    df = pd.read_csv(filename, sep='\s+', header=1)
    values = df.to_numpy()

    a = values[:, :6]
    a = a.reshape(-1, 1)
    a = a[~np.isnan(a)]

    file = open(filename)
    # get the first line of the file
    line1 = file.readline()
    line2 = file.readline()
    items = line2.split(' ')
    items = np.asarray(items)
    items = items[items != '']
    dt = float(items[1])
    t = np.arange(len(a)) * dt

    return (t, a)


pga = 5

time, acc = read_accelerogram('Earthquakes/NF13')
acc = pga * acc / np.max(abs(acc))

if time[-1] < Time[-1]:
    time = np.append(time, np.linspace(time[-1], Time[-1], 10))
    acc = np.append(acc, np.zeros(10))

dt_new = dt
new_time = np.arange(time[0], Time[-1], dt_new)

from scipy.interpolate import interp1d

interpolator = interp1d(time, acc, kind="linear", fill_value="extrapolate")
new_acc = interpolator(new_time)

print(np.max(U_conv))
print(np.min(U_conv))

for i in range(frames_needed):

    y = U_conv[:ratio * i]
    x = Time[:ratio * i]
    z = new_acc[:ratio * i]

    saveto = save_path + f'/Frame{i:0{power}d}.png'
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), dpi=400)
    ax[0].plot(x, z, color='black', linewidth=.5)
    ax[1].plot(x, y, color='red', linewidth=.5, label='Fixed')
    if len(x) >= 1:
        ax[0].plot(x[-1], z[-1], marker='o', markersize=3, markerfacecolor='white', markeredgecolor='black')
        ax[1].plot(x[-1], y[-1], marker='o', markersize=3, markerfacecolor='white', markeredgecolor='red')

    ax[0].set_xlim(0, 20)
    ax[1].set_xlim(0, 20)
    ax[0].set_ylim(-5.1, 5.1)
    ax[1].set_ylim(-250, 250)
    ax[0].grid(True)
    ax[1].grid(True)
    ax[1].set_ylabel('Out of plane disp. [mm]')
    ax[0].set_ylabel('Ground acc [g]')
    ax[1].set_xlabel('Time [s]')
    # ax[1].legend(loc='upper right')
    plt.savefig(saveto)
    plt.close()
# %% Make movie

from moviepy.editor import ImageSequenceClip

frame_pattern = f'Frames/Frame%0{power}d.png'
frame_filenames = [frame_pattern % i for i in range(frames_needed)]
# Create a video clip from the images
clip = ImageSequenceClip(frame_filenames, fps=fps)  # Set the frames per second (fps) for the video

# Write the video file
clip.write_videofile(f'Graph_Frame_{a}g.mp4', codec='libx264')

# Remove Frame files from folder
print('Video done... Deleting frames')
for filename in os.listdir(save_path):
    file_path = os.path.join(save_path, filename)
    os.unlink(file_path)
