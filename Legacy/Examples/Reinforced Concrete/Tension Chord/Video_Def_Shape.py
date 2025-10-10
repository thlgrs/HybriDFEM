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
    U = hf['U_conv'][:, :1000]
    P = hf['P_r_conv'][:, :1000]

save_path = os.path.dirname(os.path.abspath(__file__)) + f'/Deformed_{int(D * 1000)}'

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
    St.U = U[:, i * ratio]
    St.get_P_r()
    saveto = save_path + f'/Frame{i:04d}.png'

    St.plot_structure(scale=1, plot_cf=False, plot_supp=False, plot_forces=True, lims=[[-.1, 2], [-D, D + .3]],
                      show=False, save=saveto)

    if i % fps == 0:
        print(f'I created {i} frames already.')

# %% Make movie

from moviepy.editor import ImageSequenceClip

frame_pattern = f'Deformed_{int(D * 1000)}/Frame%04d.png'
frame_filenames = [frame_pattern % i for i in range(frames_needed)]
# Create a video clip from the images
clip = ImageSequenceClip(frame_filenames, fps=fps)  # Set the frames per second (fps) for the video

# Write the video file
clip.write_videofile(f'Deformed_{int(D * 1000)}_Part1.mp4', codec='libx264')

# %%Remove Frame files from folder
print('Video done... Deleting frames')
for filename in os.listdir(save_path):
    file_path = os.path.join(save_path, filename)
    os.unlink(file_path)
