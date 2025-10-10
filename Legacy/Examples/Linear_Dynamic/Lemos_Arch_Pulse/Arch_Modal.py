import numpy as np
import os
import h5py
import sys
import pathlib
import matplotlib.pyplot as plt

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as cont
import ContactPair as cp
import Surface as surf

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "serif",  # Use a serif font
    "font.size": 15  # Adjust font size
})

save_path = os.path.dirname(os.path.abspath(__file__))

kn = 20e9
ks = 0.4 * kn

B = .5
H = 0.5
R = 7.5 / 2 + H / 2
mu = .7

nb_blocks = 17

rho = 2700

N1 = np.array([0, 0], dtype=float)
x = np.array([.5, 0])
y = np.array([0, 1])

St1 = st.Structure_2D()

H_base = .5
L_base = 1.1 * 2 * R
vertices = np.array([N1, N1, N1, N1])
vertices[0] += L_base * x - H_base * y
vertices[1] += L_base * x
vertices[2] += - L_base * x
vertices[3] += - L_base * x - H_base * y

St1.add_block(vertices, rho, B)

St1.add_arch(N1, 0, np.pi, R, nb_blocks, H, rho, B)
St1.make_nodes()

nb_cps = np.linspace(-1, 1, 3)
nb_cps = nb_cps.tolist()
weights = [1 / 6, 2 / 3, 1 / 6]
St1.make_cfs(False, nb_cps=nb_cps, offset=-1, surface=surf.Surface(kn, ks), weights=weights)
Node = 9
St1.list_cfs[0].change_cps(nb_cp=[-1, 1], offset=-1, surface=surf.Surface(kn, ks), weights=[1] * 2)
St1.list_cfs[1].change_cps(nb_cp=[-1, 1], offset=-1, surface=surf.Surface(kn, ks), weights=[1] * 2)
# St1.loadNode(Node, 1,-0.00001)
St1.plot_structure(scale=0, plot_forces=True, plot_cf=True)
St1.fixNode(0, [0, 1, 2])
for i in range(1, nb_blocks + 1):
    M = St1.list_blocks[i].m
    W = -M * .15 * 10
    St1.loadNode(i, [0], -W)

St1.solve_modal()

# St1.plot_modes(10,scale=50,save=True)

# Modal superposition: 

t_end = .25
nb_steps = 1000
time = np.linspace(0, t_end, nb_steps)

St1.set_damping_properties(0.05, damp_type='STIFF', stiff_type='INIT')
St1.get_C_str()

# Excitation function lambda(t)
w_s = 4 * np.pi
lambda_t = np.sin(w_s * time)

# Express P, M and K in terms of modal coordinates : P_m, M_m, K_m
Phi = St1.eig_modes
M_c = St1.M[np.ix_(St1.dof_free, St1.dof_free)]
K_c = St1.K[np.ix_(St1.dof_free, St1.dof_free)]
M_c = St1.M[np.ix_(St1.dof_free, St1.dof_free)]
M_m = np.transpose(Phi) @ M_c @ Phi
K_m = np.transpose(Phi) @ K_c @ Phi
P_m = np.transpose(Phi) @ St1.P[St1.dof_free]
# Computing damping matrix : 

n_modes = 10

q_t = np.zeros((St1.nb_dof_free, nb_steps))

for i in range(40):
    p = P_m[i]
    k = K_m[i][i]
    w_i = St1.eig_vals[i]
    xsi = St1.damp_coeff[1] * w_i / 2
    if xsi >= 1: xsi = .99

    ratio_w = w_s / w_i
    w_d = w_i * np.sqrt(1 - xsi ** 2)
    C = (p / k) * (1 - ratio_w ** 2) / (((1 - ratio_w ** 2) ** 2) + (2 * xsi * ratio_w) ** 2)
    D = (p / k) * (-2 * xsi * ratio_w) / (((1 - ratio_w ** 2) ** 2) + (2 * xsi * ratio_w) ** 2)
    A = -D
    B = A * xsi * w_i / w_d - C * w_s / w_d
    for j in range(nb_steps):
        t = time[j]
        # Transient response
        # q_t[i,j] = np.e**(-xsi*w_i*t) * (A*np.cos(w_d*t)+B*np.sin(w_d*t))
        # Steady-state response
        # q_t[i,j] = D*np.cos(w_s*t) + C*np.sin(w_s*t)
        # Total response
        q_t[i, j] = np.e ** (-xsi * w_i * t) * (A * np.cos(w_d * t) + B * np.sin(w_d * t)) + C * np.sin(
            w_s * t) + D * np.cos(w_s * t)

U_t = np.zeros((St1.nb_dof_free, nb_steps))

for i in range(St1.nb_dof_free):
    for j in range(St1.nb_dof_free):
        for t in np.arange(0, nb_steps):
            U_t[i, t] += Phi[i, j] * q_t[j, t]

t_end = .75
nb_steps = 3000
time = np.linspace(0, t_end, nb_steps)

U0 = U_t[:, -1]
q0 = np.linalg.inv(Phi) @ U0
Ud0 = (U_t[:, -1] - U_t[:, -2]) / (0.25 / 1000)
qd0 = np.linalg.inv(Phi) @ Ud0

q_t = np.zeros((St1.nb_dof_free, nb_steps))

for i in range(2):
    for t in np.arange(0, nb_steps):
        w_i = St1.eig_vals[i]
        xsi = St1.damp_coeff[1] * w_i / 2
        if xsi >= 1: xsi = .99
        w_d = w_i * np.sqrt(1 - xsi ** 2)

        q_t[i, t] = q0[i] * np.cos(w_d * time[t]) + (qd0[i] + xsi * w_i * q0[i]) * np.sin(w_d * time[t]) / w_d
        q_t[i, t] *= np.exp(-xsi * w_i * time[t])

U_t2 = np.zeros((St1.nb_dof_free, nb_steps))

for i in range(St1.nb_dof_free):
    for j in range(St1.nb_dof_free):
        for t in np.arange(0, nb_steps):
            U_t2[i, t] += Phi[i, j] * q_t[j, t]

time = np.linspace(0, 1, 4000)
U_t = np.append(U_t[3 * Node], U_t2[3 * Node])


def make_plot(U_t, time, saveto):
    plt.figure(None, (6, 6), dpi=400)
    plt.plot(time, U_t * 1000)
    plt.xlim(0, 1)
    data = np.loadtxt("fig-11/acne-pp15-s5-h7.txt", skiprows=2)
    # Extract columns as separate arrays
    t_ls, d_ls = data[:, 0], data[:, 1] * 1000
    plt.plot(t_ls, d_ls, color='green', label='3DEC Stiff.', linestyle=':', linewidth=.75)
    # plt.legend()
    plt.grid(True)

    plt.xlim((0, 1))
    plt.ylim((-0.1, 0.35))
    plt.xlabel('Time [s]', fontsize=17)
    plt.ylabel('Horizontal displacement [mm]', fontsize=17)
    plt.tight_layout()
    plt.savefig(saveto)
    plt.close()


save_path = os.path.dirname(os.path.abspath(__file__)) + '/Frames'

if not pathlib.Path(save_path).exists():
    pathlib.Path(save_path).mkdir(exist_ok=True)
else:
    print('Deleting Frames folder content...')
    for filename in os.listdir(save_path):
        file_path = os.path.join(save_path, filename)
        os.unlink(file_path)

fps = 400

frames_needed = int(1 * fps)
ratio = int(len(time) / frames_needed)

speed = .25

power = int(np.ceil(np.log10(frames_needed)))

print(f'Making {frames_needed} frames for movie...')
for i in range(frames_needed):
    saveto = save_path + f'/Frame{i:0{power}d}.png'
    U = U_t[:i * ratio]
    T = time[:i * ratio]
    make_plot(U, T, saveto)
    # St.plot_structure(scale=1000, plot_cf=False, plot_forces=False, show=False, save=saveto, lims=[[-5.,5.], [-.5, 5]])

# %% Make movie

from moviepy.editor import ImageSequenceClip

frame_pattern = f'Frames/Frame%0{power}d.png'
frame_filenames = [frame_pattern % i for i in range(frames_needed)]
# Create a video clip from the images
clip = ImageSequenceClip(frame_filenames, fps=fps * speed)  # Set the frames per second (fps) for the video

# Write the video file
clip.write_videofile(f'Video_Response.mp4', codec='libx264')

# Remove Frame files from folder
print('Video done... Deleting frames')
for filename in os.listdir(save_path):
    file_path = os.path.join(save_path, filename)
    os.unlink(file_path)

# plt.ylim()
# St1.solve_forcecontrol(20, tol=1e-3,max_iter=100)

# file = f'Results_ForceControl.h5'
# with h5py.File(file, 'r') as hf:

#         U = hf['U_conv'][Node*3+1]*1000

# U_max = np.max(abs(U))

# print(f'Vertical displacement under self-weight is {U_max} mm')
