# postprocess_impact.py

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

# --- Load results from HDF5 file
filename = r"W11_1_Impact_CDM_CDM_.h5"

with h5py.File(filename, 'r') as f:
    U = f['U_conv'][:]  # Displacement [ndofs x nsteps]
    V = f['V_conv'][:]  # Velocity
    A = f['A_conv'][:]  # Acceleration
    P = f['P_ref'][:]  # External forces
    Time = f['Time'][:]  # Time vector
    P_r = f['F_conv'][:]  # Internal forces
    try:
        Load = f['Load_Multiplier'][:]
    except:
        Load = None

# --- Define parameters (adjust if geometry changes)
dof_impact_x = 93  # DOF corresponding to Ux of the impactor block
mass_impact = 116  # Mass in kg (obtained during setup)
nsteps = len(Time)

# === PLOTS ===

# 1. Displacement in X
plt.figure()
plt.plot(Time, U[dof_impact_x, :], label='Ux impactor')
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.title("Impactor Displacement vs Time")
plt.grid()
plt.legend()

# 2. Velocity in X
plt.figure()
plt.plot(Time, V[dof_impact_x, :], label='Vx impactor')
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Impactor Velocity vs Time")
plt.grid()
plt.legend()

# 3. Kinetic Energy
Ek = 0.5 * mass_impact * V[dof_impact_x, :] ** 2
plt.figure()
plt.plot(Time, Ek, label='Kinetic Energy')
plt.xlabel("Time [s]")
plt.ylabel("Energy [J]")
plt.title("Impactor Kinetic Energy vs Time")
plt.grid()
plt.legend()

# 4. Contact Force in X
Fx_impact = P_r[dof_impact_x, :]
plt.figure()
plt.plot(Time, Fx_impact, label='Reaction Force Fx')
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.title("Horizontal Contact Force vs Time")
plt.grid()
plt.legend()

# 5. Contact Force with Impact Marker
impact_threshold = 1e-3  # Small threshold to detect contact
impact_start_index = np.argmax(np.abs(Fx_impact) > impact_threshold)
impact_time = Time[impact_start_index]

plt.figure()
plt.plot(Time, Fx_impact, label='Contact Force Fx [N]')
plt.axvline(x=impact_time, color='red', linestyle='--', label=f'Impact starts ≈ {impact_time:.5f} s')
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.title("Horizontal Contact Force vs Time (with impact marker)")
plt.grid()
plt.legend()


# 6. Horizontal Displacement of DOF 10
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


smoothed_deflection = moving_average(U[10, :] * 1000)  # in mm

plt.figure()
plt.plot(Time, U[10, :], label='DOF 10 - Ux', color='tab:brown')
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.title("Horizontal Displacement of DOF 10 vs Time")
plt.grid()
plt.legend()

# === SAVE FIGURES ===

save_dir = r"Figures"
os.makedirs(save_dir, exist_ok=True)

# Switch backend to non-interactive for saving
plt.switch_backend('agg')

# Save each figure
for i, fig_num in enumerate(plt.get_fignums()):
    fig = plt.figure(fig_num)
    filename = f"TIMImpact_Figure_{i + 1}.png"
    full_path = os.path.join(save_dir, filename)
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {full_path}")

# Show figures only if needed
plt.show()  # Uncomment if running interactively

# === ANIMATION SETUP ===
import sys

sys.path.append(
    r"C:\1.MARIANA\4. Research HybridFEM\HybriDFEM - Mariana - Research Intership\HybriDFEM---Mariana\HybriDFEM-3\Objects")
import pickle
import moviepy

print(moviepy.__version__)

from moviepy import ImageSequenceClip

# --- Load original structure
pkl_path = r"Structure.pkl"
with open(pkl_path, 'rb') as f:
    S = pickle.load(f)

# --- Setup for frames
frame_folder = "Frames"
os.makedirs(frame_folder, exist_ok=True)

# --- Delete tests frames if any
for file in os.listdir(frame_folder):
    os.remove(os.path.join(frame_folder, file))

# --- Frame parameters
fps = 10000
speed = 0.005  # 50% slower
frames_needed = int(Time[-1] * fps)
ratio = max(1, len(Time) // frames_needed)
power = len(str(frames_needed))

print(f"Generating {frames_needed} frames...")

# --- Generate frames
for i in range(frames_needed):
    step = i * ratio
    S.U = U[:, step]
    fname = os.path.join(frame_folder, f"Frame{i:0{power}d}.png")
    S.plot_structure(scale=2, plot_cf=False, plot_forces=False, show=False, save=fname, lims=[[-0.5, 0.5], [-0.2, 2.]])
    print(f"Saved frame {i + 1}/{frames_needed}")

# --- Make video
video_path = "Impact_Animation.mp4"
frame_files = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.png')])
clip = ImageSequenceClip(frame_files, fps=int(fps * speed))
clip.write_videofile(video_path)

# --- Cleanup
print("Cleaning up temporary frames...")
for file in os.listdir(frame_folder):
    os.remove(os.path.join(frame_folder, file))

print(f"Animation saved: {video_path}")
