# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import h5py

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "serif",  # Use a serif font
    "font.size": 15  # Adjust font size
})

# %% Plot Pushover
Node = 9
W = 8517.20

# %% Results With Initial Stiffness

a = .15
damp = 0.05

file = f'Tan_B_{a}g_{damp}_NWK_g=0.5_b=0.25.h5'

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_B = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time_B = hf['Time'][:last_conv]

file = f'Tan_C_{a}g_{damp}_NWK_g=0.5_b=0.25.h5'

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_C = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time_C = hf['Time'][:last_conv]

file = f'Tan_D_{a}g_{damp}_NWK_g=0.5_b=0.25.h5'

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_D = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time_D = hf['Time'][:last_conv]

file = f'Mod_Lin_Arch_{a}g_{damp}_NWK_g=0.5_b=0.25.h5'

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_mod = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time_mod = hf['Time'][:last_conv]

# %% Import Lemos results
data = np.loadtxt("fig-11/acne-pp15-s5-h7.txt", skiprows=2)
# Extract columns as separate arrays
t_ls, d_ls = data[:, 0], data[:, 1] * 1000

# data = np.loadtxt("fig-11/acne-pp15-mx5-h7.txt", skiprows=2)
# # Extract columns as separate arrays
# t_mx5, d_mx5 = data[:, 0], data[:, 1]*1000

# data = np.loadtxt("fig-11/acnek-pp15-mx5-h7.txt", skiprows=2)
# # Extract columns as separate arrays
# t_kmx5, d_kmx5 = data[:, 0], data[:, 1]*1000

# %% Plotting
plt.figure(None, (6, 6))

plt.plot(t_ls, d_ls, color='green', label='3DEC Stiff.', linestyle='-.', linewidth=.75)
# plt.plot(t_mx5, d_mx5, color='orange',label='3DEC Max.', linestyle='-.',linewidth=.75)
# plt.plot(t_kmx5, d_kmx5, color='red',label='3DEC Max. mod. stiffness', linestyle='-.',linewidth=.75)
# plt.plot(Time_mod, U_mod, color='purple',label='HDFEM - $1/6$', linestyle='-',linewidth=.75)
plt.plot(Time_B, U_B, color='blue', label=r'Model B', linewidth=.75)
plt.plot(Time_C, U_C, color='red', label=r'Model C', linewidth=.75)
plt.plot(Time_D, U_D, color='orange', label=r'Model D', linewidth=.75)

plt.legend()
plt.grid(True)

plt.xlim((0, 1))
plt.ylim((-0.1, 0.4))

plt.xlabel('Time [s]', fontsize=17)
plt.ylabel('Horizontal displacement [mm]', fontsize=17)
# plt.title('Initial stiffness proportional damping',f
