# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import h5py

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "serif",  # Use a serif font
    "font.size": 12  # Adjust font size
})

# %% Plot Pushover
Node = 9
W = 8517.20

# file1 = 'Coulomb_DispControl.h5'

# with h5py.File(file1, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     P_c = hf['P_r_conv'][3*Node,:last_conv] / W
#     U_c = hf['U_conv'][3*Node,:last_conv]*1000

# file2 = 'Elastic_DispControl.h5'

# with h5py.File(file2, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     P_e = hf['P_r_conv'][3*Node,:last_conv] / W
#     U_e = hf['U_conv'][3*Node,:last_conv]*1000

# plt.figure(None, (6,6))
# plt.grid(True)
# plt.xlabel(r'Control displacement [mm]')
# plt.ylabel(r'Load multiplier [-]')
# plt.plot(U_c,P_c,label='Coulomb - No Tension')
# plt.plot(U_e,P_e,label='Elastic')
# plt.legend()
# plt.xlim((0,40))
# plt.ylim((0, .1))

# #%% Plot Elastic Results from dyn analysis

# file = 'Elastic_NWK_g=0.5_b=0.25.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]

# plt.figure(None, (6,6))
# plt.xlabel('Time')
# plt.ylabel('Horizontal displacement')
# plt.plot(Time, U_conv, color='black')
# plt.grid(True)
# plt.xlim(0,1)
# # plt.ylim(-0.1, 0.4)

# %% Plot Plastic Results from dyn analysis
plt.figure(None, (4.5, 4.5))
plt.xlabel('Time [s]')
plt.ylabel('Horizontal displacement [mm]')

data = np.loadtxt("fig-12/acn103-pp15-s5-h7.txt", skiprows=2)
# Extract columns as separate arrays
t_ls, d_ls = data[:, 0], data[:, 1] * 1000
plt.plot(t_ls, d_ls, color='black', label='Lemos \& Sarhosis, 2023', linestyle='-.')

file = 'NoTension_0.15g_0.05_NWK_g=0.5_b=0.25.h5'

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time = hf['Time'][:last_conv]

plt.plot(Time, U_conv, color='black', label=r'$\mathbf{K}^{tan} - 5\%$')

file = 'NoTension_0.15g_0.04_NWK_g=0.5_b=0.25.h5'

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time = hf['Time'][:last_conv]

plt.plot(Time, U_conv, color='black', label=r'$\mathbf{K}^{tan} - 4\%$ ')

file = 'Init_NoTension_0.15g_0.05_NWK_g=0.5_b=0.25.h5'

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time = hf['Time'][:last_conv]

plt.plot(Time, U_conv, color='black', label=r'$\mathbf{K}_0$ - CAA', linestyle=':')

# file = 'NoTension_0.15g_0.05_CDM_.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]

plt.plot(Time, U_conv, color='black', label=r'$\mathbf{K}^{tan}$ - CDM', linestyle='--')

# file = 'NoTension_0.15g_0.04_NWK_g=0.5_b=0.25.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]

# plt.plot(Time, U_conv, color='blue',label=r'HybriDFEM_$\xi=0.04$')

# file = 'NoTension_0.15g_0.03_NWK_g=0.5_b=0.25.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]

# plt.plot(Time, U_conv, color='green',label=r'HybriDFEM_$\xi=0.03$')

# file = 'NoTension_0.15g_0.02_NWK_g=0.5_b=0.25.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]

# plt.plot(Time, U_conv, color='orange',label=r'HybriDFEM_$\xi=0.02$')


plt.grid(True)
plt.legend(fontsize=11)
plt.xlim(0, 2.5)
plt.ylim(-20, 25)
plt.savefig('Arch_Pulse.eps')

# %% Plot Plastic Results from dyn analysis
# fig, ax = plt.subplots(2,1, figsize=(8,8))

# vlocy = np.loadtxt("data-eq/eq23s1.txt", skiprows=2)

# nb_points = 3005
# dt = 5e-3
# t_end = dt * nb_points
# time = np.arange(nb_points-1)*dt
# # print(data, time)
# acc = np.zeros(nb_points-1)

# for i in np.arange(nb_points-1): 
#     acc[i] = (vlocy[i+1] - vlocy[i]) / dt

# ax[0].plot(time, acc)
# ax[0].set_title('Ground acceleration')
# ax[0].set_xlabel('Time [s]')
# ax[0].set_ylabel('Ground acceleration [g]')
# ax[0].grid(True)
# ax[0].set_xlim(0,16)

# data = np.loadtxt("fig-17/acn103-eqaxg20-s5-h5.txt", skiprows=2)
# # Extract columns as separate arrays
# t_ls, d_ls = data[:, 0], data[:, 1]*1000
# ax[1].plot(t_ls, d_ls, color='red',label='L\&S, 2023 - Stiffness')

# data = np.loadtxt("fig-17/acn103-eqaxg20-mx5-h5.txt", skiprows=2)
# # Extract columns as separate arrays
# t_ls, d_ls = data[:, 0], data[:, 1]*1000
# ax[1].plot(t_ls, d_ls, color='green',label='L\&S, 2023 - Maxwell')

# data = np.loadtxt("fig-17/acnk103-eqaxg20-mxk5-h5.txt", skiprows=2)
# # Extract columns as separate arrays
# t_ls, d_ls = data[:, 0], data[:, 1]*1000
# ax[1].plot(t_ls, d_ls, color='orange',label='L\&S, 2023 - Maxwell Mod.')


# file = 'NoTension_EQ_0.2g_NWK_g=0.5_b=0.25.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]

# ax[1].plot(Time, U_conv, color='blue',label='HybriDFEM')

# ax[1].grid(True)
# ax[1].legend()
# ax[1].set_xlim(0,16)
# ax[1].set_ylim(-10, 20)
# ax[1].set_xlabel('Time[s]')
# ax[1].set_ylabel('Displacement [mm]')

# plt.tight_layout()

# #%% Plot Plastic Results from dyn analysis
# plt.figure(None, (8,4))
# plt.xlabel('Time [s]')
# plt.ylabel('Horizontal displacement [mm]')

# data = np.loadtxt("fig-16/acn103-eqaxg10-s5-h5.txt", skiprows=2)
# # Extract columns as separate arrays
# t_ls, d_ls = data[:, 0], data[:, 1]*1000
# plt.plot(t_ls, d_ls, color='red',label='L\&S, 2023 - Stiffness')

# data = np.loadtxt("fig-16/acn103-eqaxg10-mx5-h5.txt", skiprows=2)
# # Extract columns as separate arrays
# t_ls, d_ls = data[:, 0], data[:, 1]*1000
# # plt.plot(t_ls, d_ls, color='green',label='L&S, 2023 - Maxwell')

# data = np.loadtxt("fig-16/acnk103-eqaxg10-mxk5-h5.txt", skiprows=2)
# # Extract columns as separate arrays
# t_ls, d_ls = data[:, 0], data[:, 1]*1000
# # plt.plot(t_ls, d_ls, color='orange',label='L&S, 2023 - Maxwell Mod.')


# file = 'NoTension_EQ_0.1g_NWK_g=0.5_b=0.25.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]

# plt.plot(Time, U_conv, color='blue',label='HybriDFEM')

# plt.grid(True)
# plt.legend()
# # plt.xlim(0,1)
# # plt.ylim(-.1, .1)
