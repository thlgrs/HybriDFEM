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

model = 'C'
stiff_type1 = 'TAN'
stiff_type2 = 'TAN_LG'
# %% Plot Pushover
Node = 9
W = 8517.20

# %% Results With Initial Stiffness

plt.figure(None, (6, 6), dpi=600)
plt.xlabel('Time [s]', fontsize=17)
plt.ylabel('Horizontal displacement [mm]', fontsize=17)
# plt.title('Initial stiffness proportional damping',fontsize=17)


file = stiff_type1 + '_' + model + '_Coulomb_0.15g_0.05_NWK_g=0.5_b=0.25.h5'

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time = hf['Time'][:last_conv]

t_max = Time[np.argmax(U_conv)]
u_max = np.max(U_conv)

print(f'Peak displacement {u_max} at {t_max}s for model C - 5\%')

plt.plot(Time, U_conv, color='blue', label=r'$K^{tan} - 5\%$', linestyle=':')

file = stiff_type1 + '_' + model + '_Coulomb_0.15g_0.01_NWK_g=0.5_b=0.25.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time = hf['Time'][:last_conv]

t_max = Time[np.argmax(U_conv)]
u_max = np.max(U_conv)

print(f'Peak displacement {u_max} at {t_max}s for model C - 1\%')

plt.plot(Time, U_conv, color='green', label=r'$K^{tan} - 1\%$', linestyle=':')

file = stiff_type1 + '_' + model + '_Coulomb_0.15g_0.0_NWK_g=0.5_b=0.25.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time = hf['Time'][:last_conv]

t_max = Time[np.argmax(U_conv)]
u_max = np.max(U_conv)

print(f'Peak displacement {u_max} at {t_max}s for model C - 1\%')

plt.plot(Time, U_conv, color='red', label=r'$K^{tan} - 0\%$', linestyle=':')

file = stiff_type2 + '_' + model + '_Coulomb_0.15g_0.05_NWK_g=0.5_b=0.25.h5'

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time = hf['Time'][:last_conv]

t_max = Time[np.argmax(U_conv)]
u_max = np.max(U_conv)

print(f'Peak displacement {u_max} at {t_max}s for model C - 5\%')

plt.plot(Time, U_conv, color='blue', label=r'$K^{tan,LG} - 5\%$')

file = stiff_type2 + '_' + model + '_Coulomb_0.15g_0.01_NWK_g=0.5_b=0.25.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time = hf['Time'][:last_conv]

t_max = Time[np.argmax(U_conv)]
u_max = np.max(U_conv)

print(f'Peak displacement {u_max} at {t_max}s for model C - 1\%')

plt.plot(Time, U_conv, color='green', label=r'$K^{tan,LG} - 1\%$')

file = stiff_type2 + '_' + model + '_Coulomb_0.15g_0.0_NWK_g=0.5_b=0.25.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time = hf['Time'][:last_conv]

t_max = Time[np.argmax(U_conv)]
u_max = np.max(U_conv)

print(f'Peak displacement {u_max} at {t_max}s for model C - 1\%')

plt.plot(Time, U_conv, color='red', label=r'$K^{tan,LG} - 0\%$')

data = np.loadtxt("fig-12/acn103-pp15-s5-h7.txt", skiprows=2)
# Extract columns as separate arrays
t_ls, d_ls = data[:, 0], data[:, 1] * 1000
plt.plot(t_ls, d_ls, color='orange', label='3DEC, $5\%$', linestyle='-')
print(np.max(d_ls))

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)

plt.xlim((0, 2.5))
plt.ylim((-25, 25))
# plt.savefig('Response_all.eps')
