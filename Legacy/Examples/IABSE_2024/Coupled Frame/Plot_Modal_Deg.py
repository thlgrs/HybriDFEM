import matplotlib.pyplot as plt
import h5py
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 13

with h5py.File('Results_Modal_Deg.h5', 'r') as hf:
    P = np.array(hf['P'])
    U = np.array(hf['U'])
    w = np.array(hf['w'])

plt.figure(None, figsize=(6, 6))
plt.xlabel(r'Top horizontal displacement [mm]')
plt.ylabel(r'$F / F_{max}$')
plt.grid()
# plt.xlim([0, 100])
# plt.ylim([0, 1.05])

plt.plot(U, P, color='black', linewidth=.75, label=r'$F-\Delta$')
plt.legend(fontsize=13)

plt.savefig('Freqs_Force_Disp.eps')

# %% Plot evolution of frequencies
plt.figure(None, figsize=(6, 6))
nb_steps = w.shape[1]
steps = np.arange(1, nb_steps + 1)

print(w.shape)

for i in range(10):
    plt.plot(steps, w[i, :] / w[i, 0], label=f'$\omega_{{{i + 1}}}$')

plt.ylabel('Normalized frequency [rad/s]')
plt.xlabel('Loading/Unloading cycle')
plt.xlim([1, nb_steps])
plt.ylim([0, 1.05])
plt.grid()
plt.legend()
plt.xticks(steps)
# %%

# %% Plot evolution of frequencies
plt.figure(None, figsize=(6, 6))
nb_steps = w.shape[1]
steps = np.arange(1, nb_steps + 1)

print(w.shape)

for i in range(10):
    plt.plot(steps, w[i, :], label=f'$\omega_{{{i + 1}}}$')

plt.ylabel('Frequency [rad/s]')
plt.xlabel('Loading/Unloading cycle')
plt.xlim([1, nb_steps])

plt.grid()
plt.legend()
plt.xticks(steps)
# %%
