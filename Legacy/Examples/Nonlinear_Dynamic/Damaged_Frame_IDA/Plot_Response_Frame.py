# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import h5py
import os
import re

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "serif",  # Use a serif font
    "font.size": 12  # Adjust font size
})

# %% Plot accelerogram
import pandas as pd


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

    return (dt, a)


fig = plt.figure(None, (6, 5), dpi=800)
subfigs = fig.subfigures(2, 1)

dt, lmbda = read_accelerogram('Earthquakes/NF13')
lmbda = np.append(lmbda, np.zeros(2000)) / 100
time = np.arange(len(lmbda)) * dt
pga = 1
print(pga * np.max(abs(lmbda)))
lmbda = pga * lmbda / np.max(abs(lmbda))

ax1 = subfigs[0].subplots()
ax1.plot(time, lmbda, color='black')
# ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Ground acceleration [g]')
# ax1.set_xticklabels([])
ax1.grid(True)
# plt.legend()
ax1.set_xlim(0, 20)
ax1.set_ylim(-1.1, 1.1)

# plt.savefig('earthquake.eps')


# plt.figure(None, figsize=(6,3),dpi=800)
ax2 = subfigs[1].subplots()

file = f'Dynamic/PGA=5.0'
file += 'g_NWK_g=0.5_b=0.25.h5'
Node = 3
with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time = hf['Time'][:last_conv]

ax2.plot(Time, U_conv, color='lightgrey', label=rf'5g')

file = f'Dynamic/PGA=4.0'
file += 'g_NWK_g=0.5_b=0.25.h5'

Node = 3

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time = hf['Time'][:last_conv]
ax2.plot(Time, U_conv, color='grey', label=rf'4g')

file = f'Dynamic/PGA=3.0'
file += 'g_NWK_g=0.5_b=0.25.h5'

Node = 3

with h5py.File(file, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
    Time = hf['Time'][:last_conv]
ax2.plot(Time, U_conv, color='black', label=rf'3g')

ax2.set_ylabel('Displacement [mm]')
ax2.set_xlabel('Time [s]')

ax2.grid(True)
# plt.legend()
ax2.set_xlim(0, 20)
ax2.set_ylim(-1.1, 1.1)
ax2.legend()
ax2.set_xlim(0, 20)
ax2.set_ylim(-250, 250)

plt.savefig('earthquake_resp.eps')

# %% Plot Classical IDA

# accelerations = np.arange(15, 48) / 100
ACC = [0]
PRD = [0]

plt.figure(None, figsize=(4.5, 4.5), dpi=800)
folder_path = 'Dynamic/'
for file in os.listdir(folder_path):

    match = re.search(r"\d+\.\d{1,2}|\d+", file)
    if match:
        number = float(match.group())
        # print(number)
    ACC.append(number)

    with h5py.File(folder_path + file, 'r') as hf:

        # Import what you need
        last_conv = hf['Last_conv'][()]
        U_conv = hf['U_conv'][3 * Node, :last_conv] * 1000
        Time = hf['Time'][:last_conv]

    PRD.append(max(abs(U_conv)))

Node = 8
results = f'Results_DispControl.h5'
with h5py.File(results, 'r') as hf:
    # Import what you need
    P = hf['P_r_conv'][3 * Node]
    U = hf['U_conv'][3 * Node] * 1000

    W = 2000 * 9.81 * .2 * .2 * (3 / 29)

    plt.plot(U, P / W, linestyle='-.', color='darkgrey', label='Pushover')

# print(a, PRD)
plt.xlabel('DM: Peak roof displacement [mm]')
plt.ylabel('IM: Peak ground acceleration [g]')
plt.plot(PRD, ACC, marker='.', color='black', label='IDA')
plt.xlim(0, 500)
plt.ylim(0, 9)
plt.legend()
plt.grid(True)
plt.savefig('classicIDA.eps')

# %% Plot freqency IDA

Fr_reduc = [np.zeros(10)]
ACC = [0]
Periods = []

folder_path = 'Modal/'
for file in os.listdir(folder_path):

    match = re.search(r"\d+\.\d{1,2}|\d+", file)
    if match:
        number = float(match.group())
        print(number)
    ACC.append(number)

    with h5py.File(folder_path + file, 'r') as hf:

        # Import what you need
        if number == .5:
            eig_vals_0 = hf['eig_vals'][()]
            Periods.append(np.pi * 2 / eig_vals_0)
        eig_vals = hf['eig_vals'][()]

    Periods.append(np.pi * 2 / eig_vals)
    Fr_reduc.append(np.ones(10) - eig_vals / eig_vals_0)

plt.figure(None, figsize=(4.5, 4.5), dpi=800)
plt.plot([arr[0] * 100 for arr in Fr_reduc], ACC, marker='.', color='black', label=r'$\omega_1$', linewidth=.75,
         markersize=4)
plt.plot([arr[1] * 100 for arr in Fr_reduc], ACC, marker='x', color='black', label=r'$\omega_2$', linewidth=.75,
         markersize=3)
plt.plot([arr[2] * 100 for arr in Fr_reduc], ACC, marker='^', color='black', label=r'$\omega_3$', linewidth=.75,
         markersize=2)
plt.plot([arr[3] * 100 for arr in Fr_reduc], ACC, marker='D', color='black', label=r'$\omega_4$', linewidth=.75,
         markersize=2)
plt.plot([arr[4] * 100 for arr in Fr_reduc], ACC, marker='s', color='black', label=r'$\omega_5$', linewidth=.75,
         markersize=2)
plt.grid()
plt.xlabel('DM: Frequency reduction [\%]')
plt.ylabel('IM: Peak ground acceleration [g]')
plt.xlim(0, 100)
plt.ylim(0, 9)
plt.legend(loc='lower right')
plt.savefig('modal_ida.eps')

# %% Plot eigenperiods

# %% Plot integral curve
# plt.figure(None, figsize=(6,6))
# y1 = [arr[0]*100 for arr in Fr_reduc]
# y2 = [arr[1]*100 for arr in Fr_reduc]
# y3 = [arr[2]*100 for arr in Fr_reduc]
# y4 = [arr[3]*100 for arr in Fr_reduc]
# x = ACC
# from scipy.integrate import cumulative_trapezoid
# integral_curve1 = cumulative_trapezoid(y1, x, initial=0)
# integral_curve2 = cumulative_trapezoid(y2, x, initial=0)
# integral_curve3 = cumulative_trapezoid(y3, x, initial=0)
# integral_curve4 = cumulative_trapezoid(y4, x, initial=0)

# plt.plot(integral_curve1, ACC, marker='*', color='red',label=r'$\omega_1$')
# plt.plot(integral_curve2, ACC, marker='*', color='blue',label=r'$\omega_2$')
# plt.plot(integral_curve3, ACC, marker='*', color='orange',label=r'$\omega_3$')
# plt.plot(integral_curve4, ACC, marker='*', color='green',label=r'$\omega_4$')

# %% Plot mode shapes


files = ['PGA=5.0g.h5']
folder_path = 'Modal/'
for file in files:
    with h5py.File(folder_path + file, 'r') as hf:
        eig_modes = hf['eig_modes'][()]
        eig_vals = hf['eig_vals'][()]
# print(eig_modes)
import pickle
import pathlib
import sys

print(eig_vals[:5])
# folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
# sys.path.append(str(folder))

# import Structure as st
# import Material as mat

with open('Undamaged_Frame.pkl', 'rb') as file:
    St = pickle.load(file)

for i in range(5):
    St.U[St.dof_free] = eig_modes.T[i]
    St.plot_structure(scale=5, plot_cf=False, plot_forces=False, plot_supp=False, save=f'Mode{i}_5g.eps')
