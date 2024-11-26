#%% Libraries imports
import matplotlib as mpl

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

import matplotlib.pyplot as plt
from scipy import linalg

import h5py
import os
import sys
import pathlib
import numpy as np

# Folder to access the HybriDFEM files
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

plt.figure(figsize=(5,5),dpi=800)
   
plt.xlim([0, 10])
# plt.ylim([-250, 250])
plt.title(r'2DoF System with excitation')
plt.xlabel(r'Time [s]')
plt.ylabel(r'Displacement[mm]')

# styles = ['dashed', 'dotted', 'dashdot', ':']

files = []

for file_name in os.listdir():
    
    if file_name.endswith('.h5') and file_name.startswith('_'):
        files.append(file_name)

for i, file in enumerate(files): 
    
    with h5py.File(file, 'r') as hf:
        
        #Import what you need
        U1 = hf['U_conv'][3]*1000
        U2 = hf['U_conv'][6]*1000
        Time = hf['Time'][:]
    
    plt.plot(Time, U1, label=r'$U_1$ - CDM', linewidth=1)
    plt.plot(Time, U2, label=r'$U_2$ - CDM', linewidth=1)
    
#%% Analytical solution
K = 200e3 * np.array([[2, -1],[-1, 1]])
M = 2000 * np.array([[1, 0],[0, 1]])
P = 1.66e4 * np.array([1, 0])

(w, Phi) = linalg.eig(K,M)
w = np.real(np.sqrt(w))    

Mm = Phi.T @ M @ Phi
Km = Phi.T @ K @ Phi
Pm = Phi.T @ P

n_steps = 500
time = np.linspace(0,10,n_steps)
q = np.zeros((n_steps, 2))
U = np.zeros((n_steps, 2))
w_s = 10

for i in range(2): 
    
    p = Pm[i]
    k = Km[i][i]
    w_i = np.real(w[i])   
    xsi = 0.05
    w_d = w_i * np.sqrt(1-xsi**2)
    ratio_w = w_s/w_i
    C = (p/k) * (1-ratio_w**2) / (((1-ratio_w**2)**2)+(2*xsi*ratio_w)**2)
    D = (p/k) * (-2*xsi*ratio_w) / (((1-ratio_w**2)**2)+(2*xsi*ratio_w)**2)
    A = -D
    B = A*xsi*w_i / w_d - C*w_s/w_d
    
    for j,t in enumerate(time): 
        q[j,i] = np.e**(-xsi*w_i*t) * (A*np.cos(w_d*t)+B*np.sin(w_d*t)) + C*np.sin(w_s*t) + D*np.cos(w_s*t) 
        
for i in range(2):
    for j in range(2):
        for k,t in enumerate(time):
            U[k,i] += Phi[i,j] * q[k,j]
            
plt.plot(time, U[:,0]*1000, label='$U_1$ - Modal sup.', linewidth=.5, linestyle='dashed')
plt.plot(time, U[:,1]*1000, label='$U_2$ - Modal sup.', linewidth=.5, linestyle='dashed')
    

plt.legend()
plt.grid(True)