# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""
#%% Library imports

import numpy as np
import os
import h5py
import sys
import pathlib
import importlib
from copy import deepcopy  
import pickle 

def reload_modules(): 
    importlib.reload(st)
    importlib.reload(mat)

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat

reload_modules()

#%% Structure parameters 

N1 = np.array([0, 0],dtype=float)
N2 = np.array([0, 3],dtype=float)
N3 = np.array([3, 3],dtype=float)
N4 = np.array([3, 0],dtype=float)

B_b = .2
H_b = .2
H_c = .2*2**(1/3)

CPS = 20
BLOCKS = 30

E = 30e9
NU = 0.0
FY = 20e6
ALPHA = .0

RHO = 2000.

# MAT = mat.Bilinear_Mat(E, NU, FY)
MAT = mat.Plastic_Mat(E, NU, FY)
# MAT.plot_stress_strain()
#%% Building structure

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H_c, RHO, b = B_b, material=MAT)
St.add_beam(N2, N3, BLOCKS, H_b, RHO, b = B_b, material=MAT)
St.add_beam(N3, N4, BLOCKS, H_c, RHO, b = B_b, material=MAT)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

#%% BCs and Forces
        
St.fixNode(N1, [0,1])
St.fixNode(N4, [0,1])

St.loadNode(BLOCKS-1, [0], 1e4)

# Save modes in undeformed configuration
nb_modes = 3
St.save_structure('Plastic_Frame')
St.solve_modal(filename='Step_0_Modal', save=True)
St.plot_modes(nb_modes, scale=10)

n_incr = 50
INCR = 2e-3
STEPS = 3

LIST = [0]

for i in range(n_incr):

    with open(f'Plastic_Frame.pkl', 'rb') as file: 
        St = pickle.load(file)
    
    LIST = np.linspace(LIST[-1], INCR*(i+1), STEPS)
    LIST = LIST.tolist()
    St.solve_dispcontrol(LIST, 0, (BLOCKS-1), 0, tol=1, filename=f'Step_{i+1}_DispControl')
    St.save_structure('Plastic_Frame')

    St.solve_modal(filename=f'Step_{i+1}_Modal', save=True)
    print(f'Natural frequencies for step {i+1}: {np.around(St.eig_vals[:nb_modes],3)}')
    # St.plot_modes(nb_modes, scale=10)
    # St.plot_stresses()
    # St.plot_stiffness()


#%% Plot results

P = np.array([])
U = np.array([])
for i in range(1,n_incr+1): 
    
    results = f'Step_{i}_DispControl.h5'
    
    with h5py.File(results, 'r') as hf:
            
        #Import what you need
        P = np.append(P, hf['P_r_conv'][3*(BLOCKS-1)]/1000)
        U = np.append(U, hf['U_conv'][3*(BLOCKS-1)]*1000)
        

import matplotlib.pyplot as plt 
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.figure(None, figsize=(6,6))
plt.xlabel(r'Top horizontal displacement [mm]', fontsize=18)
plt.ylabel(r'Applied Force $F$ [kN]', fontsize=18)
plt.grid()

plt.plot(U, P, '-o', markerfacecolor='white',color='black',markersize=2,label='HybriDFEM')
plt.legend(fontsize=16)

# %%
