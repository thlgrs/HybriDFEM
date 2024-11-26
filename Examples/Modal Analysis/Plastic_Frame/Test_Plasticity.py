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
N2 = np.array([3, 0],dtype=float)

B = .2
H = .5

CPS = 25
BLOCKS = 25

E = 30e9
NU = 0.0
FY = 20e6

RHO = 2000.

MAT = mat.Plastic_Mat(E, NU, FY)
# MAT = mat.Bilinear_Mat(E, NU, FY)
# MAT.plot_stress_strain()
#%% Building structure

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, RHO, b=B, material=MAT)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

#%% BCs and Forces
        
St.fixNode(N1, [0,1,2])

Node = St.get_node_id(N2)
# print(Node)
DOF = 1

St.loadNode(St.get_node_id(N2), 1, -1000)

St.plot_structure(scale=0, plot_cf=False)

# St.solve_modal()
N = 10
INCR = -40e-3

# St.plot_modes(12, scale=10,save=False)
LIST = np.linspace(0,INCR,N)
LIST = np.append(LIST, np.linspace(LIST[-1],-INCR,2*N))
LIST = np.append(LIST, np.linspace(LIST[-1],2*INCR,3*N))
# LIST*=10
LIST = LIST.tolist()
# St.solve_dispcontrol(N, INCR, Node, DOF, tol=1)
St.solve_dispcontrol(LIST, 0, Node, DOF, tol=1)


#%% Plot results

results = f'Results_ForceControl.h5'
    
with h5py.File(results, 'r') as hf:
        
    #Import what you need
    P = -hf['P_r_conv'][3*St.get_node_id(N2) + 1]/1000
    U = -hf['U_conv'][3*Node+DOF]*1000
    
    last_conv = hf['Last_conv'][()]
    last_def = hf['U_conv'][:,last_conv]

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
St.U = last_def
St.plot_structure(scale=10, plot_cf=False)

# %%
