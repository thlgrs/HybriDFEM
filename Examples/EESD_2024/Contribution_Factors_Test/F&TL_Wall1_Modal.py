# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""

import numpy as np
import os
import h5py
import sys
import pathlib
import importlib

def reload_modules():
    importlib.reload(st)
    importlib.reload(surf)
    importlib.reload(ct)

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Surface as surf
import Contact as ct

reload_modules()

N1 = np.array([0,0], dtype=float)

H_b = 1
L_b = 1
B = 1

kn = 1
ks = 1

#%% 
RHO = 1

St = st.Structure_2D()

for i in np.arange(6): 
    
    
    vertices = np.array([N1, N1, N1, N1])
    vertices += np.array([[0, i*H_b], 
                          [0, i*H_b], 
                          [0, i*H_b], 
                          [0, i*H_b]], dtype=float)
    
    vertices += np.array([[ L_b/2, -H_b/2], 
                          [ L_b/2,  H_b/2], 
                          [-L_b/2,  H_b/2], 
                          [-L_b/2, -H_b/2]])
    
    St.add_block(vertices, RHO, b=1)

St.make_nodes()
St.make_cfs(True, nb_cps=2, contact=ct.Contact(kn, ks), offset=0.0)

St.plot_structure(scale=0, plot_cf=False, plot_forces=False)
#%% BCs and Forces

Total_mass = 0

St.fixNode(0,[0,1,2])

for i in range(1,len(St.list_blocks)): 
    St.fixNode(i, [1,2])
    # St.loadNode(i, 0, W)
    # Total_mass += W

St.loadNode(5,0,2)
St.loadNode(4,0,-1) 
print(f'Total mass: {Total_mass} kg')
#%% Solving the eigenvalue problem
St.solve_modal()

#%% Plotting results 
# St.plot_modes(5, scale=1)

print(np.around(St.eig_vals,3))

#%% Modal contribution factors for horizontal displacemnt of top right corner

nb_modal_contributions = 5
St.get_P_r()
St.get_K_str0()

sum_contr = 0

U_ref = np.linalg.solve(St.K0[np.ix_(St.dof_free, St.dof_free)], St.P[St.dof_free])
print(f'Reference corner displacement: {U_ref[-1]*1000} mm')

for i in range(nb_modal_contributions): 
    M_i = St.eig_modes[:,i].T @ St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:,i]
    G_i = St.eig_modes[:,i].T @ St.P[St.dof_free] / M_i
    P_ref_i = G_i * St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:,i]
    
    U_i = np.linalg.solve(St.K0[np.ix_(St.dof_free, St.dof_free)], P_ref_i)
    
    print(f'Modal contribution {i+1} for corner disp.: {np.around(U_i[-1]*100/U_ref[-1],3)}%') 
    sum_contr += U_i[-1]/U_ref[-1]
    print(f'Sum of modal contributions for control disp.: {np.around(sum_contr*100,3)}%')

#%% Modal contribution factors for base shear 
St.U[St.dof_free] = U_ref.copy()
St.get_P_r()
V_ref = St.P_r[0]
print(f'Reference base shear: {V_ref/1000} kN')
sum_contr = 0

for i in range(nb_modal_contributions): 
    M_i = St.eig_modes[:,i].T @ St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:,i]
    G_i = St.eig_modes[:,i].T @ St.P[St.dof_free] / M_i
    P_ref_i = G_i * St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:,i]
    U_i = np.linalg.solve(St.K0[np.ix_(St.dof_free, St.dof_free)], P_ref_i)
    St.U[St.dof_free] = U_i.copy()
    St.get_P_r()
    V_i = St.P_r[0]
    
    print(f'Modal contribution {i+1} for base shear: {np.around(V_i*100/V_ref,3)}%') 
    sum_contr += V_i/V_ref
    print(f'Sum of modal contributions for base shear: {np.around(sum_contr*100,3)}%')
    
# Compute damping ratios

St.set_damping_properties(xsi=0.01, damp_type='RAYLEIGH')
St.get_C_str()
St.solve_modal(nb_modal_contributions)
#%% 

for i in range(nb_modal_contributions): 
    
    M_i = St.eig_modes[:,i].T @ St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:,i]
    C_i = St.eig_modes[:,i].T @ St.C[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:,i]
    
    ksi_i = C_i / (2 * St.eig_vals[i] * M_i)    
    print(f'Damping ratio for mode {i+1}: {np.around(ksi_i,3)}')
    
