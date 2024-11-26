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

N1 = np.array([0,0])

H_b = .175
L_b = .4
B = 1

kn = 1e7
ks = 1e7

#%% 
RHO = 2000

Blocks_Bed = 20
Blocks_Head = 25

Line1 = []
Line2 = []

for i in range(Blocks_Bed): 
    Line1.append(1.)
    if i==0: 
        Line2.append(.5)
        Line2.append(1.)
    elif i == Blocks_Bed-1: 
        Line2.append(.5)
    else: Line2.append(1.)
    
vertices = np.array([[Blocks_Bed*L_b,-H_b], 
                     [Blocks_Bed*L_b,   0], 
                     [0,0], 
                     [0,-H_b]])
    
PATTERN = []

for i in range(Blocks_Head): 
    if i%2 == 0: PATTERN.append(Line2)
    else: PATTERN.append(Line1)
    
St = st.Structure_2D()

St.add_block(vertices, RHO, b=1)
St.add_wall(N1, L_b, H_b, PATTERN, RHO, b=B, material=None)

St.make_nodes()
St.make_cfs(True, nb_cps=2, contact=ct.Contact(kn, ks), offset=0.0)

#%% BCs and Forces

St.fixNode(0,[0,1,2])

w_s = 20
g = 9.81

A = 0.2*g
t = 10

def excitation(x): return A * np.sin(w_s*x) 
# def excitation(x): 
    # return ug[int(x/dt)]

for i in range(len(St.list_nodes)): 
    W = St.list_blocks[i].m 
    St.loadNode(i, 0, W)


# St.solve_linear()
St.plot_structure(scale=0, plot_cf=False)

# U0 = St.U.copy()

St.set_damping_properties(xsi=0.05, damp_type='RAYLEIGH')

Meth = 'NWK'
St.solve_dyn_linear(t, 5e-4, lmbda=excitation, Meth=Meth)

St.save_structure(filename='F&TL_Wall1')


