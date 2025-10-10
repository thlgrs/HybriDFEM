# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""
# %% Library imports

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
    importlib.reload(ct)
    importlib.reload(surf)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat
import Contact as ct
import Surface as surf

reload_modules()

# %% Structure parameters

N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, 3], dtype=float)
N3 = np.array([0, 6], dtype=float)
N4 = np.array([3, 6], dtype=float)
N5 = np.array([3, 3], dtype=float)
N6 = np.array([3, 0], dtype=float)

B = .2
H = .2

BLOCKS = 15

E = 10e9
NU = 0.0

RHO_s = 2000.
RHO_b = 2000.

k = 14 * E * H * B / 3
CONT = ct.NoTension(k, k * 100)
SURF = surf.NoTension(k, k)
# %% Building structure

St = st.Structure_2D()

St.add_fe(N1, N2, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_beam(N4, N5, BLOCKS, H, rho=RHO_b, b=B)
St.add_fe(N3, N4, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N2, N3, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N5, N6, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N2, N5, E, NU, H, b=B, lin_geom=False, rho=RHO_s)

St.make_nodes()
St.make_cfs(False, offset=0, nb_cps=2, contact=CONT)

# St.plot_structure(scale=0, plot_cf=False)

# %% BCs and Forces

St.fixNode(N1, [0, 1, 2])
St.fixNode(N6, [0, 1, 2])

St.get_M_str()

a = 3

for i in range(len(St.list_blocks)):
    W = St.list_blocks[i].m
    N = St.list_blocks[i].ref_point
    Node = St.get_node_id(N)
    St.loadNode(Node, 1, -W, fixed=True)
    St.loadNode(Node, 0, a * W)

for i in range(len(St.list_fes)):
    W = St.list_fes[i].mass[0, 0]
    N_1 = St.list_fes[i].N1
    N_2 = St.list_fes[i].N2
    Node1 = St.get_node_id(N_1)
    Node2 = St.get_node_id(N_2)
    St.loadNode(Node1, 1, -W, fixed=True)
    St.loadNode(Node2, 1, -W, fixed=True)
    St.loadNode(Node1, 0, a * W)
    St.loadNode(Node2, 0, a * W)

# %% Pushover analysis
Node = 7

LIST = np.linspace(0, 5e-3, 1000)
LIST = np.append(LIST, np.linspace(5e-3 + 1e-6, 2e-1, 1500))

LIST = LIST.tolist()

# %% Simulation
St.solve_dispcontrol(LIST, 0, Node, 0, tol=.1, max_iter=1000)
# St.solve_forcecontrol(3000,tol=1e-1,max_iter=1000)

# %% Plot Structure at end of simulation

St.plot_structure(scale=1, plot_cf=False, plot_forces=False)
# %%
St.save_structure(filename='Composite_Frame_Pushover')
