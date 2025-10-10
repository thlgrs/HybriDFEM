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
# mu=10
SURF = surf.NoTension_EP(k, k)
# %% Building structure

St = st.Structure_2D()

St.add_fe(N1, N2, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_beam(N4, N5, BLOCKS, H, rho=RHO_b, b=B)
St.add_fe(N3, N4, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N2, N3, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N5, N6, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N2, N5, E, NU, H, b=B, lin_geom=False, rho=RHO_s)

x = .5 * np.array([1, 0])
y = .5 * np.array([0, 1])
h_b = .05

vertices = np.array([N1, N1, N1, N1])
vertices[0] += H * x - h_b * y
vertices[1] += H * x + h_b * y
vertices[2] += - H * x + h_b * y
vertices[3] += - H * x - h_b * y
St.add_block(vertices, b=B, rho=RHO_b)

vertices = np.array([N6, N6, N6, N6])
vertices[0] += H * x - h_b * y
vertices[1] += H * x + h_b * y
vertices[2] += - H * x + h_b * y
vertices[3] += - H * x - h_b * y
St.add_block(vertices, b=B, rho=RHO_b)

h_b = .2
N7 = np.array([1.5, -0.05 / 2 - h_b / 2], dtype=float)
b_b = 4
vertices = np.array([N7, N7, N7, N7])
vertices[0] += b_b * x - h_b * y
vertices[1] += b_b * x + h_b * y
vertices[2] += - b_b * x + h_b * y
vertices[3] += - b_b * x - h_b * y
St.add_block(vertices, b=B, rho=RHO_b)

St.make_nodes()
St.make_cfs(False, offset=-1, nb_cps=[-1, -.5, 0, .5, 1.], surface=SURF)

St.plot_structure(scale=0, plot_cf=True)

# %% BCs and Forces

St.fixNode(N7, [0, 1, 2])
# St.fixNode(N6, [0,1,2])

St.get_M_str()

for i in range(len(St.list_blocks)):
    W = St.list_blocks[i].m
    N = St.list_blocks[i].ref_point
    Node = St.get_node_id(N)
    St.loadNode(Node, 1, -W, fixed=True)
    St.loadNode(Node, 0, W)

for i in range(len(St.list_fes)):
    W = St.list_fes[i].mass[0, 0]
    N_1 = St.list_fes[i].N1
    N_2 = St.list_fes[i].N2
    Node1 = St.get_node_id(N_1)
    Node2 = St.get_node_id(N_2)
    St.loadNode(Node1, 1, -W, fixed=True)
    St.loadNode(Node2, 1, -W, fixed=True)
    St.loadNode(Node1, 0, W)
    St.loadNode(Node2, 0, W)

# St.solve_modal()
# print(St.eig_vals)

# %% Dynamic analysis
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


dt, lmbda = read_accelerogram('Earthquakes/NF13')
time = np.arange(len(lmbda)) * dt
pga = 1
lmbda = pga * lmbda / np.max(abs(lmbda))

dt_new = 5e-3
new_time = np.arange(time[0], time[-1], dt_new)

from scipy.interpolate import interp1d

interpolator = interp1d(time, lmbda, kind="linear", fill_value="extrapolate")
new_lmbda = interpolator(new_time)

Meth = 'CAA'
St.set_damping_properties(xsi=0.02, damp_type='STIFF', stiff_type='TAN')

St.solve_dyn_nonlinear(20, dt_new, Meth=Meth, lmbda=new_lmbda.tolist(), filename=f'Response_NF13_SSI_{pga}')
# %% Plot Structure at end of simulation

St.plot_structure(scale=1, plot_cf=False, plot_forces=False)
# %%
St.save_structure(filename='Composite_Frame_SSI')
# %%
