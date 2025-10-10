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
import pandas as pd


def reload_modules():
    importlib.reload(st)
    importlib.reload(mat)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat

reload_modules()

# %% Structure parameters

N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, 1], dtype=float)
N3 = np.array([0, 2], dtype=float)
N4 = np.array([0, 3], dtype=float)
N5 = np.array([3, 3], dtype=float)
N6 = np.array([3, 2], dtype=float)
N7 = np.array([3, 1], dtype=float)
N8 = np.array([3, 0], dtype=float)

B_b = .2
H_b = .2
H_c = .2 * 2 ** (1 / 3)

CPS = 25
BLOCKS = 30

E = 30e9
NU = 0.0
FY = 20e6

RHO = 2000.
LIN = False

# MAT = mat.Bilinear_Mat(E, NU, FY)
MAT = mat.Plastic_Stiffness_Deg(E, NU, FY)
# MAT = mat.Plastic_Mat(E,NU,FY)
# MAT = mat.Material(E,NU)
# MAT.plot_stress_strain()
# %% Building structure

St = st.Structure_2D()

St.add_fe(N1, N2, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_fe(N2, N3, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_fe(N3, N4, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_beam(N4, N5, BLOCKS, H_b, RHO, b=B_b, material=MAT)
St.add_fe(N5, N6, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_fe(N6, N7, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)
St.add_fe(N7, N8, E, NU, H_c, b=B_b, lin_geom=LIN, rho=RHO)

St.make_nodes()
St.make_cfs(LIN, nb_cps=CPS)

# %% BCs and Forces

St.fixNode(N1, [0, 1])
St.fixNode(N8, [0, 1])
# %% Modal analysis

St.save_structure('Undamaged_Frame')
St.solve_modal(4)
St.plot_modes(4, scale=10)

with open(f'Undamaged_Frame.pkl', 'rb') as file:
    St = pickle.load(file)

# Apply self-weight statically
St.get_M_str()
W_tot = 0
# Start simulation

for i in range(len(St.list_blocks)):
    m = St.list_blocks[i].m
    W = 9.81 * m
    W_tot += W
    Node = St.get_node_id(St.list_blocks[i].ref_point)
    St.loadNode(Node, 1, -W)

for i in range(len(St.list_fes)):
    m = St.list_fes[i].mass[0, 0]
    W = 9.81 * m
    W_tot += W
    Node1 = St.get_node_id(St.list_fes[i].N1)
    Node2 = St.get_node_id(St.list_fes[i].N2)
    St.loadNode(Node1, 1, -W)
    St.loadNode(Node2, 1, -W)

St.solve_forcecontrol(10, max_iter=100)
St.plot_structure(scale=1000, plot_cf=True, plot_forces=True, plot_supp=True, save=None)

# Loading for dynamic sim. 
St.reset_loading()

for i in range(len(St.list_blocks)):
    m = St.list_blocks[i].m
    W = 9.81 * m
    Node = St.get_node_id(St.list_blocks[i].ref_point)
    St.loadNode(Node, 1, -W, fixed=True)
    St.loadNode(Node, 0, -W)

for i in range(len(St.list_fes)):
    m = St.list_fes[i].mass[0, 0]
    W = 9.81 * m
    Node1 = St.get_node_id(St.list_fes[i].N1)
    Node2 = St.get_node_id(St.list_fes[i].N2)
    St.loadNode(Node1, 0, -W)
    St.loadNode(Node2, 0, -W)
    St.loadNode(Node1, 1, -W, fixed=True)
    St.loadNode(Node2, 1, -W, fixed=True)


# Parameters of simulation
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
pga = 7.75
lmbda = pga * lmbda / np.max(abs(lmbda))

dt_new = 1e-2
new_time = np.arange(time[0], time[-1], dt_new)

from scipy.interpolate import interp1d

interpolator = interp1d(time, lmbda, kind="linear", fill_value="extrapolate")
new_lmbda = interpolator(new_time)

St.set_damping_properties(xsi=0.02, damp_type='RAYLEIGH')

# #%% Computation 
Meth = 'NWK'
St.solve_dyn_nonlinear(20, dt_new, Meth=Meth, lmbda=new_lmbda.tolist(), filename=f'Dynamic/FPGA={pga}g')

St.solve_modal(10, filename=f'Modal/PGA={pga}g', save=True)
St.plot_modes(4, scale=10)
