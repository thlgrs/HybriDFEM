import os
import pathlib
import sys

import numpy as np

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat

save_path = os.path.dirname(os.path.abspath(__file__))

# Meth = 'CDM'
# Meth = 'CAA'
Meth = 'LA'
Meth = ['WIL', 1.4]
Meth = ['HHT', .1]
Meth = ['HHT', .3]
Meth = ['WBZ', .1, .6, .3025]
Meth = ['WBZ', -.1, .6, .3025]

N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, 7], dtype=float)
N3 = np.array([7, 7], dtype=float)

H = .124
B = .124

BLOCKS = 10
CPS = 100

E = 210e9
NU = 0.0

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, 5574., b=B, material=mat.Material(E, NU, shear_def=True))
St.add_beam(N2, N3, BLOCKS, H, 5574., b=B, material=mat.Material(E, NU, shear_def=True))

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

F = 0
F_ref = 500

w_s = 10


def excitation(x): return -np.sin(w_s * x)


# St.loadNode(N3, [1], F, fixed=True)
St.loadNode(N3, [1], F_ref)
St.fixNode(N1, [0, 1, 2])
# St.fixNode(N2, [1])


# St.solve_linear()
# St.plot_structure(scale=10, plot_cf=False)

# U0 = St.U.copy()

St.set_damping_properties(xsi=0.05, damp_type='RAYLEIGH')

St.solve_dyn_linear(20, 1e-2, lmbda=excitation, Meth=Meth)

St.save_structure(filename='Crane')
