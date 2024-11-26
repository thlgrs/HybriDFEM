import numpy as np
import os
import h5py
import sys
import pathlib


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat
import Surface as surf

save_path = os.path.dirname(os.path.abspath(__file__))

# Meth = 'CDM'
# Meth = 'CAA'
# Meth = 'LA'
# Meth = ['WIL', 1.4]
Meth = ['HHT', .0]
# Meth = ['HHT', .3]
# Meth = ['WBZ', .1, .6, .3025]
# Meth = ['WBZ', -.1, .6, .3025]

N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, 1], dtype=float)

BLOCKS = 3
CPS = 100

kn = 200e3 * 2
ks = kn
w_s = 10
p0 = 1.66e4

St = st.Structure_2D()

pattern  = np.ones((1,3))

St.add_wall(N1, 1/3, 1., pattern, 6000, b=1)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS, surface=surf.Surface(kn,ks))

def excitation(x): return np.sin(w_s*x) 

St.loadNode(1, [0], p0)
St.fixNode(0, [0,1,2])
# St.fixNode([1,2], [1,2])

# St.solve_modal()
# St.plot_modes()

# St.solve_linear()
St.plot_structure(scale=1, plot_cf=True)

# U0 = St.U.copy()

St.set_damping_properties(xsi=0.05, damp_type='RAYLEIGH')

St.solve_dyn_linear(10, 1e-2, lmbda=excitation, Meth=Meth)

St.save_structure(filename='2DoF_Vibration')

