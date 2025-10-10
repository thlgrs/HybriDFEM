import numpy as np
import os
import h5py
import sys
import pathlib
import matplotlib.pyplot as plt

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as cont
import ContactPair as cp
import Surface as surf

save_path = os.path.dirname(os.path.abspath(__file__))

# Meth =['HHT', 0.3]
# Meth = 'CAA'
# Meth = 'CDM'

kn = 20e5
ks = 20e5

B = 1.
H = 1.

nb_blocks = 2

RHO = 1000

N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, 1], dtype=float)

St1 = st.Structure_2D()
St1.add_beam(N1, N2, 2, H, RHO)
St1.make_nodes()

nb_cps = np.linspace(-1, 1, 1000)
nb_cps = nb_cps.tolist()

nb_cps = 100
St1.make_cfs(False, nb_cps=nb_cps, offset=-1, surface=surf.Surface(kn, ks))
# St1.make_cfs(False, nb_cps=nb_cps, offset=-1, surface=surf.NoTension_CD(kn,ks))
St1.plot_structure(scale=0, plot_forces=False, plot_cf=True)
St1.fixNode(0, [0, 1, 2])

St1.solve_modal(3)
St1.plot_modes(3, scale=1)
