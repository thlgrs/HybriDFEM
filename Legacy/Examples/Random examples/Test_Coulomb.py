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

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat
import Contact as cont
import Surface as surf

save_path = os.path.dirname(os.path.abspath(__file__))

N0 = np.array([0, 0], dtype=float)
N1 = np.array([0, 1], dtype=float)

St = st.Structure_2D()

St.add_beam(N0, N1, 2, 1., 0., b=1.)

kn = 10.
ks = 5.

c = 1.2
mu = .7
psi = .35

surface = surf.Coulomb(kn, ks, mu, c=c, psi=psi)

dL = np.array([-1 / kn, 0])

surface.update(dL)
surface.commit()

print('stress: ', surface.stress)
print('disps: ', surface.disps)

dL = .5 * np.array([.1, 1.3])

surface.update(dL)
surface.commit()

print('stress: ', surface.stress)
print('disps: ', surface.disps)
print('stiff: ', surface.stiff)
# St.make_nodes()
# St.make_cfs(True, nb_cps=1, offset = -1, surface=surf.Coulomb(kn, ks, mu, c=c, psi=psi))

# St.plot_structure(plot_cf=True, scale=0)
