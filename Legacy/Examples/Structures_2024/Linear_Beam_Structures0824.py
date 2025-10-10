import numpy as np

"""
folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))
"""

from Legacy.Objects import Structure as st
from Legacy.Objects import Material as mat

N1 = np.array([0, 0], dtype=float)

BLOCKS = 15
PATTERN = np.ones((BLOCKS, 6 * BLOCKS))
CPS = 6

H_B = .5 / BLOCKS
L_B = 3 / (6 * BLOCKS)
B = .2

E = 60e9
NU = 0.0

St = st.Structure_2D()

St.add_wall(N1, L_B, H_B, PATTERN, 0., b=B, material=mat.Material(E, NU))
St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

F = -100e3

to_load = []
to_fix = []

N = BLOCKS

for i in range(N):
    to_fix.append(i * 6 * BLOCKS)
    to_load.append((i + 1) * 6 * BLOCKS - 1)

St.loadNode(to_load, 1, F)
St.fixNode(to_fix, [0, 1, 2])

St.solve_linear()

print(St.U[-2] * 1000)

St.get_P_r()

St.plot_stresses(angle=np.pi / 2, save=False)

St.plot_structure(plot_cf=False, scale=40)
