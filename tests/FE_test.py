from tests.Mesh_test import gen_sizes
from Objects.FE_2D import *


def cantilever_steel_beam(file):
    steel = Material(E=210e9, nu=0.30, plane='stress', rho=0)
    beam = FE_2D(file, material=steel)
    min_x, max_x = beam.domain.get_mesh_bounding_box()[:, 0]
    mid_y = beam.domain.get_mesh_bounding_box()[:, 1].mean()

    beam.add_dirichlet('clamp', 'left', {'u.all': 0.0})
    beam.add_surface_load('load', 'top', 0.0, -1e6)

    beam.solve()
    xs = np.linspace(min_x, max_x, 50)
    pts = np.column_stack([xs, np.full_like(xs, mid_y)])
    uy = beam.probe(pts, 'uy')
    print('flèche :', uy[-1])
    return uy[-1]

    #strain_voigt = beam.probe(pts, 'strain')  # [ε_xx, ε_yy, γ_xy]
    #stress = beam.probe(pts, 'syy')  # [σ_xx, σ_yy, σ_xy]
    #print(f"mid‑point coordinates   : {pts}")
    #print(f"displacement            :\n{uy} m")
    #print(f"strain (Voigt)          :\n{strain_voigt}")
    #print(f"stress (Pa, Voigt)      :\n{stress}")


from pathlib import Path


def clear_suffix(suffixes, root='.'):
    """
    Delete all files ending with any suffix in *suffixes*.

    Parameters
    ----------
    suffixes : str | list[str]
        '.vtk', '.msh', or a list such as ['.vtk', '.msh'].
    root     : str | Path, optional
        Top‑level directory where the search starts (default: current dir).
    """
    if isinstance(suffixes, str):
        suffixes = [suffixes]

    root = Path(root)
    for sfx in suffixes:
        for file in root.rglob(f'*{sfx}'):  # recursive search
            try:
                file.unlink(missing_ok=True)  # Py ≥ 3.8
                print('removed', file)
            except Exception as err:
                print('could not remove', file, '->', err)

def convergence(sizes=None, clear=False):
    w_tri = []
    w_quad = []
    if sizes is None:
        sizes = [1, 1 / 2, 1 / 5, 1 / 20, 1 / 50, 1 / 100, 1 / 200]
    for size in sizes:
        w_tri.append(cantilever_steel_beam(f'output/cantilever_quad{size}.msh'))
        w_quad.append(cantilever_steel_beam(f'output/cantilever_tri{size}.msh'))

    print('triangles :', w_tri)
    print('quadrilaterals :', w_quad)
    if clear: clear_suffix(['.vtk', '.msh'],'output')

if __name__ == '__main__':
    clear_suffix(['.vtk', '.msh'], 'output')
    gen_sizes()
    convergence()


