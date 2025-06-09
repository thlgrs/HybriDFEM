from FE_2D_dev.Mesh_test import gen_sizes
from Objects.FE_2D import FE_2D, Material
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

STEEL = Material(E=210e9, nu=0.30, plane="stress", rho=0)
SIZES = [
    4,
    2,
    1,
    3 / 4,
    1 / 2,
    1 / 3,
    1 / 4,
    1 / 6,
    1 / 8,
    1 / 12,
    1 / 16,
    1 / 24,
    1 / 32,
]  # 1/48, 1/64, 1/128]#, 1/256]#, 1/512]


def clear_suffix(suffixes, root="."):
    if isinstance(suffixes, str):
        suffixes = [suffixes]

    root = Path(root)
    for sfx in suffixes:
        for file in root.rglob(f"*{sfx}"):  # recursive search
            try:
                file.unlink(missing_ok=True)  # Py ≥ 3.8
                print("removed", file)
            except Exception as err:
                print("could not remove", file, "->", err)


def convergence(func, sizes=None, clear=False):
    w_tri = []
    w_quad = []
    n_elem = [[], []]
    if sizes is None:
        sizes = [
            1,
            3 / 4,
            1 / 2,
            1 / 3,
            1 / 4,
            1 / 6,
            1 / 8,
            1 / 12,
            1 / 16,
            1 / 24,
            1 / 32,
            1 / 48,
            1 / 64,
            1 / 128,
        ]
    for size in sizes:
        wq, nq = func(f"output/beam_quad{size}.msh")
        wt, nt = func(f"output/beam_tri{size}.msh")
        n_elem[0].append(nt)
        w_tri.append(wt)
        n_elem[1].append(nq)
        w_quad.append(wq)

    print("triangles :", w_tri)
    print("quadrilaterals :", w_quad)
    if clear:
        clear_suffix([".vtk", ".msh"], "output")
    return w_tri, w_quad, n_elem


def error(approach, exact):
    err = []
    for a, e in zip(approach, exact):
        err.append(100 * (e - a) / e)
    return err


def scientific_plot_latex(
    x,
    ys,
    labels=None,
    xlabel="",
    ylabel="",
    title="",
    legend_loc="best",
    save_path=None,
    use_latex=True,
):
    """
    Create a clean, publication-quality scientific plot with optional LaTeX typesetting.

    Parameters:
        x (array-like or list of array-like): x-axis data, either a single array or list of arrays (one per y).
        ys (list of array-like): A list of y-axis datasets to plot.
        labels (list of str, optional): Labels for each y dataset.
        xlabel (str): Label for x-axis (supports LaTeX).
        ylabel (str): Label for y-axis (supports LaTeX).
        title (str): Plot title (supports LaTeX).
        legend_loc (str): Location of the legend.
        save_path (str, optional): If provided, saves the figure to this path.
        use_latex (bool): Enables LaTeX typesetting if True.
    """
    if use_latex:
        mpl.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times New Roman"],
                "axes.labelsize": 12,
                "font.size": 12,
                "legend.fontsize": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )

    plt.figure(figsize=(6, 4))

    # Check if x is a single array (broadcast to all ys), or a list of arrays
    if not isinstance(x, list):
        x_list = [x] * len(ys)
    else:
        if len(x) != len(ys):
            raise ValueError("Length of x and ys must match when x is a list.")
        x_list = x

    for idx, (x_vals, y_vals) in enumerate(zip(x_list, ys)):
        label = labels[idx] if labels else None
        plt.plot(x_vals, y_vals, linewidth=0.7, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    plt.grid(True, linestyle="--", alpha=0.6)
    if labels:
        plt.legend(loc=legend_loc)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def cantilever_steel_beam_uniform_load(file):
    beam = FE_2D(file, material=STEEL)
    min_x, max_x = beam.domain.get_mesh_bounding_box()[:, 0]
    mid_y = beam.domain.get_mesh_bounding_box()[:, 1].mean()

    beam.add_dirichlet("clamp", "left", {"u.all": 0.0})
    beam.add_surface_load("top", py=-1e6)

    beam.solve()
    xs = np.linspace(min_x, max_x, 50)
    pts = np.column_stack([xs, np.full_like(xs, mid_y)])
    uy = beam.probe(pts, "uy")
    print("flèche :", uy[-1])
    n_el = beam.domain.shape.n_el
    return uy[-1][0], n_el


def simple_steel_beam_uniform_load(file):
    beam = FE_2D(file, material=STEEL)
    min_x, max_x = beam.domain.get_mesh_bounding_box()[:, 0]
    mid_y = beam.domain.get_mesh_bounding_box()[:, 1].mean()

    beam.add_dirichlet("left_sup", "left", {"u.all": 0.0})
    beam.add_dirichlet("right_sup", "right", {"u.all": 0.0})
    beam.add_surface_load("top", py=-1e6)

    beam.solve()
    xs = np.linspace(min_x, max_x, 50)
    pts = np.column_stack([xs, np.full_like(xs, mid_y)])
    uy = beam.probe(pts, "uy")
    mid_idx = len(uy) // 2
    print("flèche :", uy[mid_idx])
    n_el = beam.domain.shape.n_el
    return uy[mid_idx][0], n_el


def cantilever_steel_point_load(file):
    beam = FE_2D(file, material=STEEL)
    min_x, max_x = beam.domain.get_mesh_bounding_box()[:, 0]
    mid_y = beam.domain.get_mesh_bounding_box()[:, 1].mean()

    beam.add_dirichlet("clamp", "left", {"u.all": 0.0})
    beam.add_point_load(3, 0.3, fy=-1e6)

    beam.solve()
    xs = np.linspace(min_x, max_x, 50)
    pts = np.column_stack([xs, np.full_like(xs, mid_y)])
    uy = beam.probe(pts, "uy")
    print("flèche :", uy[-1])
    n_el = beam.domain.shape.n_el
    return uy[-1][0], n_el


def plots_point_load_cantilever():
    clear_suffix([".vtk", ".msh"], "output")
    gen_sizes(SIZES)
    w_tri, w_quad, n_elem = convergence(func=cantilever_steel_point_load, sizes=SIZES)
    analytic = -np.ones_like(w_tri) * 0.019047619  # Analytical deflection
    errors = [error(w_tri, analytic), error(w_quad, analytic)]
    scientific_plot_latex(
        [n_elem[0], n_elem[1]],
        errors,
        labels=["tri_elem", "quad_elem"],
        xlabel="Number of elements",
        ylabel=r"Error $[\%]$",
        save_path="output/point_load_errors.png",
        title="Deflection error : Cantilever point load",
        use_latex=True,
    )
    scientific_plot_latex(
        [n_elem[0], n_elem[1], n_elem[0]],
        [w_tri, w_quad, analytic],
        labels=["tri_elem", "quad_elem", "E-B solution"],
        xlabel="Number of elements",
        ylabel=r"Cantilever free hand deflection $[m]$",
        save_path="output/point_load_deflection.png",
        title="Free end deflection : Cantilever point load",
        use_latex=True,
    )
    # clear_suffix(['.vtk', '.msh'], 'output')


def plots_uniform_load_cantilever():
    # clear_suffix(['.vtk', '.msh', '.png'], 'output')
    # gen_sizes(SIZES)
    w_tri, w_quad, n_elem = convergence(
        func=cantilever_steel_beam_uniform_load, sizes=SIZES
    )
    analytic = -np.ones_like(w_tri) * (1 / 14)  # Analytical deflection
    errors = [error(w_tri, analytic), error(w_quad, analytic)]
    scientific_plot_latex(
        [n_elem[0], n_elem[1]],
        errors,
        labels=["tri_elem", "quad_elem"],
        xlabel="Number of elements",
        ylabel=r"Error $[\%]$",
        save_path="output/uni_load_errors.png",
        title="Deflection error : Cantilever uniformly distributed load",
        use_latex=True,
    )
    scientific_plot_latex(
        [n_elem[0], n_elem[1], n_elem[0]],
        [w_tri, w_quad, analytic],
        labels=["tri_elem", "quad_elem", "E-B solution"],
        xlabel="Number of elements",
        ylabel=r"Cantilever free hand deflection $[m]$",
        save_path="output/uni_load_deflection.png",
        title="Free end deflection : Cantilever uniformly distributed load",
        use_latex=True,
    )
    # clear_suffix(['.vtk', '.msh'], 'output')


def plots_uniform_load_simple_beam():
    # clear_suffix(['.vtk', '.msh'], 'output')
    # gen_sizes(SIZES)
    w_tri, w_quad, n_elem = convergence(
        func=simple_steel_beam_uniform_load, sizes=SIZES
    )
    analytic = -np.ones_like(w_tri) * (1 / 672)  # Analytical deflection
    errors = [error(w_tri, analytic), error(w_quad, analytic)]
    scientific_plot_latex(
        [n_elem[0], n_elem[1]],
        errors,
        labels=["tri_elem", "quad_elem"],
        xlabel="Number of elements",
        ylabel=r"Error $[\%]$",
        save_path="output/simple_beam_errors.png",
        title="Deflection error : Simple beam",
        use_latex=True,
    )
    scientific_plot_latex(
        [n_elem[0], n_elem[1], n_elem[0]],
        [w_tri, w_quad, analytic],
        labels=["tri_elem", "quad_elem", "E-B solution"],
        xlabel="Number of elements",
        ylabel=r"Simple beam deflection $[m]$",
        save_path="output/simple_beam_deflection.png",
        title="Mid-span deflection : Simple beam load",
        use_latex=True,
    )


if __name__ == "__main__":
    # clear_suffix(['.vtk', '.msh','.png'], 'output')
    # gen_sizes(SIZES)
    plots_uniform_load_cantilever()
    plots_uniform_load_simple_beam()
