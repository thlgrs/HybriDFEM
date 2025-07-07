from dataclasses import dataclass
from typing import List, Union, Callable

import meshio
import numpy as np
from scipy.spatial import KDTree
from sfepy.discrete import (
    FieldVariable,
    Integral,
    Equation,
    Equations,
    Problem,
    Conditions,
)
from sfepy.discrete import Material as sfepyMaterial
from sfepy.discrete.conditions import EssentialBC
from sfepy.discrete.fem import FEDomain, Field
from sfepy.discrete.fem import Mesh as sfepyMesh
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.terms import Term
from sfepy.terms.terms import Terms

@dataclass
class Material:
    E: float        # Young's modulus [Pa]
    nu: float       # Poisson's ratio
    rho: float      # Density [kg/m^3]
    plane: str      # "stress" or "strain"

class FE_2D:
    def __init__(self, mesh_file: str, material: Material, order: int = 1):
        # --- Mesh & domain setup ------------------------------------------
        self.mesh_name = mesh_file.removesuffix(".msh")
        mesh = sfepyMesh.from_file(mesh_file)
        if mesh.coors.shape[1] == 3:
            mesh.coors[:, 2] = 0.0
        mesh.write(f"{self.mesh_name}.vtk", io="auto")
        mesh = sfepyMesh.from_file(f"{self.mesh_name}.vtk")

        self.domain = FEDomain("domain", mesh)

        # --- Bounding box & automatic gtol --------------------------------
        bb = self.domain.get_mesh_bounding_box()  # [[xmin,ymin],[xmax,ymax]]
        (xmin, ymin), (xmax, ymax) = bb

        # build KD‐tree of node coords and get typical spacing h_typ
        coords = mesh.coors[:, :2]
        tree = KDTree(coords)
        dists, _ = tree.query(coords, k=2)
        h_typ = float(np.median(dists[:, 1]))

        # gtol = fraction of h_typ, with safe fallback
        self.gtol = max(1e-8, 1e-3 * h_typ)
        tol = self.gtol

        # --- Selector definitions -----------------------------------------
        self.selector_list = {
            "all": lambda: "all",
            "left": lambda: f"vertices in (x < {xmin + tol:.10e})",
            "right": lambda: f"vertices in (x > {xmax - tol:.10e})",
            "bottom": lambda: f"vertices in (y < {ymin + tol:.10e})",
            "top": lambda: f"vertices in (y > {ymax - tol:.10e})",
            "boundary": lambda: "vertices of surface",
            "line_at_x": lambda x: f"vertices in (x > {x - tol:.10e}) * (x < {x + tol:.10e})",
            "line_at_y": lambda y: f"vertices in (y > {y - tol:.10e}) * (y < {y + tol:.10e})",
            "box": lambda x0, x1, y0, y1: (
                f"vertices in (x > {x0 - tol:.10e}) * (x < {x1 + tol:.10e})"
                f" * (y > {y0 - tol:.10e}) * (y < {y1 + tol:.10e})"
            ),
        }

        # --- Regions -------------------------------------------------------
        self.regions = {}
        # Omega = whole domain cells
        self.regions["Omega"] = self.domain.create_region("Omega", "all")

        # --- Field & variables --------------------------------------------
        self.field = Field.from_args(
            "u_field", np.float64, "vector", self.regions["Omega"], approx_order=order
        )
        self.u = FieldVariable("u", "unknown", self.field)
        self.v = FieldVariable("v", "test", self.field, primary_var_name="u")

        # --- Material & internal term -------------------------------------
        self.D = stiffness_from_youngpoisson(
            2, material.E, material.nu, plane=material.plane
        )
        self.mat = sfepyMaterial("mat", D=self.D)

        self.integral = Integral("i", order=2)
        self.t_int = Term.new(
            "dw_lin_elastic(mat.D, v, u)",
            self.integral,
            self.regions["Omega"],
            mat=self.mat,
            v=self.v,
            u=self.u,
        )

        # --- BC & load containers -----------------------------------------
        self.ebcs = []
        self.load_terms = []
        self.loads = {}

    def selector(self, selector: Union[str, Callable[..., str]], **kwargs) -> str:
        """
        Turn a selector key/callable/raw‐string into a SfePy region expression.

        Args:
            selector:  one of
                         - a key in self.selector_list (e.g. 'left', 'surface', …)
                         - a callable returning a region‐string (must accept **kwargs)
                         - a raw region expression string
            **kwargs:  parameters passed to the above if needed

        Returns:
            A valid SfePy region‐expression string (e.g. 'vertices in (x < 1e-8) *v …').

        Raises:
            KeyError:   if selector is a str but not in self.selector_list and not looking like an expression.
            TypeError:  if selector is neither str nor callable.
            ValueError: if the resulting expression is empty.
        """
        # 1) name → callable
        if callable(selector):
            expr = selector(**kwargs)

        # 2) string → lookup in self.selector_list
        elif isinstance(selector, str) and selector in self.selector_list:
            expr = self.selector_list[selector](**kwargs)

        # 3) raw expression string (heuristic: must contain at least one space or comparison)
        elif isinstance(selector, str) and any(
            tok in selector for tok in (" ", "<", ">", "*", "vertices", "cells")
        ):
            expr = selector

        else:
            raise KeyError(
                f"Selector '{selector}' not found in predefined selectors "
                f"and not a valid raw expression."
            )

        expr = expr.strip()
        if not expr:
            raise ValueError("Built selector expression is empty.")
        return expr

    def combine_selectors(
        self, *selectors: Union[str, Callable[..., str]], op: str = "and", **kwargs
    ) -> str:
        """
        Build a combined SfePy selector expression by AND/OR-ing
        any number of existing selectors or raw expressions.

        Args:
            *selectors:  names in self.selector_list, callables, or raw expr strings
            op:          'and' (default) or 'or'
            **kwargs:    keyword args passed to each selector callable

        Returns:
            A single region‐expression string, e.g.
            'vertices in (x < 0.1) * (y > 0.5)' or
            'vertices in (...) *v vertices in (...)'.

        Raises:
            KeyError/TypeError/ValueError from build_selector if something’s wrong.
        """
        # pick the SfePy operator
        if op == "and":
            joiner = " * "
        elif op == "or":
            joiner = " *v "
        else:
            raise ValueError(f"op must be 'and' or 'or', got {op!r}")

        parts = [self.selector(sel, **kwargs) for sel in selectors]
        # wrap each part in parentheses if it’s more than a single token
        parts = [p if p.isidentifier() else f"({p})" for p in parts]
        expr = joiner.join(parts)
        return expr

    def create_region(
        self,
        name: str,
        selector: Union[str, Callable[..., str]],
        kind: str = "cell",
        override: bool = False,
        **kwargs,
    ):
        """
        Create and store a region in the model.

        Args:
            name:        the (unique) name of the new region.
            selector:    either
                           - a key of self.selector_list (e.g. 'left', 'surface', etc.),
                           - a raw SfePy region expression string, or
                           - a callable returning such a string.
            kind:        one of 'cell', 'facet' or 'vertex'.
            override:    if False (default), raises if `name` already exists.
            **kwargs:    any parameters passed to the selector callable.

        Returns:
            The new region object.

        Raises:
            ValueError: if the region is empty or name already exists.
            KeyError:   if selector is a string but not in self.selector_list.
        """
        # 1) check name
        if (not override) and (name in self.regions):
            raise ValueError(
                f"Region '{name}' already exists; use override=True to replace."
            )

        # 2) build the SfePy‐style expression string
        if callable(selector):
            expr = selector(**kwargs)
        elif isinstance(selector, str) and (selector in self.selector_list):
            expr = self.selector_list[selector](**kwargs)
        elif isinstance(selector, str):
            expr = selector
        else:
            raise TypeError(f"Selector must be a str or callable, got {type(selector)}")

        # 3) create region on the domain
        region = self.domain.create_region(name, expr, kind=kind)

        # 4) sanity check: must have at least one entity
        count = getattr(
            region,
            {"cell": "n_cells", "facet": "n_facets", "vertex": "n_vertices"}[kind],
        )
        if count == 0:
            # clean up
            del region
            raise ValueError(
                f"Region '{name}' with expression '{expr}' is empty (kind={kind})."
            )

        # 5) store and return
        self.regions[name] = region
        return region

    def add_dirichlet(self, name: str, selector: str, comp_dict: dict, **kwargs):
        """
        Fix a displacement component (or all) on a given boundary.
        """
        reg = self.domain.create_region(
            name, self.selector_list[selector](**kwargs), kind="facet"
        )
        self.regions[name] = reg
        self.ebcs.append(EssentialBC(name, reg, comp_dict))

    def add_functional_dirichlet(
        self, name: str, selector: str, funcs: List[callable] = None, **kwargs
    ) -> None:
        expr = self.selector_list[selector](**kwargs)
        reg = self.domain.create_region(name, expr, kind="facet")
        self.regions[name] = reg
        if funcs is None:
            self.ebcs.append(EssentialBC(name, reg, {"u.all": 0.0}))
        elif len(funcs) == 2:
            self.ebcs.append(EssentialBC(name + "x", reg, {"u.0": funcs[0]}))
            self.ebcs.append(EssentialBC(name + "y", reg, {"u.1": funcs[1]}))
        else:
            self.ebcs.append(EssentialBC(name, reg, {"u.all": funcs[0]}))

    def add_surface_load(self, selector: str, px: float = 0, py: float = 0, **kwargs):
        if px == 0.0 and py == 0.0:
            return
        idx = len(self.load_terms)
        load_name = f"load_{idx}"

        load_val = np.array([[px], [py]], dtype=float)  # shape (2,1)
        surf_load = sfepyMaterial(load_name, val=load_val)

        expr = self.selector_list[selector](**kwargs)
        reg = self.domain.create_region(f"{load_name}_reg", expr, kind="facet")

        term = Term.new(
            f"dw_surface_ltr({load_name}.val, v)",
            self.integral,
            region=reg,
            **{load_name: surf_load},
            v=self.v,
        )

        self.regions[f"{load_name}_reg"] = reg
        self.load_terms.append(term)
        self.loads[load_name] = surf_load

    def solve(self, vtk_out: str = None):
        """assemble, solve and store the converged state"""
        if vtk_out is None:
            vtk_out = "{}_solved_lin.vtk".format(self.mesh_name)
        elif not vtk_out.endswith(".vtk"):
            vtk_out += ".vtk"

        term_list = [self.t_int] + [-lt for lt in self.load_terms]

        rhs = term_list[0] if len(term_list) == 1 else Terms(term_list)
        eq = Equation("balance", rhs)

        pb = Problem("elasticity_2D", equations=Equations([eq]), domain=self.domain)
        pb.set_bcs(ebcs=Conditions(self.ebcs))

        ls = ScipyDirect({})
        nls = Newton({"i_max": 1}, lin_solver=ls)
        pb.set_solver(nls)

        self.state = pb.solve()  # keep for later use
        self.u = pb.get_variables()["u"]  # update variable handle
        pb.save_state(vtk_out, self.state)

        return self.state

    def probe(self, coors, quantity="u", return_status=False):
        """
        Interpolate solution‑related data at coordinates *coors*.
        quantity ∈ {'u','ux','uy','strain','stress','sxx','syy','sxy'}.
        """
        if not hasattr(self, "state"):
            raise RuntimeError("solve() first!")

        pts = np.atleast_2d(coors)
        u_val, _, inside = self.u.evaluate_at(pts, ret_cells=True, ret_status=True)

        # strains from displacement gradient
        if (
            quantity.startswith("strain")
            or quantity.startswith("stress")
            or quantity[-2:] in ("xx", "yy", "xy")
        ):
            grad = self.u.evaluate_at(pts, mode="grad")
            eps = 0.5 * (grad + np.transpose(grad, (0, 2, 1)))
            strain = np.stack(
                [eps[:, 0, 0], eps[:, 1, 1], 2 * eps[:, 0, 1]], axis=1
            )  # [ε_xx, ε_yy, γ_xy]

            if quantity == "strain":
                out = strain
            else:
                stress = strain @ self.D.T  # Voigt
                voigt = dict(
                    sxx=stress[:, 0],
                    syy=stress[:, 1],
                    sxy=stress[:, 2],
                    exx=strain[:, 0],
                    eyy=strain[:, 1],
                    exy=strain[:, 2],
                )
                out = voigt[quantity]
        else:  # displacement components
            comp = dict(u=u_val, ux=u_val[:, 0], uy=u_val[:, 1])
            out = comp[quantity]
        return (out, inside) if return_status else out