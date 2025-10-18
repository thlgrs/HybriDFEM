# tests/conftest.py
import importlib
import os
import types

import numpy as np
import pytest

TOL = 1e-9


def _import(path: str):
    return importlib.import_module(path)


def _import_attr(path: str):
    """Load 'pkg.sub:attr' or 'pkg.sub.attr'."""
    if ":" in path:
        mod, attr = path.split(":", 1)
        return getattr(_import(mod), attr)
    if "." in path:
        # last segment as attr
        mod, attr = path.rsplit(".", 1)
        return getattr(_import(mod), attr)
    raise ValueError(f"Bad import path: {path}")


@pytest.fixture(scope="session")
def impls():
    new_path = os.environ.get("NEW_IMPL")
    old_path = os.environ.get("OLD_IMPL")
    if not new_path or not old_path:
        raise RuntimeError(
            "Please set NEW_IMPL and OLD_IMPL env vars to importable modules, e.g.\n"
            "  NEW_IMPL=hybrid_fem_new.core OLD_IMPL=hybrid_fem_old.core"
        )
    return {
        "new": _import(new_path),
        "old": _import(old_path),
    }


@pytest.fixture(scope="session")
def factories(impls):
    """Return factory callables for new/old implementations."""
    out = {}

    for tag in ("new", "old"):
        fact_path = os.environ.get(f"{tag.upper()}_FACTORIES")
        if fact_path:
            mod = _import(fact_path)  # must expose create_structure, create_block
            create_structure = getattr(mod, "create_structure")
            create_block = getattr(mod, "create_block")
        else:
            # Heuristic fallback: try to find Structure and Block class
            impl = impls[tag]
            # Common names to try
            Structure = getattr(impl, "Structure", None) or getattr(impl, "Structure_2D", None)
            Block = getattr(impl, "Block_2D", None) or getattr(impl, "Block", None)

            if Structure is None:
                raise RuntimeError(f"Cannot find Structure class in {impl.__name__}. Provide {tag.upper()}_FACTORIES.")

            def create_structure():
                return Structure()

            def create_block(vertices, connect=(), ref_point=None):
                if Block is not None:
                    blk = Block()
                else:
                    # create a tiny object with attributes
                    blk = types.SimpleNamespace()

                v = np.asarray(vertices, dtype=float)
                blk.v = v
                blk.nb_vertices = len(v)
                # approximate circle for pruning
                cen = v.mean(axis=0)
                rad = float(np.max(np.linalg.norm(v - cen, axis=1)))
                blk.circle_center = cen
                blk.circle_radius = rad
                blk.ref_point = np.asarray(ref_point if ref_point is not None else (cen + np.array([0.1, 0.1])))
                blk.connect = tuple(connect)

                # attach compute_triplets if missing
                if not hasattr(blk, "compute_triplets"):
                    def compute_triplets_self():
                        list_triplets = []
                        for i in range(blk.nb_vertices - 1):
                            a, b = blk.v[i], blk.v[i + 1]
                            list_triplets.append(_make_triplet(a, b))
                        list_triplets.append(_make_triplet(blk.v[-1], blk.v[0]))
                        return list_triplets

                    def _make_triplet(a, b):
                        # normalized normal-form ABC as recommended
                        d = b - a
                        L = np.linalg.norm(d)
                        if L <= 1e-12:
                            return {"ABC": np.array([np.nan, np.nan, np.nan]), "Vertices": np.vstack([a, b])}
                        t = d / L
                        n = np.array([-t[1], t[0]])
                        C = -np.dot(n, a)
                        # canonical sign
                        if (n[0] < 0.0) or (abs(n[0]) <= 1e-12 and n[1] < 0.0):
                            n = -n;
                            C = -C
                        return {"ABC": np.array([n[0], n[1], C]), "Vertices": np.vstack([a, b])}

                    blk.compute_triplets = compute_triplets_self

                return blk

        out[tag] = types.SimpleNamespace(
            create_structure=create_structure,
            create_block=create_block,
        )

    return out


@pytest.fixture
def toy_scene(factories):
    """Build two overlapping rectangles + one separate rectangle."""
    rect1 = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    rect2 = np.array([[1.5, 0.2], [3.0, 0.2], [3.0, 0.8], [1.5, 0.8]])  # overlaps rect1
    rect3 = np.array([[5.0, 0.0], [6.0, 0.0], [6.0, 1.0], [5.0, 1.0]])  # far away

    data = {}
    for tag in ("new", "old"):
        s = factories[tag].create_structure()
        # Ensure required arrays exist if Structure expects them
        if not hasattr(s, "list_blocks"): s.list_blocks = []
        if not hasattr(s, "list_nodes"):  s.list_nodes = []
        if not hasattr(s, "dof_fix"):     s.dof_fix = np.array([], dtype=int)
        if not hasattr(s, "dof_free"):    s.dof_free = np.array([], dtype=int)
        if not hasattr(s, "nb_dofs"):     s.nb_dofs = 0
        if not hasattr(s, "P"):           s.P = np.zeros(0)
        if not hasattr(s, "P_fixed"):     s.P_fixed = np.zeros(0)

        b1 = factories[tag].create_block(rect1, connect=(1,), ref_point=np.array([-1.0, 0.5]))
        b2 = factories[tag].create_block(rect2, connect=(2,), ref_point=np.array([4.0, 0.5]))
        b3 = factories[tag].create_block(rect3, connect=(3,), ref_point=np.array([7.0, 0.5]))

        s.list_blocks = [b1, b2, b3]
        data[tag] = s
    return data, (rect1, rect2, rect3), TOL
