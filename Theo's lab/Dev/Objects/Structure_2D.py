# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:02:25 2024

@author: ibouckaert
"""

import math
# Standart imports
import os
import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.spatial import cKDTree

from .Block import Block_2D
from .ContactFace import CF_2D
from .FE import FE, Timoshenko_FE_2D, FE_2D


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"


warnings.formatwarning = custom_warning_format


class Structure_2D(ABC):
    DOF_PER_NODE = 3  # [ux, uy, rz] typical planar frame; adjust if needed

    def __init__(self):
        super().__init__()
        self.list_nodes = []

        # OPTIMISATION
        # KD-tree cache
        self._kdtree = None
        self._kdtree_n = 0
        # Optional hash map: (rounded_x, rounded_y) -> node index
        self._node_hash = None
        self._node_hash_decimals = None  # number of decimals used to round

        self.nb_dofs = None
        self.U = None
        self.P = None
        self.P_fixed = None
        self.dof_fix = None
        self.dof_free = None
        self.nb_dof_fix = None
        self.nb_dof_free = None

        self.K = None
        self.K0 = None
        self.K_LG = None
        self.P_r = None
        self.M = None

    @abstractmethod
    def make_nodes(self):
        pass

    # ============================================================
    # ---------------------- CORE HELPERS ------------------------
    # ============================================================
    def dofs_defined(self):
        if not self.nb_dofs:
            warnings.warn("The DoFs of the structure were not defined")

    # ---------------------------------------------------------
    # Get node id optimized KDTree(more nodes with 2D-elements)
    # ---------------------------------------------------------
    def build_node_tree(self):
        """
        Build (or rebuild) the KD-tree from current self.list_nodes.
        Call this after you finalize/modify the mesh if you will do many lookups.
        """
        nodes = np.asarray(self.list_nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("list_nodes must be an (N, 2) array-like for KD-tree.")
        self._kdtree = cKDTree(nodes)
        self._kdtree_n = nodes.shape[0]

    def _refresh_node_tree(self):
        if (self._kdtree is None) or (self._kdtree_n != len(self.list_nodes)):
            self.build_node_tree()

    def _refresh_node_hash(self, tol: float):
        """Create/refresh the rounding-hash using a tolerance-driven precision."""
        # decimals from tol ~ 10^{-decimals}
        decimals = max(0, int(round(-math.log10(max(tol, 1e-15)))))
        if self._node_hash is None or self._node_hash_decimals != decimals:
            self._node_hash = {}
            self._node_hash_decimals = decimals
            for idx, node in enumerate(self.list_nodes):
                key = (round(float(node[0]), decimals), round(float(node[1]), decimals))
                self._node_hash[key] = idx

    def _add_node_if_new(self, node, tol: float = 1e-9, optimized: bool = True, use_hash: bool = False):
        """
        Insert node if not already present (within tol). Return its index.

        Parameters
        ----------
        node : array-like (2,)
            Coordinates [x, y].
        tol : float
            Matching tolerance.
        optimized : bool
            If True, use KD-tree nearest query; else vectorized allclose.
        use_hash : bool
            If True, use a rounding-based hash (O(1) avg). Good for exact/snap grids.
            Hash uses 'tol' to set rounding precision and is updated on insert.
        """
        try:
            q = np.asarray(node, dtype=float).reshape(-1)
        except Exception:
            raise ValueError("node must be array-like of two numeric values")
        if q.size != 2:
            raise ValueError("node must be array-like of size 2")

        # 1) Hash path (fastest when applicable)
        if use_hash:
            self._refresh_node_hash(tol)
            decimals = self._node_hash_decimals
            key = (round(float(q[0]), decimals), round(float(q[1]), decimals))
            hit = self._node_hash.get(key)
            if hit is not None:
                return int(hit)
            # Fall through to precise check (KD or vectorized) to avoid hash collisions
            idx = self.get_node_id(q, tol=tol, optimized=optimized)
            if idx is not None:
                # ensure hash maps to the found index
                self._node_hash[key] = int(idx)
                return int(idx)
            # Not found: append and update hash
            self.list_nodes.append(q.copy())
            new_idx = len(self.list_nodes) - 1
            self._node_hash[key] = new_idx
            # Invalidate KD-tree (will rebuild lazily)
            self._kdtree = None
            self._kdtree_n = 0
            return new_idx

        # 2) No-hash path: KD-tree (optimized) or vectorized fallback
        idx = self.get_node_id(q, tol=tol, optimized=optimized)
        if idx is not None:
            return int(idx)

        # Not found: append and invalidate KD-tree cache
        self.list_nodes.append(q.copy())
        new_idx = len(self.list_nodes) - 1
        self._kdtree = None
        self._kdtree_n = 0
        # If a hash exists from a previous call, keep it coherent
        if self._node_hash is not None:
            decimals = self._node_hash_decimals
            key = (round(float(q[0]), decimals), round(float(q[1]), decimals))
            self._node_hash[key] = new_idx
        return new_idx

    def get_node_id(self, node, tol: float = 1e-8, optimized: bool = False):
        """
        Return the node index matching 'node'. If not found, returns None and warns.

        Parameters
        ----------
        node : int | array-like of shape (2,)
            - int: node index returned directly (no range check here)
            - coordinates: [x, y] to locate
        tol : float, default 1e-9
            Tolerance used for matching:
              - optimized=True (KD-tree): absolute Euclidean distance threshold, i.e.
                ||x_i - x||_2 <= tol
              - optimized=False (vectorized): allclose-style per-component tolerance
        optimized : bool, default False
            If True, use a cached KD-tree for O(log N) lookup (rebuilds lazily).
            If False, use vectorized allclose (O(N) but robust and zero setup).

        Notes
        -----
        - KD-tree uses absolute distance; pick 'tol' consistent with your model units.
        - If you frequently modify 'list_nodes', either call 'build_node_tree()'
          after modifications or let '_refresh_node_tree()' rebuild lazily.
        """
        # Fast path: integer index
        if isinstance(node, (int, np.integer)):
            return int(node)

        # Normalize coordinate-like input
        if node is None:
            warnings.warn("Input node is None")
            return None
        try:
            node_arr = np.asarray(node, dtype=float).reshape(-1)
        except Exception:
            warnings.warn("Input node should be an array-like of two numeric values")
            return None
        if node_arr.size != 2:
            warnings.warn("Input node should be an array-like of size 2")
            return None

        if optimized:
            # KD-tree path (absolute Euclidean tol)
            try:
                self._refresh_node_tree()
            except ValueError as e:
                warnings.warn(str(e))
                return None
            dist, idx = self._kdtree.query(node_arr)
            if np.isfinite(dist) and dist <= tol:
                return int(idx)
            warnings.warn("Input node not found within KD-tree tolerance")
            return None
        else:
            # Vectorized allclose path (component-wise tolerance)
            nodes = np.asarray(self.list_nodes, dtype=float)
            if nodes.ndim != 2 or nodes.shape[1] != 2:
                warnings.warn("list_nodes must be shape (N, 2) for coordinate lookup")
                return None

            # Equivalent to np.allclose row-wise, but vectorized for speed
            # allclose(a, b): |a-b| <= atol + rtol*|b|
            rtol = 1e-9
            atol = tol
            diffs = np.abs(nodes - node_arr)
            thresh = atol + rtol * np.maximum(np.abs(nodes), np.abs(node_arr))
            mask = np.all(diffs <= thresh, axis=1)
            matches = np.nonzero(mask)[0]
            if matches.size > 0:
                return int(matches[0])

            warnings.warn("Input node does not exist in list_nodes")
            return None

    def _iter_dofs(self, dofs):
        """
        Yield local DoFs (integers) from various container types.
        Accepts:
          - int
          - list / tuple (possibly nested)
          - 1D numpy arrays
        Warns on invalid types or bad entries.
        """
        if isinstance(dofs, (int, np.integer)):
            yield int(dofs)

        elif isinstance(dofs, (list, tuple)):
            for item in dofs:
                for d in self._iter_dofs(item):
                    yield d

        elif isinstance(dofs, np.ndarray) and dofs.ndim == 1:
            for item in dofs:
                if isinstance(item, (int, np.integer)):
                    yield int(item)
                else:
                    try:
                        yield int(item)
                    except Exception:
                        warnings.warn(f"Invalid DoF value inside array: {item!r}")

        else:
            warnings.warn("DoFs must be an int or a 1D iterable of ints")

    def _global_dof(self, node_id: int, local_dof: int) -> int:
        """Map (node_id, local_dof) -> global dof index."""
        return self.DOF_PER_NODE * int(node_id) + int(local_dof)

    # ============================================================
    # ------------------------ LOADING ---------------------------
    # ============================================================
    def _load_one_global_dof(self, gidx: int, force, fixed: bool = False):
        """
        Low-level: add 'force' to a single global dof index.
        """
        if fixed:
            self.P_fixed[gidx] += force
        else:
            self.P[gidx] += force

    def _apply_dofs_load(self, node_id: int, dofs, force, fixed: bool = False):
        """
        For a given node index, apply 'force' to each local dof provided.
        """
        for dof in self._iter_dofs(dofs):
            gidx = self._global_dof(node_id, dof)
            self._load_one_global_dof(gidx, force, fixed)

    def load_node(self, node_ids, dofs, force, fixed: bool = False):
        """
        Apply 'force' to one or several nodes and DoFs.

        Parameters
        ----------
        node_ids : int | list[int] | np.ndarray(size==2)
            - int: single node index
            - list[int]: multiple node indices
            - numpy array with size==2: coordinates of a single node
        dofs : int | Iterable[int] | np.ndarray(1D)
            Local dof index/indices (0..DOF_PER_NODE-1).
        force : Any
            Value(s) to add to load vector(s). Your model defines meaning/units.
        fixed : bool
            If True, add to P_fixed; else to P.
        """
        # Case 1: single node index
        if isinstance(node_ids, (int, np.integer)):
            self._apply_dofs_load(int(node_ids), dofs, force, fixed)

        # Case 2: list of node indices
        elif isinstance(node_ids, list):
            for node_id in node_ids:
                self._apply_dofs_load(int(node_id), dofs, force, fixed)

        # Case 3: coordinates array-like (size==2)
        elif isinstance(node_ids, np.ndarray) and node_ids.size == 2:
            nid = self.get_node_id(node_ids)
            if nid is None:
                warnings.warn("Input node to be loaded does not exist")
            else:
                self._apply_dofs_load(nid, dofs, force, fixed)

        # Case 4: invalid
        else:
            warnings.warn("Nodes to be loaded must be int, list of ints, or a 2D coordinate numpy array")

    def reset_loading(self):
        """Zero both load vectors."""
        self.P_fixed = np.zeros(self.nb_dofs)
        self.P = np.zeros(self.nb_dofs)

    # ============================================================
    # ------------------------- FIXING ---------------------------
    # ============================================================
    def _fix_one_global_dof(self, gidx: int):
        """
        Fix a single global DoF index:
        - append to dof_fix
        - remove from dof_free
        - update counters
        """
        self.dof_fix = np.append(self.dof_fix, gidx)
        self.dof_free = self.dof_free[self.dof_free != gidx]
        self.nb_dof_fix = int(len(self.dof_fix))
        self.nb_dof_free = int(len(self.dof_free))

    def _apply_fix_for_node(self, node_id: int, dofs):
        """
        Fix all local DoFs for a given node: gidx = DOF_PER_NODE*node_id + dof
        """
        for dof in self._iter_dofs(dofs):
            gidx = self._global_dof(node_id, dof)
            self._fix_one_global_dof(gidx)

    def fix_node(self, node_ids, dofs):
        """
        Fix DoFs on one or several nodes.

        Parameters
        ----------
        node_ids : int | list[int] | np.ndarray(size==2)
            - int: single node index
            - list[int]: multiple node indices
            - numpy array with size==2: coordinates of a single node
        dofs : int | Iterable[int] | np.ndarray(1D)
            Local dof index/indices to fix (0..DOF_PER_NODE-1).
        """
        # Case 1: single node index
        if isinstance(node_ids, (int, np.integer)):
            self._apply_fix_for_node(int(node_ids), dofs)

        # Case 2: list of node indices
        elif isinstance(node_ids, list):
            for node_id in node_ids:
                self._apply_fix_for_node(int(node_id), dofs)

        # Case 3: coordinates array-like (size==2)
        elif isinstance(node_ids, np.ndarray) and node_ids.size == 2:
            nid = self.get_node_id(node_ids)
            if nid is None:
                warnings.warn("Input node to be fixed does not exist")
            else:
                self._apply_fix_for_node(nid, dofs)

        # Case 4: invalid
        else:
            warnings.warn("Nodes to be fixed must be int, list of ints or numpy array (size==2)")

    # ===========================================================
    # -------------------- Solving methods ----------------------
    # ===========================================================
    @abstractmethod
    def get_P_r(self):
        pass

    @abstractmethod
    def get_M_str(self):
        pass

    @abstractmethod
    def get_K_str(self):
        pass

    @abstractmethod
    def get_K_str0(self):
        pass

    @abstractmethod
    def get_K_str_LG(self):
        pass

    def solve_linear(self):
        self.get_P_r()
        self.get_K_str0()

        K_ff = self.K0[np.ix_(self.dof_free, self.dof_free)]
        K_fr = self.K0[np.ix_(self.dof_free, self.dof_fix)]
        K_rf = self.K0[np.ix_(self.dof_fix, self.dof_free)]
        K_rr = self.K0[np.ix_(self.dof_fix, self.dof_fix)]

        self.U[self.dof_free] = sc.linalg.solve(
            K_ff,
            self.P[self.dof_free]
            + self.P_fixed[self.dof_free]
            - K_fr @ self.U[self.dof_fix],
        )
        # self.P[self.dof_fix] = K_rf @ self.U[self.dof_free] + K_rr @ self.U[self.dof_fix]
        self.get_P_r()

    def set_damping_properties(self, xsi=0.0, damp_type="RAYLEIGH", stiff_type="INIT"):
        if isinstance(xsi, float):
            self.xsi = [xsi, xsi]

        elif isinstance(xsi, list) and len(xsi) == 2:
            self.xsi = xsi

        self.damp_type = damp_type
        self.stiff_type = stiff_type

    def ask_method(self, Meth=None):
        if Meth is None:
            Meth = input(
                "Which method do you want to use ? CDM, CAA, LA, NWK, WIL, HHT, WBZ or GEN - Default is CDM "
            )

            if Meth == "CDM" or Meth == "":
                return Meth, None
            elif Meth == "CAA" or Meth == "NWK":
                return "NWK", {
                    "g": 1 / 2,
                    "b": 1 / 4,
                }  # If not specified run CAA by default
            elif Meth == "LA":
                return "NWK", {"g": 1 / 2, "b": 1 / 6}
            elif Meth == "NWK":
                g = input("Which value for Gamma ? - Default is 1/2")
                b = input("Which value for Beta ? - Default is 1/4")

                if g == "":
                    g = 1 / 2
                else:
                    g = float(g)
                if b == "":
                    b = 1 / 4
                else:
                    b = float(b)

                return "NWK", {"g": g, "b": b}

            elif Meth == "WIL":
                t = input("Which value for Gamma ? - Default is 1.5")
                if t == "":
                    t = 1.5
                else:
                    t = float(t)
                if t < 1:
                    warnings.warn(
                        "Theta should be larger or equal to one for Wilson's theta method"
                    )
                elif t < 1.37:
                    warnings.warn(
                        "Theta should be larger or equal to one for unconditional stability in Wilson's theta method"
                    )
                return "WIL", {"t": t}

            elif Meth == "HHT":
                a = input("Which value for Alpha ? - Default is 1/4")
                g = input("Which value for Gamma ? - Default is (1+2a)/2")
                b = input("Which value for Beta ? - Default is (1+a)^2/4")

                if a == "":
                    a = 1 / 4
                else:
                    a = float(a)
                if a < 0 or a > 1 / 3:
                    warnings.warn(
                        "Alpha should be between 0 and 1/3 for unconditional stability in HHT Method"
                    )
                if g == "":
                    g = (1 + 2 * a) / 2
                else:
                    g = float(g)
                if b == "":
                    b = (1 + a) ** 2 / 4
                else:
                    b = float(b)

                return "GEN", {"am": 0, "af": a, "g": g, "b": b}

            elif Meth == "WBZ":
                a = input("Which value for Alpha ? - Default is 1/2")
                g = input("Which value for Gamma ? - Default is (1-2a)/2")
                b = input("Which value for Beta ? - Default is 1/4")

                if a == "":
                    a = 1 / 2
                else:
                    a = float(a)
                if a > 1 / 2:
                    warnings.warn(
                        "Alpha should be smaller thann 1/2 for unconditional stability in WBZ Method"
                    )
                if g == "":
                    g = (1 - 2 * a) / 2
                else:
                    g = float(g)
                if g < (1 - 2 * a) / 2:
                    warnings.warn(
                        "Gamma should be larger than (1-2a)/2 for unconditional stability in WBZ Method"
                    )
                if b == "":
                    b = 1 / 4
                else:
                    b = float(b)
                if b < g / 2:
                    warnings.warn(
                        "Beta should be larger than g/2 for unconditional stability in WBZ Method"
                    )

                return "GEN", {"am": a, "af": 0, "g": g, "b": b}

            elif Meth == "GEN":
                m = input("Which value for Mu ? - Default is 1")

                if m == "":
                    m = 1
                else:
                    m = float(m)
                if m < 0 or m > 1:
                    warnings.warn(
                        "Mu should be between 0 and 1 for Generalized-alpha Method"
                    )

                return "GEN", {
                    "am": (2 * m - 1) / (m + 1),
                    "af": m / (m + 1),
                    "g": (3 * m - 1) / (2 * (m + 1)),
                    "b": (m / (m + 1)) ** 2,
                }

        elif isinstance(Meth, str):
            if Meth == "CDM":
                return Meth, {}
            elif Meth == "CAA" or Meth == "NWK":
                return "NWK", {
                    "g": 1 / 2,
                    "b": 1 / 4,
                }  # If not specified run CAA by default
            elif Meth == "LA":
                return "NWK", {"g": 1 / 2, "b": 1 / 6}
            elif Meth == "NWK":
                return "GEN", {"am": 0, "af": 0, "g": 1 / 2, "b": 1 / 4}
            elif Meth == "WIL":
                return "WIL", {"t": 1.5}
            elif Meth == "HHT":
                return "GEN", {"am": 0, "af": 0, "g": 1 / 2, "b": 1 / 4}
            elif Meth == "WBZ":
                return "GEN", {"am": 0, "af": 0, "g": 1 / 2, "b": 1 / 4}
            elif Meth == "GEN":
                m = 1
                return "GEN", {
                    "am": (2 * m - 1) / (m + 1),
                    "af": m / (m + 1),
                    "g": (3 * m - 1) / (2 * (m + 1)),
                    "b": (m / (m + 1)) ** 2,
                }

        elif isinstance(Meth, list):
            if Meth[0] == "NWK":
                if len(Meth) != 3:
                    warnings.warn("Requiring 2 parameters for Newmark method")

                g = Meth[1]
                b = Meth[2]

                return "GEN", {"am": 0, "af": 0, "g": g, "b": b}

            elif Meth[0] == "WIL":
                if len(Meth) != 2:
                    warnings.warn("Requiring 1 parameters for Wilson's theta method")

                t = Meth[1]
                if t < 1:
                    warnings.warn(
                        "Theta should be larger or equal to one for Wilson's theta method"
                    )
                elif t < 1.37:
                    warnings.warn(
                        "Theta should be larger or equal to one for unconditional stability in Wilson's theta method"
                    )
                return "WIL", {"t": t}

            elif Meth[0] == "HHT":
                if len(Meth) == 2:
                    a = Meth[1]
                    g = (1 + 2 * a) / 2
                    b = (1 + a) ** 2 / 4

                elif len(Meth) == 4:
                    a = Meth[1]
                    g = Meth[2]
                    b = Meth[3]

                else:
                    warnings.warn("Requiring 3 parameters for HHT method")

                if a < 0 or a > 1 / 3:
                    warnings.warn(
                        "Alpha should be between 0 and 1/3 for unconditional stability in HHT Method"
                    )

                return "GEN", {"am": 0, "af": a, "g": g, "b": b}

            elif Meth[0] == "WBZ":
                if len(Meth) != 4:
                    warnings.warn("Requiring 3 parameters for WBZ method")

                a = Meth[1]
                g = Meth[2]
                b = Meth[3]

                if a > 1 / 2:
                    warnings.warn(
                        "Alpha should be smaller thann 1/2 for unconditional stability in WBZ Method"
                    )
                if g < (1 - 2 * a) / 2:
                    warnings.warn(
                        "Gamma should be larger than (1-2a)/2 for unconditional stability in WBZ Method"
                    )
                if b < g / 2:
                    warnings.warn(
                        "Beta should be larger than g/2 for unconditional stability in WBZ Method"
                    )

                return "GEN", {"am": a, "af": 0, "g": g, "b": b}

            elif Meth[0] == "GEN":
                if len(Meth) != 2:
                    warnings.warn("Requiring 1 parameters for Generalized Alpha method")

                m = Meth[1]

                if m < 0 or m > 1:
                    warnings.warn(
                        "Mu should be between 0 and 1 for Generalized-alpha Method"
                    )

                return "GEN", {
                    "am": (2 * m - 1) / (m + 1),
                    "af": m / (m + 1),
                    "g": (3 * m - 1) / (2 * (m + 1)),
                    "b": (m / (m + 1)) ** 2,
                }

        return None, None

    def save_structure(self, filename):
        import pickle

        with open(filename + ".pkl", "wb") as file:
            pickle.dump(self, file)

    @abstractmethod
    def set_lin_geom(self, lin_geom=True):
        pass


class Structure_block(Structure_2D):
    def __init__(self, listBlocks: Union[List[Block_2D], None] = None):
        super().__init__()
        self.list_blocks = listBlocks or []

    # Construction methods
    def add_block(self, vertices, rho, b=1, material=None, ref_point=None):
        self.list_blocks.append(
            Block_2D(vertices, rho, b=b, material=material, ref_point=ref_point)
        )

    def add_beam(
            self, N1, N2, n_blocks, h, rho, b=1, material=None, end_1=True, end_2=True
    ):
        lx = N2[0] - N1[0]
        ly = N2[1] - N1[1]
        L = np.sqrt(lx ** 2 + ly ** 2)
        L_b = L / (n_blocks - 1)

        long = np.array([lx, ly]) / L
        tran = np.array([-ly, lx]) / L

        # Loop to create the blocks
        ref_point = N1.copy()

        for i in np.arange(n_blocks):
            # Initialize array of vertices
            vertices = np.array([ref_point, ref_point, ref_point, ref_point])

            if i == 0:  # First block is half block
                vertices[0] += L_b / 2 * long - h / 2 * tran
                vertices[1] += L_b / 2 * long + h / 2 * tran
                vertices[2] += h / 2 * tran
                vertices[3] += -h / 2 * tran

                if end_1:
                    ref = ref_point
                else:
                    ref = None

            elif i == n_blocks - 1:  # Last block is also a half_block
                vertices[0] += -h / 2 * tran
                vertices[1] += h / 2 * tran
                vertices[2] += h / 2 * tran - L_b / 2 * long
                vertices[3] += -h / 2 * tran - L_b / 2 * long

                if end_2:
                    ref = ref_point
                else:
                    ref = None

            else:
                vertices[0] += -h / 2 * tran + L_b / 2 * long
                vertices[1] += h / 2 * tran + L_b / 2 * long
                vertices[2] += h / 2 * tran - L_b / 2 * long
                vertices[3] += -h / 2 * tran - L_b / 2 * long
                ref = None

            self.add_block(vertices, rho, b=b, material=material, ref_point=ref)

            ref_point += L_b * long

    def add_tapered_beam(
            self,
            N1,
            N2,
            n_blocks,
            h1,
            h2,
            rho,
            b=1,
            material=None,
            contact=None,
            end_1=True,
            end_2=True,
    ):
        lx = N2[0] - N1[0]
        ly = N2[1] - N1[1]
        L = np.sqrt(lx ** 2 + ly ** 2)
        L_b = L / (n_blocks - 1)

        heights = np.linspace(h1, h2, n_blocks)
        d_h = (heights[1] - heights[0]) / 2

        long = np.array([lx, ly]) / L
        tran = np.array([-ly, lx]) / L

        # Loop to create the blocks
        ref_point = N1.copy()

        for i in np.arange(n_blocks):
            # Initialize array of vertices
            vertices = np.array([ref_point, ref_point, ref_point, ref_point])

            if i == 0:  # First block is half block
                h1 = heights[i]
                h2 = heights[i] + d_h
                vertices[0] += L_b / 2 * long - h2 / 2 * tran
                vertices[1] += L_b / 2 * long + h2 / 2 * tran
                vertices[2] += h1 / 2 * tran
                vertices[3] += -h1 / 2 * tran

                if end_1:
                    ref = ref_point
                else:
                    ref = None

            elif i == n_blocks - 1:  # Last block is also a half_block
                h2 = heights[i]
                h1 = heights[i] - d_h
                vertices[0] += -h2 / 2 * tran
                vertices[1] += h2 / 2 * tran
                vertices[2] += h1 / 2 * tran - L_b / 2 * long
                vertices[3] += -h1 / 2 * tran - L_b / 2 * long

                if end_2:
                    ref = ref_point
                else:
                    ref = None

            else:
                h1 = heights[i] - d_h
                h2 = heights[i] + d_h
                vertices[0] += -h2 / 2 * tran + L_b / 2 * long
                vertices[1] += h2 / 2 * tran + L_b / 2 * long
                vertices[2] += h1 / 2 * tran - L_b / 2 * long
                vertices[3] += -h1 / 2 * tran - L_b / 2 * long
                ref = None

            self.add_block(vertices, rho, b=b, material=material, ref_point=ref)

            ref_point += L_b * long

    def add_arch(
            self, c, a1, a2, R, n_blocks, h, rho, b=1, material=None, contact=None
    ):
        d_a = (a2 - a1) / (n_blocks)
        angle = a1

        R_int = R - h / 2
        R_out = R + h / 2

        for i in np.arange(n_blocks):
            # Initialize array of vertices
            vertices = np.array([c, c, c, c])

            unit_dir_1 = np.array([np.cos(angle), np.sin(angle)])
            unit_dir_2 = np.array([np.cos(angle + d_a), np.sin(angle + d_a)])
            vertices[0] += R_int * unit_dir_1
            vertices[1] += R_out * unit_dir_1
            vertices[2] += R_out * unit_dir_2
            vertices[3] += R_int * unit_dir_2

            # print(vertices)
            self.add_block(vertices, rho, b=b, material=material)

            angle += d_a

    def add_wall(
            self, c1, l_block, h_block, pattern, rho, b=1, material=None, orientation=None
    ):
        if orientation is not None:
            long = orientation
            tran = np.array([-orientation[1], orientation[0]])
        else:
            long = np.array([1, 0], dtype=float)
            tran = np.array([0, 1], dtype=float)

        for j, line in enumerate(pattern):
            ref_point = (
                    c1 + 0.5 * abs(line[0]) * l_block * long + (j + 0.5) * h_block * tran
            )

            for i, brick in enumerate(line):
                if brick > 0:
                    vertices = np.array([ref_point, ref_point, ref_point, ref_point])
                    vertices[0] += brick * l_block / 2 * long - h_block / 2 * tran
                    vertices[1] += brick * l_block / 2 * long + h_block / 2 * tran
                    vertices[2] += -brick * l_block / 2 * long + h_block / 2 * tran
                    vertices[3] += -brick * l_block / 2 * long - h_block / 2 * tran

                    self.add_block(vertices, rho, b=b, material=material)

                if not i == len(line) - 1:
                    ref_point += 0.5 * l_block * long * (abs(brick) + abs(line[i + 1]))

    def add_voronoi_surface(self, surface, list_of_points, rho, b=1, material=None):
        # Surface is a list of points defining the surface to be subdivided into
        # Voronoi cells.

        def point_in_surface(point, surface):
            # Check if a point lies on the surface
            # Surface is a list of points delimiting the surface
            # Point is a 2D numpy array

            n = len(surface)

            for i in range(n):
                A = surface[i]
                B = surface[(i + 1) % n]
                C = point

                if np.cross(B - A, C - A) < 0:
                    return False

            return True

        for point in list_of_points:
            # Check if all points lie on the surface
            if not point_in_surface(point, surface):
                warnings.warn("Not all points lie on the surface")
                return

        # Create Voronoi cells
        vor = sc.spatial.Voronoi(list_of_points)

        # Create block for each Voronoi region
        # If region is finite, it's easy
        # If region is infinite, delimit it with the edge of the surface
        for region in vor.regions[1:]:
            if not -1 in region:
                vertices = np.array([vor.vertices[i] for i in region])
                self.add_block(vertices, rho, b=b, material=material)

            else:
                vertices = []
                for i in region:
                    if not i == -1:
                        vertices.append(vor.vertices[i])

                # Find the edges of the surface that intersect the infinite cell
                for i in range(len(vertices)):
                    A = vertices[i]
                    B = vertices[(i + 1) % len(vertices)]

                    for j in range(len(surface)):
                        C = surface[j]
                        D = surface[(j + 1) % len(surface)]

                        if np.cross(B - A, C - A) * np.cross(B - A, D - A) < 0:
                            # Intersection between AB and CD
                            if np.cross(D - C, A - C) * np.cross(D - C, B - C) < 0:
                                # Intersection between CD and AB
                                vertices.insert(
                                    i + 1,
                                    C
                                    + np.cross(D - C, A - C)
                                    / np.cross(D - C, B - C)
                                    * (B - A),
                                )
                                vertices.insert(i + 2, D)
                                break

                self.add_block(np.array(vertices), rho, b=b, material=material)

    # Generation methods
    def _make_nodes_block(self):
        for block in self.list_blocks:
            index = self._add_node_if_new(block.ref_point)
            block.make_connect(index)

    def make_nodes(self):
        self._make_nodes_block()
        self.nb_dofs = 3 * len(self.list_nodes)
        self.U = np.zeros(self.nb_dofs, dtype=float)
        self.P = np.zeros(self.nb_dofs, dtype=float)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=float)
        self.dof_fix = np.array([], dtype=int)
        self.dof_free = np.arange(self.nb_dofs, dtype=int)
        self.nb_dof_fix = 0
        self.nb_dof_free = self.nb_dofs

    def detect_interfaces(self, eps=1e-9, margin=0.01):
        def overlap_colinear(seg1, seg2, eps=1e-9):
            """
            seg1, seg2: each as np.ndarray shape (2,2), rows are endpoints [[x1,y1],[x2,y2]]
            Return (has_overlap: bool, overlap_endpoints: np.ndarray shape (2,2) or None)
            """
            p1, p2 = np.asarray(seg1, float)
            q1, q2 = np.asarray(seg2, float)

            v = p2 - p1
            Lv = np.linalg.norm(v)
            if Lv <= eps:
                return False, None  # degenerate segment
            u = v / Lv  # unit direction

            # project all endpoints onto u, using p1 as origin
            tp = np.array([0.0, Lv])  # p1->p1, p1->p2
            tq = np.array([np.dot(q1 - p1, u), np.dot(q2 - p1, u)])
            tq.sort()

            t_start = max(tp.min(), tq.min())
            t_end = min(tp.max(), tq.max())

            if t_end - t_start < -eps:
                return False, None

            a = p1 + t_start * u
            b = p1 + t_end * u
            return True, np.vstack([a, b])

        def are_colinear(p1, p2, q1, q2, eps=1e-9):
            v = p2 - p1
            w1 = q1 - p1
            w2 = q2 - p1

            # 2D “cross product” magnitude for (a,b)×(c,d) := a*d - b*c
            def cross(a, b): return a[0] * b[1] - a[1] * b[0]

            return (abs(cross(v, w1)) <= eps) and (abs(cross(v, w2)) <= eps)

        def circles_separated_sq(c1, r1, c2, r2, margin=0.01):
            # return True if centers are farther than (r1+r2)*(1+margin)
            d2 = np.sum((c1 - c2) ** 2)
            thr = (r1 + r2) ** 2 * (1.0 + margin) ** 2
            return d2 >= thr

        # prefetch
        blocks = self.list_blocks
        B = len(blocks)
        triplets = [blk.compute_triplets() for blk in blocks]

        interfaces = []
        self.interf_counter = 0

        for i in range(B):
            cand = blocks[i]
            for j in range(i + 1, B):
                anta = blocks[j]

                # 1) quick prunes
                if cand.connect == anta.connect:
                    continue
                if circles_separated_sq(cand.circle_center, cand.circle_radius,
                                        anta.circle_center, anta.circle_radius,
                                        margin=margin):
                    continue

                # 2) test edges on the same line
                ifaces_ij = []
                for t1 in triplets[i]:
                    A1, B1, C1 = t1["ABC"]
                    P = np.asarray(t1["Vertices"], float)  # shape (2,2)
                    for t2 in triplets[j]:
                        if not np.allclose(t1["ABC"], t2["ABC"], rtol=1e-8, atol=eps):
                            continue
                        Q = np.asarray(t2["Vertices"], float)

                        # now both segments lie on the same infinite line; check finite overlap
                        has, seg = overlap_colinear(P, Q, eps=eps)
                        if not has:
                            continue

                        a, b = seg  # endpoints
                        u = (b - a)
                        Lu = np.linalg.norm(u)
                        if Lu <= eps:  # zero-length overlap
                            continue
                        u /= Lu
                        n = np.array([-u[1], u[0]])  # left-hand normal
                        # Decide block A vs B via normal direction
                        if np.dot(cand.ref_point - a, n) > 0:
                            blA, blB = cand, anta
                        else:
                            blA, blB = anta, cand
                        ifaces_ij.append({
                            "Block A": blA,
                            "Block B": blB,
                            "x_e1": a,
                            "x_e2": b,
                            # (optionally keep unit vectors if useful)
                            # "tangent": u, "normal": n
                        })

                self.interf_counter += 1
                if ifaces_ij:
                    interfaces.extend(ifaces_ij)

        return interfaces

    def make_cfs(self, lin_geom, nb_cps=2, offset=-1, contact=None, surface=None, weights=None, interfaces=None):
        if interfaces is None:
            interfaces = self.detect_interfaces()
        self.list_cfs = []
        for i, face in enumerate(interfaces):
            cf = CF_2D(face, nb_cps, lin_geom, offset=offset, contact=contact, surface=surface, weights=weights)
            self.list_cfs.append(cf)
            cf.bl_A.cfs.append(i)
            cf.bl_B.cfs.append(i)

    # Solving methods
    def _get_P_r_block(self):
        self.dofs_defined()
        if not hasattr(self, "P_r"):
            self.P_r = np.zeros(self.nb_dofs, dtype=float)

        for CF in self.list_cfs:
            qf_glob = np.zeros(6)
            qf_glob[:3] = self.U[CF.bl_A.dofs]
            qf_glob[3:] = self.U[CF.bl_B.dofs]
            pf_glob = CF.get_pf_glob(qf_glob)
            self.P_r[CF.bl_A.dofs] += pf_glob[:3]
            self.P_r[CF.bl_B.dofs] += pf_glob[3:]

    def get_P_r(self):
        self.P_r = np.zeros(self.nb_dofs, dtype=float)
        self._get_P_r_block()

    def _mass_block(self, no_inertia: bool = False):
        for block in getattr(self, "list_blocks", []):
            # block mass matrix must align with block.dofs length
            M_block = block.get_mass(no_inertia=no_inertia)
            dofs = np.asarray(block.dofs, dtype=int)
            self.M[np.ix_(dofs, dofs)] += M_block

    def get_M_str(self, no_inertia: bool = False):
        self.dofs_defined()
        self.M = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._mass_block(no_inertia=no_inertia)
        return self.M

    def _stiffness_block(self):
        for CF in getattr(self, "list_cfs", []):
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob = CF.get_kf_glob()

            self.K[np.ix_(dof1, dof1)] += kf_glob[:3, :3]
            self.K[np.ix_(dof1, dof2)] += kf_glob[:3, 3:]
            self.K[np.ix_(dof2, dof1)] += kf_glob[3:, :3]
            self.K[np.ix_(dof2, dof2)] += kf_glob[3:, 3:]

    def _stiffness0_block(self):
        for CF in getattr(self, "list_cfs", []):
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob0 = CF.get_kf_glob0()

            self.K0[np.ix_(dof1, dof1)] += kf_glob0[:3, :3]
            self.K0[np.ix_(dof1, dof2)] += kf_glob0[:3, 3:]
            self.K0[np.ix_(dof2, dof1)] += kf_glob0[3:, :3]
            self.K0[np.ix_(dof2, dof2)] += kf_glob0[3:, 3:]

    def _stiffness_LG_block(self):
        for CF in getattr(self, "list_cfs", []):
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob_LG = CF.get_kf_glob_LG()

            self.K_LG[np.ix_(dof1, dof1)] += kf_glob_LG[:3, :3]
            self.K_LG[np.ix_(dof1, dof2)] += kf_glob_LG[:3, 3:]
            self.K_LG[np.ix_(dof2, dof1)] += kf_glob_LG[3:, :3]
            self.K_LG[np.ix_(dof2, dof2)] += kf_glob_LG[3:, 3:]

    def get_K_str(self):
        self.dofs_defined()
        self.K = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_block()
        return self.K

    def get_K_str0(self):
        self.dofs_defined()
        self.K0 = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness0_block()
        return self.K0

    def get_K_str_LG(self):
        self.dofs_defined()
        self.K_LG = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_LG_block()
        return self.K_LG

    def set_lin_geom(self, lin_geom=True):
        for cf in self.list_cfs:
            cf.set_lin_geom(lin_geom)

    def get_C_str(self):
        if not (hasattr(self, "K")):
            self.get_K_str()
        # if not (hasattr(self, 'M')): self.get_M_str()

        if not hasattr(self, "damp_coeff"):
            # No damping
            if self.xsi[0] == 0 and self.xsi[1] == 0:
                self.damp_coeff = np.zeros(2)

            elif self.damp_type == "RAYLEIGH":
                try:
                    self.solve_modal(modes=2, save=False, initial=True)
                except Exception:
                    self.solve_modal(save=False, initial=True)

                A = np.array(
                    [
                        [1 / self.eig_vals[0], self.eig_vals[0]],
                        [1 / self.eig_vals[1], self.eig_vals[1]],
                    ]
                )

                if isinstance(self.xsi, float):
                    self.xsi = [self.xsi, self.xsi]
                    self.damp_coeff = 2 * sc.linalg.solve(A, np.array(self.xsi))

                if isinstance(self.xsi, list) and len(self.xsi) == 2:
                    self.damp_coeff = 2 * sc.linalg.solve(A, np.array(self.xsi))

                else:
                    warnings.warn(
                        "Xsi is not a list of two damping ratios for Rayleigh damping"
                    )

            elif self.damp_type == "STIFF":
                if not hasattr(self, "eig_vals"):
                    try:
                        self.solve_modal(modes=1, save=False, initial=True)
                    except Exception:
                        self.solve_modal(save=False, initial=True)
                self.damp_coeff = np.array([0, 2 * self.xsi[0] / self.eig_vals[0]])

            elif self.damp_type == "MASS":
                try:
                    self.solve_modal(modes=1, save=False, initial=True)
                except Exception:
                    self.solve_modal(save=False, initial=True)
                self.damp_coeff = np.array([2 * self.xsi[0] * self.eig_vals[0], 0])
                print(self.damp_coeff)

        if self.stiff_type == "INIT":
            if not (hasattr(self, "C")):
                self.get_K_str0()
                self.C = self.damp_coeff[0] * self.M + self.damp_coeff[1] * self.K0

        elif self.stiff_type == "TAN":
            self.get_K_str()
            self.C = self.damp_coeff[0] * self.M + self.damp_coeff[1] * self.K

        elif self.stiff_type == "TAN_LG":
            self.get_K_str_LG()
            self.C = self.damp_coeff[0] * self.M + self.damp_coeff[1] * self.K_LG

    def commit(self):
        for CF in self.list_cfs:
            CF.commit()

    def revert_commit(self):
        for CF in self.list_cfs:
            CF.revert_commit()

    def solve_forcecontrol(
            self,
            steps,
            tol=1,
            stiff="tan",
            max_iter=25,
            filename="Results_ForceControl",
            dir_name="",
    ):
        time_start = time.time()

        if isinstance(steps, list):
            nb_steps = len(steps) - 1
            lam = steps

        elif isinstance(steps, int):
            nb_steps = steps
            lam = np.linspace(0, 1, nb_steps + 1)

        else:
            warnings.warn(
                "Steps of the simulation should be either a list or a number of steps (int)"
            )

        # Displacements, forces and stiffness
        U_conv = np.zeros((self.nb_dofs, nb_steps + 1), dtype=float)
        P_r_conv = np.zeros((self.nb_dofs, nb_steps + 1), dtype=float)
        save_k = False
        if save_k:
            K_conv = np.zeros((self.nb_dofs, self.nb_dofs, nb_steps + 1), dtype=float)

        self.get_P_r()
        self.get_K_str()
        self.get_K_str0()
        U_conv[:, 0] = deepcopy(self.U)
        P_r_conv[:, 0] = deepcopy(self.P_r)
        if save_k:
            K_conv[:, :, 0] = deepcopy(self.K)

        # Parameters of the simulation
        iter_counter = np.zeros(nb_steps)
        res_counter = np.zeros(nb_steps)

        non_conv = False

        for i in range(1, nb_steps + 1):
            converged = False
            iteration = 0

            P_target = lam[i] * self.P + self.P_fixed
            R = P_target[self.dof_free] - self.P_r[self.dof_free]

            while not converged:
                # print(self.K[np.ix_(self.dof_free, self.dof_free)])

                try:
                    if (
                            np.linalg.cond(self.K[np.ix_(self.dof_free, self.dof_free)])
                            < 1e12
                    ):
                        dU = sc.linalg.solve(
                            self.K[np.ix_(self.dof_free, self.dof_free)], R
                        )
                    else:
                        try:
                            dU = sc.linalg.solve(
                                K_conv[:, :, i - 1][
                                    np.ix_(self.dof_free, self.dof_free)
                                ],
                                R,
                            )
                        except Exception:
                            dU = sc.linalg.solve(
                                self.K0[np.ix_(self.dof_free, self.dof_free)], R
                            )

                except np.linalg.LinAlgError:
                    warnings.warn("The tangent and initial stiffnesses are singular")

                self.U[self.dof_free] += dU

                try:
                    self.get_P_r()
                except Exception as e:
                    non_conv = True
                    iteration = max_iter + 1
                    print(e)
                    break
                self.get_K_str()
                # print(self.P_r[self.dof_free])

                R = P_target[self.dof_free] - self.P_r[self.dof_free]
                res = np.linalg.norm(R)

                # print(res)
                if res < tol:
                    converged = True
                    # self.plot_structure(scale=20, plot_cf=True, plot_forces=False)
                else:
                    # self.revert_commit()
                    iteration += 1

                if iteration > max_iter:
                    non_conv = True
                    print(f"Method did not converge at step {i}")
                    break

            if non_conv:
                break

            else:
                self.commit()
                res_counter[i - 1] = res
                iter_counter[i - 1] = iteration
                last_conv = i

                U_conv[:, i] = deepcopy(self.U)
                P_r_conv[:, i] = deepcopy(self.P_r)
                if save_k:
                    K_conv[:, :, i] = deepcopy(self.K)

                print(f"Force increment {i} converged after {iteration + 1} iterations")

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(
            f"Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file"
        )

        filename = filename + ".h5"
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("U_conv", data=U_conv)
            hf.create_dataset("P_r_conv", data=P_r_conv)
            if save_k:
                hf.create_dataset("K_conv", data=K_conv)
            hf.create_dataset("Residuals", data=res_counter)
            hf.create_dataset("Iterations", data=iter_counter)
            hf.create_dataset("Last_conv", data=last_conv)
            hf.create_dataset("Lambda", data=lam)

            hf.attrs["Descr"] = "Results of the force_control simulation"
            hf.attrs["Tolerance"] = tol
            # hf.attrs['Lambda'] = lam
            hf.attrs["Simulation_Time"] = total_time

    def solve_dispcontrol(
            self,
            steps,
            disp,
            node,
            dof,
            tol=1,
            stiff="tan",
            max_iter=25,
            filename="Results_DispControl",
            dir_name="",
    ):
        time_start = time.time()

        if isinstance(steps, list):
            nb_steps = len(steps) - 1
            lam = [step / max(steps, key=abs) for step in steps]
            d_c = steps

        elif isinstance(steps, int):
            nb_steps = steps
            lam = np.linspace(0, 1, nb_steps + 1)
            # print(lam)
            d_c = lam * disp

        else:
            warnings.warn(
                "Steps of the simulation should be either a list or a number of steps (int)"
            )
        # Displacements, forces and stiffness

        U_conv = np.zeros((self.nb_dofs, nb_steps + 1), dtype=float)
        P_r_conv = np.zeros((self.nb_dofs, nb_steps + 1), dtype=float)
        save_k = False
        if save_k:
            K_conv = np.zeros((self.nb_dofs, self.nb_dofs, nb_steps + 1), dtype=float)

        self.get_P_r()
        self.get_K_str()
        self.get_K_str0()
        # print('K', self.K[np.ix_(self.dof_free, self.dof_free)])

        U_conv[:, 0] = deepcopy(self.U)
        P_r_conv[:, 0] = deepcopy(self.P_r)
        if save_k:
            K_conv[:, :, 0] = deepcopy(self.K0)

        # Parameters of the simulation
        iter_counter = np.zeros(nb_steps)
        res_counter = np.zeros(nb_steps)
        last_conv = 0
        if isinstance(node, int):
            control_dof = [3 * node + dof]
        elif isinstance(node, list):
            control_dof = []
            for n in node:
                control_dof.append(3 * n + dof)
        other_dofs = self.dof_free[self.dof_free != control_dof]

        self.list_norm_res = [[] for _ in range(nb_steps)]
        self.list_residual = [[] for _ in range(nb_steps)]

        P_f = self.P[other_dofs].reshape(len(other_dofs), 1)
        P_c = self.P[control_dof]
        K_ff_conv = self.K0[np.ix_(other_dofs, other_dofs)]
        K_cf_conv = self.K0[control_dof, other_dofs]
        K_fc_conv = self.K0[other_dofs, control_dof]
        K_cc_conv = self.K0[control_dof, control_dof]

        for i in range(1, nb_steps + 1):
            converged = False
            iteration = 0
            non_conv = False

            lam[i] = lam[i - 1]
            dU_c = d_c[i] - d_c[i - 1]

            R = -self.P_r + lam[i] * self.P + self.P_fixed

            Rf = R[other_dofs]
            Rc = R[control_dof]

            # print('R0', R[self.dof_free])

            while not converged:
                K_ff = self.K[np.ix_(other_dofs, other_dofs)]
                K_cf = self.K[control_dof, other_dofs]
                K_fc = self.K[other_dofs, control_dof]
                K_cc = self.K[control_dof, control_dof]

                # if i >= 40:
                #     ratio = .5
                #     K_ff = ratio * self.K0[np.ix_(other_dofs, other_dofs)] + (1-ratio) * self.K[np.ix_(other_dofs, other_dofs)]
                #     K_cf = ratio * self.K0[control_dof, other_dofs] + (1-ratio)  * self.K[control_dof, other_dofs]
                #     K_fc = ratio * self.K0[other_dofs, control_dof] + (1-ratio)  * self.K[other_dofs, control_dof]
                #     K_cc = ratio * self.K0[control_dof, control_dof] + (1-ratio)  * self.K[control_dof, control_dof]

                # if i >= 20:

                #     self.get_K_str_LG()
                #     K_ff = self.K_LG[np.ix_(other_dofs, other_dofs)]
                #     K_cf = self.K_LG[control_dof, other_dofs]
                #     K_fc = self.K_LG[other_dofs, control_dof]
                #     K_cc = self.K_LG[control_dof, control_dof]

                # print('K', np.around(self.K[np.ix_(self.dof_free, self.dof_free)],5))
                #
                solver = np.block([[K_ff, -P_f], [K_cf, -P_c]])
                solution = np.append(Rf - dU_c * K_fc, Rc - dU_c * K_cc)

                # print(np.around(solver, 10))

                try:
                    if np.linalg.cond(solver) < 1e10:
                        dU_dl = np.linalg.solve(solver, solution)

                    else:
                        solver = np.block([[K_ff_conv, -P_f], [K_cf_conv, -P_c]])
                        solution = np.append(
                            Rf - dU_c * K_fc_conv, Rc - dU_c * K_cc_conv
                        )

                        dU_dl = np.linalg.solve(solver, solution)

                except Exception as e:
                    non_conv = True
                    iteration = max_iter + 1
                    print(e)
                    break

                    # warnings.warn(f'Iteration {iteration} {i} - Tangent stiffness is singular. Trying with initial stiffness')

                # Update solution and state determination
                lam[i] += dU_dl[-1]
                self.U[other_dofs] += dU_dl[:-1]
                self.U[control_dof] += dU_c

                try:
                    self.get_P_r()
                    self.get_K_str()
                except Exception as e:
                    non_conv = True
                    iteration = max_iter + 1
                    print(e)
                    break

                R = -self.P_r + lam[i] * self.P + self.P_fixed
                Rf = R[other_dofs]
                Rc = R[control_dof]

                res = np.linalg.norm(R[self.dof_free])

                if res < tol:
                    converged = True
                    self.commit()

                    list_blocks_yielded = []
                    for cf in self.list_cfs:
                        for cp in cf.cps:
                            if cp.sp1.law.tag == "STC" and cp.sp2.law.tag == "STC":
                                if cp.sp1.law.yielded or cp.sp2.law.yielded:
                                    list_blocks_yielded.append(cf.bl_A.connect)
                                    list_blocks_yielded.append(cf.bl_B.connect)

                    for cf in self.list_cfs:
                        for cp in cf.cps:
                            if cp.sp1.law.tag == "BSTC" and cp.sp2.law.tag == "BSTC":
                                if (
                                        cf.bl_A.connect in list_blocks_yielded
                                        or cf.bl_B.connect in list_blocks_yielded
                                ):
                                    # print('Reducing')
                                    cp.sp1.law.reduced = True
                                    cp.sp2.law.reduced = True

                else:
                    # self.revert_commit()
                    iteration += 1
                    dU_c = 0

                if iteration > max_iter and not converged:
                    non_conv = True
                    print(f"Method did not converge at Increment {i}")
                    break

            if non_conv:
                self.U = U_conv[:, last_conv]
                break
                # self.U = U_conv[:,last_conv]

            if converged:
                # if i < 9:
                K_ff_conv = K_ff.copy()
                K_cf_conv = K_cf.copy()
                K_fc_conv = K_fc.copy()
                K_cc_conv = K_cc.copy()
                # self.plot_structure(scale=20, plot_cf=True, plot_forces=False)
                # else:
                # print('Vertical disp', np.around(self.U[-2],15))
                # self.commit()
                # self.plot_structure(scale=1, plot_cf=True, plot_supp=False, plot_forces=False)
                res_counter[i - 1] = res
                iter_counter[i - 1] = iteration
                last_conv = i

                U_conv[:, i] = deepcopy(self.U)
                P_r_conv[:, i] = deepcopy(self.P_r)
                if save_k:
                    K_conv[:, :, i] = deepcopy(self.K)

                print(f"Disp. Increment {i} converged after {iteration + 1} iterations")

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(
            f"Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file"
        )

        filename = filename + ".h5"
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("U_conv", data=U_conv)
            hf.create_dataset("P_r_conv", data=P_r_conv)
            if save_k:
                hf.create_dataset("K_conv", data=K_conv)
            hf.create_dataset("Residuals", data=res_counter)
            hf.create_dataset("Iterations", data=iter_counter)
            hf.create_dataset("Last_conv", data=last_conv)
            hf.create_dataset("Control_Disp", data=d_c)
            hf.create_dataset("Lambda", data=lam)

            hf.attrs["Descr"] = "Results of the force_control simulation"
            hf.attrs["Tolerance"] = tol
            hf.attrs["Simulation_Time"] = total_time

    def impose_dyn_excitation(self, node, dof, U_app, dt):
        if 3 * node + dof not in self.dof_fix:
            warnings.warn("Excited DoF should be a fixed one")

        if not hasattr(self, "dof_moving"):
            self.dof_moving = []
            self.disp_histories = []
            self.times = []

        self.dof_moving.append(3 * node + dof)
        self.disp_histories.append(U_app)
        self.times.append(dt)

        # Later, add a function to interpolate when different timesteps are used.

    def solve_dyn_linear(
            self, T, dt, U0=None, V0=None, lmbda=None, Meth=None, filename="", dir_name=""
    ):
        time_start = time.time()

        self.get_K_str0()
        self.get_M_str()
        self.get_C_str()

        if U0 is None:
            if np.linalg.norm(self.U) == 0:
                U0 = np.zeros(self.nb_dofs)
            else:
                U0 = deepcopy(self.U)

        if V0 is None:
            V0 = np.zeros(self.nb_dofs)

        if hasattr(self, "times"):
            for timestep in self.times:
                if timestep != dt:
                    warnings.warn(
                        "Unmatching timesteps between excitation and simulation"
                    )
            for i, disp in enumerate(self.disp_histories):
                if U0[self.dof_moving[i]] != disp[0]:
                    warnings.warn("Unmatching initial displacements")
        else:
            self.dof_moving = []
            self.disp_histories = []
            self.times = []

        Time = np.arange(0, T, dt, dtype=float)
        Time = np.append(Time, T)
        nb_steps = len(Time)

        loading = np.zeros(nb_steps)

        if callable(lmbda):
            for i, t in enumerate(Time):
                loading[i] = lmbda(t)
        elif isinstance(lmbda, list):
            pass

        U_conv = np.zeros((self.nb_dofs, nb_steps))
        V_conv = np.zeros((self.nb_dofs, nb_steps))
        A_conv = np.zeros((self.nb_dofs, nb_steps))
        P_conv = np.zeros((self.nb_dofs, nb_steps))

        U_conv[:, 0] = U0.copy()
        V_conv[:, 0] = V0.copy()
        A_conv[:, 0] = sc.linalg.solve(
            self.M, loading[0] * self.P - self.C @ V_conv[:, 0] - self.K0 @ U_conv[:, 0]
        )

        Meth, P = self.ask_method(Meth)

        if Meth == "CDM":
            U_conv[:, -1] = U_conv[:, 0] - dt * V_conv[:, 0] + dt ** 2 * A_conv[:, 0] / 2

            K_h = self.M / dt ** 2 + self.C / (2 * dt)
            a = self.M / dt ** 2 - self.C / (2 * dt)
            b = self.K0 - 2 * self.M / dt ** 2

            a_ff = a[np.ix_(self.dof_free, self.dof_free)]
            a_fd = a[np.ix_(self.dof_free, self.dof_moving)]
            a_df = a[np.ix_(self.dof_moving, self.dof_free)]
            a_dd = a[np.ix_(self.dof_moving, self.dof_moving)]

            b_ff = b[np.ix_(self.dof_free, self.dof_free)]
            b_fd = b[np.ix_(self.dof_free, self.dof_moving)]
            b_df = b[np.ix_(self.dof_moving, self.dof_free)]
            b_dd = b[np.ix_(self.dof_moving, self.dof_moving)]

            k_ff = K_h[np.ix_(self.dof_free, self.dof_free)]
            k_fd = K_h[np.ix_(self.dof_free, self.dof_moving)]
            k_df = K_h[np.ix_(self.dof_moving, self.dof_free)]
            k_dd = K_h[np.ix_(self.dof_moving, self.dof_moving)]

            for i in np.arange(1, nb_steps):
                P_h_f = (
                        loading[i - 1] * self.P[self.dof_free]
                        - a_ff @ U_conv[self.dof_free, i - 2]
                        - a_fd @ U_conv[self.dof_moving, i - 2]
                        - b_ff @ U_conv[self.dof_free, i - 1]
                        - b_fd @ U_conv[self.dof_moving, i - 1]
                )

                U_d = np.zeros(len(self.disp_histories))

                for j, disp in enumerate(self.disp_histories):
                    U_d[j] = disp[i]

                U_conv[self.dof_free, i] = np.linalg.solve(k_ff, P_h_f - k_fd @ U_d)
                U_conv[self.dof_moving, i] = U_d

                P_h_d = (
                        k_df @ U_conv[self.dof_free, i]
                        + k_dd @ U_d
                        + a_df @ U_conv[self.dof_free, i - 2]
                        + a_dd @ U_conv[self.dof_moving, i - 2]
                        + b_df @ U_conv[self.dof_free, i - 1]
                        + b_dd @ U_conv[self.dof_moving, i - 1]
                )

                V_conv[self.dof_free, i] = (
                                                   U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]
                                           ) / (2 * dt)
                V_conv[self.dof_moving, i] = (
                                                     U_conv[self.dof_moving, i] - U_conv[self.dof_moving, i - 1]
                                             ) / (2 * dt)

                A_conv[self.dof_free, i] = (
                                                   U_conv[self.dof_free, i]
                                                   - 2 * U_conv[self.dof_free, i - 1]
                                                   + U_conv[self.dof_free, i - 2]
                                           ) / (dt ** 2)
                A_conv[self.dof_moving, i] = (
                                                     U_conv[self.dof_moving, i]
                                                     - 2 * U_conv[self.dof_moving, i - 1]
                                                     + U_conv[self.dof_moving, i - 2]
                                             ) / (dt ** 2)

                P_conv[self.dof_free, i] = P_h_f.copy()
                P_conv[self.dof_moving, i] = P_h_d.copy()

        elif Meth == "NWK":
            A1 = self.M / (P["b"] * dt ** 2) + P["g"] * self.C / (P["b"] * dt)
            A2 = self.M / (P["b"] * dt) + (P["g"] / P["b"] - 1) * self.C
            A3 = (1 / (2 * P["b"]) - 1) * self.M + dt * (
                    P["g"] / (2 * P["b"]) - 1
            ) * self.C

            a1_ff = A1[np.ix_(self.dof_free, self.dof_free)]
            a1_fd = A1[np.ix_(self.dof_free, self.dof_moving)]

            a2_ff = A2[np.ix_(self.dof_free, self.dof_free)]
            a2_fd = A2[np.ix_(self.dof_free, self.dof_moving)]

            a3_ff = A3[np.ix_(self.dof_free, self.dof_free)]
            a3_fd = A3[np.ix_(self.dof_free, self.dof_moving)]

            K_h = self.K0 + A1

            k_ff = K_h[np.ix_(self.dof_free, self.dof_free)]
            k_fd = K_h[np.ix_(self.dof_free, self.dof_moving)]
            k_df = K_h[np.ix_(self.dof_moving, self.dof_free)]
            k_dd = K_h[np.ix_(self.dof_moving, self.dof_moving)]

            for i in np.arange(1, nb_steps):
                P_h_f = (
                        loading[i] * self.P[self.dof_free]
                        + self.P_fixed[self.dof_free]
                        + a1_ff @ U_conv[self.dof_free, i - 1]
                        + a2_ff @ V_conv[self.dof_free, i - 1]
                        + a3_ff @ A_conv[self.dof_free, i - 1]
                        + a1_fd @ U_conv[self.dof_moving, i - 1]
                        + a2_fd @ V_conv[self.dof_moving, i - 1]
                        + a3_fd @ A_conv[self.dof_moving, i - 1]
                )

                for j, disp in enumerate(self.disp_histories):
                    U_conv[self.dof_moving[j], i] = disp[i]
                    V_conv[self.dof_moving[j], i] = (
                                                            U_conv[self.dof_moving[j], i]
                                                            - U_conv[self.dof_moving[j], i - 1]
                                                    ) / dt
                    A_conv[self.dof_moving[j], i] = (
                                                            V_conv[self.dof_moving[j], i]
                                                            - V_conv[self.dof_moving[j], i - 1]
                                                    ) / dt

                U_conv[self.dof_free, i] = sc.linalg.solve(
                    k_ff, P_h_f - k_fd @ U_conv[self.dof_moving, i]
                )

                V_conv[self.dof_free, i] = (
                        (P["g"] / (P["b"] * dt))
                        * (U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1])
                        + (1 - P["g"] / P["b"]) * V_conv[self.dof_free, i - 1]
                        + dt * (1 - P["g"] / (2 * P["b"])) * A_conv[self.dof_free, i - 1]
                )
                A_conv[self.dof_free, i] = (
                        (1 / (P["b"] * dt ** 2))
                        * (U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1])
                        - V_conv[self.dof_free, i - 1] / (P["b"] * dt)
                        - (1 / (2 * P["b"]) - 1) * A_conv[self.dof_free, i - 1]
                )

                P_conv[self.dof_free, i] = P_h_f.copy()
                P_conv[self.dof_moving, i] = (
                        k_df @ U_conv[self.dof_free, i] + k_dd @ U_conv[self.dof_moving, i]
                )

        elif Meth == "WIL":
            A1 = 6 / (P["t"] * dt) * self.M + 3 * self.C
            A2 = 3 * self.M + P["t"] * dt / 2 * self.C

            K_h = self.K0 + 6 / (P["t"] * dt) ** 2 * self.M + 3 / (P["t"] * dt) * self.C

            loading = np.append(loading, loading[-1])

            for i in np.arange(1, nb_steps):
                dp_h = (
                               (P["t"] - 1) * (loading[i + 1] - loading[i])
                               + loading[i]
                               - loading[i - 1]
                       ) * self.P

                dp_h += A1 @ V_conv[:, i - 1] + A2 @ A_conv[:, i - 1]

                d_Uh = sc.linalg.solve(K_h, dp_h)

                d_A = (
                              6 / (P["t"] * dt) ** 2 * d_Uh
                              - 6 / (P["t"] * dt) * V_conv[:, i - 1]
                              - 3 * A_conv[:, i - 1]
                      ) / (P["t"])

                d_V = dt * A_conv[:, i - 1] + dt / 2 * d_A
                d_U = (
                        dt * V_conv[:, i - 1]
                        + (dt ** 2) / 2 * A_conv[:, i - 1]
                        + (dt ** 2) / 6 * d_A
                )

                U_conv[self.dof_free, i] = (U_conv[:, i - 1] + d_U)[self.dof_free]
                V_conv[self.dof_free, i] = (V_conv[:, i - 1] + d_V)[self.dof_free]
                A_conv[self.dof_free, i] = (A_conv[:, i - 1] + d_A)[self.dof_free]

        elif Meth == "GEN":
            am = 0
            b = P["b"]
            g = P["g"]
            af = P["af"]

            A1 = (1 - am) / (b * dt ** 2) * self.M + g * (1 - af) / (b * dt) * self.C
            A2 = (1 - am) / (b * dt) * self.M + (g * (1 - af) / b - 1) * self.C
            A3 = ((1 - am) / (2 * b) - 1) * self.M + dt * (1 - af) * (
                    g / (2 * b) - 1
            ) * self.C

            a1_ff = A1[np.ix_(self.dof_free, self.dof_free)]
            a1_fd = A1[np.ix_(self.dof_free, self.dof_moving)]

            a2_ff = A2[np.ix_(self.dof_free, self.dof_free)]
            a2_fd = A2[np.ix_(self.dof_free, self.dof_moving)]

            a3_ff = A3[np.ix_(self.dof_free, self.dof_free)]
            a3_fd = A3[np.ix_(self.dof_free, self.dof_moving)]

            K_h = self.K0 * (1 - af) + A1

            k_ff = K_h[np.ix_(self.dof_free, self.dof_free)]
            k_fd = K_h[np.ix_(self.dof_free, self.dof_moving)]
            k_df = K_h[np.ix_(self.dof_moving, self.dof_free)]
            k_dd = K_h[np.ix_(self.dof_moving, self.dof_moving)]

            for i in np.arange(1, nb_steps):
                P_h_f = (
                        loading[i] * self.P[self.dof_free]
                        + self.P_fixed[self.dof_free]
                        + a1_ff @ U_conv[self.dof_free, i - 1]
                        + a2_ff @ V_conv[self.dof_free, i - 1]
                        + a3_ff @ A_conv[self.dof_free, i - 1]
                        + a1_fd @ U_conv[self.dof_moving, i - 1]
                        + a2_fd @ V_conv[self.dof_moving, i - 1]
                        + a3_fd @ A_conv[self.dof_moving, i - 1]
                        - af
                        * (
                                self.K0[np.ix_(self.dof_free, self.dof_free)]
                                @ U_conv[self.dof_free, i - 1]
                                + self.K0[np.ix_(self.dof_free, self.dof_moving)]
                                @ U_conv[self.dof_moving, i - 1]
                        )
                )

                for j, disp in enumerate(self.disp_histories):
                    U_conv[self.dof_moving[j], i] = disp[i]
                    # V_conv[self.dof_moving[j],i] = (U_conv[self.dof_moving[j],i] - U_conv[self.dof_moving[j],i-1]) / dt
                    # A_conv[self.dof_moving[j],i] = (V_conv[self.dof_moving[j],i] - V_conv[self.dof_moving[j],i-1]) / dt

                U_conv[self.dof_free, i] = sc.linalg.solve(
                    k_ff, P_h_f - k_fd @ U_conv[self.dof_moving, i]
                )

                V_conv[:, i][self.dof_free] = (
                        P["g"] / (P["b"] * dt) * (U_conv[:, i] - U_conv[:, i - 1])
                        + (1 - P["g"] / P["b"]) * V_conv[:, i - 1]
                        + dt * (1 - P["g"] / (2 * P["b"])) * A_conv[:, i - 1]
                )[self.dof_free]
                A_conv[:, i][self.dof_free] = (
                        1 / (P["b"] * dt ** 2) * (U_conv[:, i] - U_conv[:, i - 1])
                        - 1 / (dt * P["b"]) * V_conv[:, i - 1]
                        - (1 / (2 * P["b"]) - 1) * A_conv[:, i - 1]
                )[self.dof_free]
                P_conv[self.dof_free, i] = P_h_f.copy()
                P_conv[self.dof_moving, i] = (
                        k_df @ U_conv[self.dof_free, i] + k_dd @ U_conv[self.dof_moving, i]
                )

        elif Meth is None:
            print("Method does not exist")

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(
            f"Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file"
        )

        Params = []
        for key, value in P.items():
            Params.append(f"{key}={np.around(value, 2)}")

        Params = "_".join(Params)

        filename = filename + "_" + Meth + "_" + Params + ".h5"
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("U_conv", data=U_conv)
            hf.create_dataset("V_conv", data=V_conv)
            hf.create_dataset("A_conv", data=A_conv)
            hf.create_dataset("P_ref", data=self.P)
            hf.create_dataset("P_conv", data=P_conv)
            hf.create_dataset("Load_Multiplier", data=loading)
            hf.create_dataset("Time", data=Time)
            hf.create_dataset("Last_conv", data=nb_steps - 1)

            hf.attrs["Descr"] = "Results of the" + Meth + "simulation"
            hf.attrs["Method"] = Meth

    def solve_dyn_nonlinear(
            self, T, dt, U0=None, V0=None, lmbda=None, Meth=None, filename="", dir_name=""
    ):
        time_start = time.time()

        if U0 is None:
            if np.linalg.norm(self.U) == 0:
                U0 = np.zeros(self.nb_dofs)
            else:
                U0 = deepcopy(self.U)

        if V0 is None:
            V0 = np.zeros(self.nb_dofs)

        Time = np.arange(0, T, dt, dtype=float)
        Time = np.append(Time, T)
        nb_steps = len(Time)

        loading = np.zeros(nb_steps)

        if callable(lmbda):
            for i, t in enumerate(Time):
                loading[i] = lmbda(t)
        elif isinstance(lmbda, list):
            loading = lmbda
            if len(loading) > nb_steps:
                print("Truncate")
                loading = loading[:nb_steps]
            elif len(loading) < nb_steps:
                print("Add 0")
                missing = nb_steps - len(loading)
                for i in range(missing):
                    loading.append(0)

        self.get_P_r()
        self.get_K_str0()
        self.get_M_str()
        self.get_C_str()

        if hasattr(self, "times"):
            for timestep in self.times:
                if timestep != dt:
                    warnings.warn(
                        "Unmatching timesteps between excitation and simulation"
                    )
            for i, disp in enumerate(self.disp_histories):
                if U0[self.dof_moving[i]] != disp[0]:
                    warnings.warn("Unmatching initial displacements")

        else:
            self.dof_moving = []
            self.disp_histories = []
            self.times = []

        U_conv = np.zeros((self.nb_dofs, nb_steps), dtype=float)
        V_conv = np.zeros((self.nb_dofs, nb_steps), dtype=float)
        A_conv = np.zeros((self.nb_dofs, nb_steps), dtype=float)
        F_conv = np.zeros((self.nb_dofs, nb_steps), dtype=float)

        U_conv[:, 0] = deepcopy(U0)
        V_conv[:, 0] = deepcopy(V0)
        F_conv[:, 0] = deepcopy(self.P_r)

        self.commit()

        last_sec = 0

        Meth, P = self.ask_method(Meth)

        if Meth == "CDM":
            self.U = U_conv[:, 0].copy()
            self.get_P_r()
            F_conv[:, 0] = self.P_r.copy()

            A_conv[:, 0] = sc.linalg.solve(
                self.M,
                loading[0] * self.P
                + self.P_fixed
                - self.C @ V_conv[:, 0]
                - F_conv[:, 0],
            )

            U_conv[:, -1] = U_conv[:, 0] - dt * V_conv[:, 0] + dt ** 2 / 2 * A_conv[:, 0]

            K_h = 1 / (dt ** 2) * self.M + 1 / (2 * dt) * self.C
            A = 1 / (dt ** 2) * self.M - 1 / (2 * dt) * self.C
            B = -2 / (dt ** 2) * self.M

            a_ff = A[np.ix_(self.dof_free, self.dof_free)]
            a_fd = A[np.ix_(self.dof_free, self.dof_moving)]
            a_df = A[np.ix_(self.dof_moving, self.dof_free)]
            a_dd = A[np.ix_(self.dof_moving, self.dof_moving)]

            b_ff = B[np.ix_(self.dof_free, self.dof_free)]
            b_fd = B[np.ix_(self.dof_free, self.dof_moving)]
            b_df = B[np.ix_(self.dof_moving, self.dof_free)]
            b_dd = B[np.ix_(self.dof_moving, self.dof_moving)]

            k_ff = K_h[np.ix_(self.dof_free, self.dof_free)]
            k_fd = K_h[np.ix_(self.dof_free, self.dof_moving)]
            k_df = K_h[np.ix_(self.dof_moving, self.dof_free)]
            k_dd = K_h[np.ix_(self.dof_moving, self.dof_moving)]

            for i in np.arange(1, nb_steps):
                if self.stiff_type[:3] == "TAN":
                    self.get_C_str()

                    K_h = 1 / (dt ** 2) * self.M + 1 / (2 * dt) * self.C
                    A = 1 / (dt ** 2) * self.M - 1 / (2 * dt) * self.C

                    a_ff = A[np.ix_(self.dof_free, self.dof_free)]
                    a_fd = A[np.ix_(self.dof_free, self.dof_moving)]
                    a_df = A[np.ix_(self.dof_moving, self.dof_free)]
                    a_dd = A[np.ix_(self.dof_moving, self.dof_moving)]

                    k_ff = K_h[np.ix_(self.dof_free, self.dof_free)]
                    k_fd = K_h[np.ix_(self.dof_free, self.dof_moving)]
                    k_df = K_h[np.ix_(self.dof_moving, self.dof_free)]
                    k_dd = K_h[np.ix_(self.dof_moving, self.dof_moving)]

                self.U = U_conv[:, i - 1].copy()
                try:
                    self.get_P_r()
                except Exception as e:
                    print(e)
                    break

                F_conv[:, i - 1] = deepcopy(self.P_r)

                P_h_f = (
                        loading[i] * self.P[self.dof_free]
                        + self.P_fixed[self.dof_free]
                        - a_ff @ U_conv[self.dof_free, i - 2]
                        - a_fd @ U_conv[self.dof_moving, i - 2]
                        - b_ff @ U_conv[self.dof_free, i - 1]
                        - b_fd @ U_conv[self.dof_moving, i - 1]
                        - F_conv[self.dof_free, i - 1]
                )

                U_d = np.zeros(len(self.disp_histories))

                for j, disp in enumerate(self.disp_histories):
                    U_d[j] = disp[i]

                U_conv[self.dof_free, i] = np.linalg.solve(k_ff, P_h_f - k_fd @ U_d)
                U_conv[self.dof_moving, i] = U_d

                P_h_d = (
                        k_df @ U_conv[self.dof_free, i]
                        + k_dd @ U_d
                        + a_df @ U_conv[self.dof_free, i - 2]
                        + a_dd @ U_conv[self.dof_moving, i - 2]
                        + b_df @ U_conv[self.dof_free, i - 1]
                        + b_dd @ U_conv[self.dof_moving, i - 1]
                )

                V_conv[self.dof_free, i] = (
                                                   U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]
                                           ) / (2 * dt)
                V_conv[self.dof_moving, i] = (
                                                     U_conv[self.dof_moving, i] - U_conv[self.dof_moving, i - 1]
                                             ) / (2 * dt)

                A_conv[self.dof_free, i] = (
                                                   U_conv[self.dof_free, i]
                                                   - 2 * U_conv[self.dof_free, i - 1]
                                                   + U_conv[self.dof_free, i - 2]
                                           ) / (dt ** 2)
                A_conv[self.dof_moving, i] = (
                                                     U_conv[self.dof_moving, i]
                                                     - 2 * U_conv[self.dof_moving, i - 1]
                                                     + U_conv[self.dof_moving, i - 2]
                                             ) / (dt ** 2)

                if i * dt >= last_sec:
                    print(
                        f"reached {np.around(last_sec, 3)} seconds out of {int(Time[-1])} seconds"
                    )
                    last_sec += 0.1

                last_conv = i

                self.commit()

        elif Meth == "NWK":
            tol = 1
            singular_steps = []
            # tol = np.max(self.M) / np.max(self.K) * 10
            print(f"Tolerance is {tol}")

            self.U = deepcopy(U_conv[:, 0])

            g = P["g"]
            b = P["b"]

            print(g)
            print(b)
            A_conv[:, 0] = sc.linalg.solve(
                self.M, loading[0] * self.P + self.P_fixed - self.C @ V0 - F_conv[:, 0]
            )

            A1 = (1 / (b * dt ** 2)) * self.M + (g / (b * dt)) * self.C
            A2 = (1 / (b * dt)) * self.M + (g / b - 1) * self.C
            A3 = (1 / (2 * b) - 1) * self.M + dt * (g / (2 * b) - 1) * self.C

            no_conv = 0

            a1 = 1 / (b * dt ** 2)
            a2 = 1 / (b * dt)
            a3 = 1 / (2 * b) - 1

            a4 = g / (b * dt)
            a5 = 1 - g / b
            a6 = dt * (1 - g / (2 * b))

            for i in np.arange(1, nb_steps):
                self.U = U_conv[:, i - 1].copy()

                try:
                    self.get_P_r()
                except Exception as e:
                    print(e)
                    break

                for j, disp in enumerate(self.disp_histories):
                    self.U[self.dof_moving[j]] = disp[i]
                    U_conv[self.dof_moving[j], i] = disp[i]
                    V_conv[self.dof_moving[j], i] = (
                            a4
                            * (
                                    U_conv[self.dof_moving[j], i]
                                    - U_conv[self.dof_moving[j], i - 1]
                            )
                            + a5 * V_conv[self.dof_moving[j], i - 1]
                            + a6 * A_conv[self.dof_moving[j], i - 1]
                    )
                    A_conv[self.dof_moving[j], i] = (
                            a1
                            * (
                                    U_conv[self.dof_moving[j], i]
                                    - U_conv[self.dof_moving[j], i - 1]
                            )
                            - a2 * V_conv[self.dof_moving[j], i - 1]
                            - a3 * A_conv[self.dof_moving[j], i - 1]
                    )

                P_h_f = (
                        loading[i] * self.P[self.dof_free]
                        + self.P_fixed[self.dof_free]
                        + A1[np.ix_(self.dof_free, self.dof_free)]
                        @ U_conv[self.dof_free, i - 1]
                        + A1[np.ix_(self.dof_free, self.dof_moving)]
                        @ U_conv[self.dof_moving, i - 1]
                        + A2[np.ix_(self.dof_free, self.dof_free)]
                        @ V_conv[self.dof_free, i - 1]
                        + A2[np.ix_(self.dof_free, self.dof_moving)]
                        @ V_conv[self.dof_moving, i - 1]
                        + A3[np.ix_(self.dof_free, self.dof_free)]
                        @ A_conv[self.dof_free, i - 1]
                        + A3[np.ix_(self.dof_free, self.dof_moving)]
                        @ A_conv[self.dof_moving, i - 1]
                )
                counter = 0
                conv = False

                while not conv:
                    # self.revert_commit()

                    try:
                        self.get_P_r()
                    except Exception as e:
                        print(e)
                        break

                    self.get_K_str()

                    counter += 1
                    if counter > 100:
                        no_conv = i
                        break

                    R = (
                            P_h_f
                            - self.P_r[self.dof_free]
                            - A1[np.ix_(self.dof_free, self.dof_free)]
                            @ self.U[self.dof_free]
                            - A1[np.ix_(self.dof_free, self.dof_moving)]
                            @ self.U[self.dof_moving]
                    )
                    if np.linalg.norm(R) < tol:
                        self.commit()
                        U_conv[:, i] = deepcopy(self.U)
                        F_conv[:, i] = deepcopy(self.P_r)
                        conv = True
                        # print(counter)
                        last_conv = i

                    Kt_p = self.K + A1

                    dU = np.linalg.solve(Kt_p[np.ix_(self.dof_free, self.dof_free)], R)
                    self.U[self.dof_free] += dU
                    # self.U[self.dof_moving] += dU_d

                if no_conv > 0:
                    print(f"Step {no_conv} did not converge")
                    break

                dU_step = U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]
                V_conv[self.dof_free, i] = (
                        a4 * dU_step
                        + a5 * V_conv[self.dof_free, i - 1]
                        + a6 * A_conv[self.dof_free, i - 1]
                )
                A_conv[self.dof_free, i] = (
                        a1 * dU_step
                        - a2 * V_conv[self.dof_free, i - 1]
                        - a3 * A_conv[self.dof_free, i - 1]
                )

                if self.stiff_type[:3] == "TAN":
                    self.get_C_str()
                    A1 = (1 / (b * dt ** 2)) * self.M + (g / (b * dt)) * self.C
                    A2 = (1 / (b * dt)) * self.M + (g / b - 1) * self.C
                    A3 = (1 / (2 * b) - 1) * self.M + dt * (g / (2 * b) - 1) * self.C

                # if np.min(np.real(np.sqrt(eigvals))) <=  1:
                #     singular_steps.append(i)
                #     print(f'Step{i} is singular - {np.linalg.cond(self.C[np.ix_(self.dof_free, self.dof_free)])}')
                #     # print(f'Step{i} is singular - {np.linalg.cond(self.K[np.ix_(self.dof_free, self.dof_free)])}')

                if i * dt >= last_sec:
                    print(
                        f"reached {np.around(last_sec, 3)} seconds out of {int(Time[-1])} seconds"
                    )
                    self.plot_structure(
                        scale=1,
                        plot_forces=False,
                        plot_cf=False,
                        plot_supp=False,
                        lims=[[-6.0, 6.0], [-1.2, 6.5]],
                    )
                    last_sec += 0.1

        elif Meth == "WIL":
            pass

        elif Meth == "GEN":
            tol = 1e-3
            singular_steps = []
            # tol = np.max(self.M) / np.max(self.K) * 10
            print(f"Tolerance is {tol}")

            self.U = deepcopy(U_conv[:, 0])

            g = P["g"]
            b = P["b"]
            af = P["af"]
            am = P["am"]

            A_conv[:, 0] = sc.linalg.solve(
                self.M, loading[0] * self.P + self.P_fixed - self.C @ V0 - F_conv[:, 0]
            )

            A1 = ((1 - am) / (b * dt ** 2)) * self.M + (g * (1 - af) / (b * dt)) * self.C
            A2 = ((1 - am) / (b * dt)) * self.M + (g * (1 - af) / b - 1) * self.C
            A3 = ((1 - am) / (2 * b) - 1) * self.M + dt * (1 - af) * (
                    g / (2 * b) - 1
            ) * self.C

            no_conv = 0

            a1 = 1 / (b * dt ** 2)
            a2 = 1 / (b * dt)
            a3 = 1 / (2 * b) - 1

            a4 = g / (b * dt)
            a5 = 1 - g / b
            a6 = dt * (1 - g / (2 * b))

            for i in np.arange(1, nb_steps):
                self.U = U_conv[:, i - 1].copy()

                try:
                    self.get_P_r()
                except Exception as e:
                    print(e)
                    break

                for j, disp in enumerate(self.disp_histories):
                    self.U[self.dof_moving[j]] = disp[i]
                    U_conv[self.dof_moving[j], i] = disp[i]
                    # V_conv[self.dof_moving[j],i] = a4 * (U_conv[self.dof_moving[j],i] - U_conv[self.dof_moving[j],i-1]) + a5*V_conv[self.dof_moving[j],i-1] + a6 * A_conv[self.dof_moving[j],i-1]
                    # A_conv[self.dof_moving[j],i] = a1 * (U_conv[self.dof_moving[j],i] - U_conv[self.dof_moving[j],i-1]) - a2*V_conv[self.dof_moving[j],i-1] - a3 * A_conv[self.dof_moving[j],i-1]

                P_h_f = (
                        loading[i] * self.P[self.dof_free]
                        + self.P_fixed[self.dof_free]
                        + A1[np.ix_(self.dof_free, self.dof_free)]
                        @ U_conv[self.dof_free, i - 1]
                        + A1[np.ix_(self.dof_free, self.dof_moving)]
                        @ U_conv[self.dof_moving, i - 1]
                        + A2[np.ix_(self.dof_free, self.dof_free)]
                        @ V_conv[self.dof_free, i - 1]
                        + A2[np.ix_(self.dof_free, self.dof_moving)]
                        @ V_conv[self.dof_moving, i - 1]
                        + A3[np.ix_(self.dof_free, self.dof_free)]
                        @ A_conv[self.dof_free, i - 1]
                        + A3[np.ix_(self.dof_free, self.dof_moving)]
                        @ A_conv[self.dof_moving, i - 1]
                        - af * F_conv[self.dof_free, i - 1]
                )

                counter = 0
                conv = False

                while not conv:
                    # self.revert_commit()

                    try:
                        self.get_P_r()
                    except Exception as e:
                        print(e)
                        break

                    self.get_K_str()

                    counter += 1
                    if counter > 100:
                        no_conv = i
                        break

                    R = (
                            P_h_f
                            - self.P_r[self.dof_free]
                            - A1[np.ix_(self.dof_free, self.dof_free)]
                            @ self.U[self.dof_free]
                            - A1[np.ix_(self.dof_free, self.dof_moving)]
                            @ self.U[self.dof_moving]
                    )
                    if np.linalg.norm(R) < tol:
                        self.commit()
                        U_conv[:, i] = deepcopy(self.U)
                        F_conv[:, i] = deepcopy(self.P_r)
                        conv = True
                        # print(counter)
                        last_conv = i

                    Kt_p = self.K + A1

                    dU = np.linalg.solve(Kt_p[np.ix_(self.dof_free, self.dof_free)], R)
                    self.U[self.dof_free] += dU
                    # self.U[self.dof_moving] += dU_d

                if no_conv > 0:
                    print(f"Step {no_conv} did not converge")
                    break

                dU_step = U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]
                V_conv[self.dof_free, i] = (
                        a4 * dU_step
                        + a5 * V_conv[self.dof_free, i - 1]
                        + a6 * A_conv[self.dof_free, i - 1]
                )
                A_conv[self.dof_free, i] = (
                        a1 * dU_step
                        - a2 * V_conv[self.dof_free, i - 1]
                        - a3 * A_conv[self.dof_free, i - 1]
                )

                if self.stiff_type[:3] == "TAN":
                    self.get_C_str()
                    A1 = ((1 - am) / (b * dt ** 2)) * self.M + (
                            g * (1 - af) / (b * dt)
                    ) * self.C
                    A2 = ((1 - am) / (b * dt)) * self.M + (
                            g * (1 - af) / b - 1
                    ) * self.C
                    A3 = ((1 - am) / (2 * b) - 1) * self.M + dt * (1 - af) * (
                            g / (2 * b) - 1
                    ) * self.C

                # if np.min(np.real(np.sqrt(eigvals))) <=  1:
                #     singular_steps.append(i)
                #     print(f'Step{i} is singular - {np.linalg.cond(self.C[np.ix_(self.dof_free, self.dof_free)])}')
                #     # print(f'Step{i} is singular - {np.linalg.cond(self.K[np.ix_(self.dof_free, self.dof_free)])}')

                if i * dt >= last_sec:
                    print(
                        f"reached {np.around(last_sec, 3)} seconds out of {int(Time[-1])} seconds"
                    )
                    self.plot_structure(
                        scale=1,
                        plot_forces=False,
                        plot_cf=False,
                        plot_supp=False,
                        lims=[[-6.0, 6.0], [-1.2, 6.5]],
                    )
                    last_sec += 0.1

        elif Meth is None:
            print("Method does not exist")

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(
            f"Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file"
        )

        Params = []
        for key, value in P.items():
            Params.append(f"{key}={np.around(value, 2)}")

        Params = "_".join(Params)

        filename = filename + "_" + Meth + "_" + Params + ".h5"
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("U_conv", data=U_conv)
            hf.create_dataset("V_conv", data=V_conv)
            hf.create_dataset("A_conv", data=A_conv)
            hf.create_dataset("F_conv", data=F_conv)
            hf.create_dataset("P_ref", data=self.P)
            hf.create_dataset("Load_Multiplier", data=loading)
            hf.create_dataset("Time", data=Time)
            hf.create_dataset("Last_conv", data=last_conv)
            # hf.create_dataset('Singular_steps', data=singular_steps)

            hf.attrs["Descr"] = "Results of the" + Meth + "simulation"
            hf.attrs["Method"] = Meth

    def solve_modal(
            self,
            modes=None,
            no_inertia=False,
            filename="Results_Modal",
            dir_name="",
            save=True,
            initial=False,
    ):
        time_start = time.time()

        self.get_P_r()
        self.get_M_str(no_inertia=no_inertia)

        if not initial:
            if not hasattr(self, "K"):
                self.get_K_str()

            if modes is None:
                # print('HEllo')
                # self.K = np.around(self.K,6)
                # self.M = np.around(self.M,8)
                omega, phi = sc.linalg.eig(
                    self.K[np.ix_(self.dof_free, self.dof_free)],
                    self.M[np.ix_(self.dof_free, self.dof_free)],
                )

            elif isinstance(modes, int):
                if np.linalg.det(self.M) == 0:
                    warnings.warn(
                        "Might need to use linalg.eig if the matrix M is non-invertible"
                    )
                omega, phi = sc.sparse.linalg.eigsh(
                    self.K[np.ix_(self.dof_free, self.dof_free)],
                    modes,
                    self.M[np.ix_(self.dof_free, self.dof_free)],
                    which="SM",
                )

            else:
                warnings.warn("Required modes should be either int or None")
        else:
            self.get_K_str0()
            if modes is None:
                omega, phi = sc.linalg.eigh(
                    self.K0[np.ix_(self.dof_free, self.dof_free)],
                    self.M[np.ix_(self.dof_free, self.dof_free)],
                )
            elif isinstance(modes, int):
                if np.linalg.det(self.M) == 0:
                    warnings.warn(
                        "Might need to use linalg.eig if the matrix M is non-invertible"
                    )
                omega, phi = sc.sparse.linalg.eigsh(
                    self.K0[np.ix_(self.dof_free, self.dof_free)],
                    modes,
                    self.M[np.ix_(self.dof_free, self.dof_free)],
                    which="SM",
                )

            else:
                warnings.warn("Required modes should be either int or None")
        # print(omega)
        # for i in range(len(omega)):
        #     if omega[i] < 0: omega[i] = 0
        self.eig_vals = np.sort(np.real(np.sqrt(omega))).copy()
        self.eig_modes = (np.real(phi).T)[np.argsort((np.sqrt(omega)))].T.copy()
        # print(self.eig_vals)

        if save:
            time_end = time.time()
            total_time = time_end - time_start
            print("Simulation done... writing results to file")

            filename = filename + ".h5"
            file_path = os.path.join(dir_name, filename)

            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("eig_vals", data=self.eig_vals)
                hf.create_dataset("eig_modes", data=self.eig_modes)

                hf.attrs["Simulation_Time"] = total_time

    def plot_stiffness(self, save=None):
        E = []
        vertices = []

        for j, CF in enumerate(self.list_cfs):
            for i, CP in enumerate(CF.cps):
                E.append(np.around(CP.sp1.law.stiff["E"], 3))
                E.append(np.around(CP.sp2.law.stiff["E"], 3))
                vertices.append(CP.vertices_fibA)
                vertices.append(CP.vertices_fibB)

        from matplotlib.colors import Normalize
        from matplotlib import cm

        def normalize(smax, smin):
            if (smax - smin) == 0 and smax < 0:
                return Normalize(
                    vmin=1.1 * smin / 1e9, vmax=0.9 * smax / 1e9, clip=False
                )
            elif (smax - smin) == 0 and smax == 0:
                return Normalize(vmin=-1e-6, vmax=1e-6, clip=False)
            elif (smax - smin) == 0:
                return Normalize(
                    vmin=0.9 * smin / 1e9, vmax=1.1 * smax / 1e9, clip=False
                )
            else:
                return Normalize(vmin=smin / 1e9, vmax=smax / 1e9, clip=False)

        def plot(stiff, vertex):
            smax = np.max(stiff)
            smin = np.min(stiff)

            plt.axis("equal")
            plt.axis("off")
            plt.title("Axial stiffness [GPa]")

            norm = normalize(smax, smin)
            cmap = cm.get_cmap("coolwarm", 200)

            for i in range(len(stiff)):
                if smax - smin == 0:
                    index = norm(np.around(stiff[i], 6) / 1e9)
                else:
                    index = norm(np.around(stiff[i], 6) / 1e9)
                color = cmap(index)
                vertices_x = np.append(vertex[i][:, 0], vertex[i][0, 0])
                vertices_y = np.append(vertex[i][:, 1], vertex[i][0, 1])
                plt.fill(vertices_x, vertices_y, color=color)

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(plt.gca())

            cax = divider.append_axes("right", size="10%", pad=0.2)
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

        plt.figure()

        plot(E, vertices)

        if save is not None:
            plt.savefig(save)

    def get_stresses(self, angle=None, tag=None):
        # Compute maximal stress and minimal stress:

        eps = np.array([])
        sigma = np.array([])
        x_s = np.array([])

        for j, CF in enumerate(self.list_cfs):
            if (angle is None) or (abs(CF.angle - angle) < 1e-6):
                for i, CP in enumerate(CF.cps):
                    # print(CF.bl_B.disps[0])
                    # print

                    if not CP.to_ommit():
                        if tag is None or CP.sp1.law.tag == tag:
                            eps = np.append(eps, np.around(CP.sp1.law.strain["e"], 12))
                            # print(np.around(CP.sp1.law.strain['e'],12))
                            sigma = np.append(
                                sigma, np.around(CP.sp1.law.stress["s"], 12)
                            )
                            x_s = np.append(x_s, CP.x_cp[0])
        return sigma, eps, x_s

    def plot_stresses(self, angle=None, save=None, tag=None):
        # Compute maximal stress and minimal stress:

        tau = []
        sigma = []
        vertices = []

        for j, CF in enumerate(self.list_cfs):
            if (angle is None) or (abs(CF.angle - angle) < 1e-6):
                for i, CP in enumerate(CF.cps):
                    if not CP.to_ommit():
                        if tag is None or CP.sp1.law.tag == tag:
                            tau.append(np.around(CP.sp1.law.stress["t"], 12))
                            tau.append(np.around(CP.sp2.law.stress["t"], 12))
                            sigma.append(np.around(CP.sp1.law.stress["s"], 12))
                            sigma.append(np.around(CP.sp2.law.stress["s"], 12))
                            vertices.append(CP.vertices_fibA)
                            vertices.append(CP.vertices_fibB)

        from matplotlib.colors import Normalize
        from matplotlib import cm

        def normalize(smax, smin):
            if (smax - smin) == 0 and smax < 0:
                return Normalize(
                    vmin=1.1 * smin / 1e6, vmax=0.9 * smax / 1e6, clip=False
                )
            elif (smax - smin) == 0 and smax == 0:
                return Normalize(vmin=-1e-6, vmax=1e-6, clip=False)
            elif (smax - smin) == 0:
                return Normalize(
                    vmin=0.9 * smin / 1e6, vmax=1.1 * smax / 1e6, clip=False
                )
            else:
                return Normalize(vmin=smin / 1e6, vmax=smax / 1e6, clip=False)

        def plot(stress, vertex, name_stress=None):
            smax = np.max(stress)
            smin = np.min(stress)

            print(
                f"Maximal {'axial' if name_stress == 'sigma' else 'shear'} stress is {np.around(smax / 1e6, 3)} MPa"
            )
            print(
                f"Minimum {'axial' if name_stress == 'sigma' else 'shear'} stress is {np.around(smin / 1e6, 3)} MPa"
            )
            # Plot sigmas

            plt.axis("equal")
            plt.axis("off")
            plt.title(
                f"{'Axial' if name_stress == 'sigma' else 'Shear'} stresses [MPa]"
            )

            norm = normalize(smax, smin)
            cmap = cm.get_cmap("viridis", 200)

            for i in range(len(sigma)):
                if smax - smin == 0:
                    index = norm(np.around(stress[i], 6) / 1e6)
                else:
                    index = norm(np.around(stress[i], 6) / 1e6)
                color = cmap(index)
                vertices_x = np.append(vertex[i][:, 0], vertex[i][0, 0])
                vertices_y = np.append(vertex[i][:, 1], vertex[i][0, 1])
                plt.fill(vertices_x, vertices_y, color=color)

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(plt.gca())

            cax = divider.append_axes("right", size="10%", pad=0.2)
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

        plt.figure()

        plt.subplot(2, 1, 1)
        plot(sigma, vertices, name_stress="sigma")
        plt.subplot(2, 1, 2)
        plot(tau, vertices, name_stress="tau")

        if save is not None:
            plt.savefig(save)

    def plot_stress_profile(self, cf_index=0, save=None):
        stresses = []
        x = []
        counter = 0
        for cp in self.list_cfs[cf_index].cps:
            counter += 1
            if not cp.to_ommit():
                stresses.append(cp.sp1.law.stress["s"] / 1e6)
                x.append(cp.x_cp[1] * 100)

        offset = 0.5 / (2 * len(stresses))
        # x = np.linspace(-.25+offset,0.25-offset,len(stresses))

        # x2 = np.linspace(-.25,0.25,100)
        # y_sigma = np.linspace(-36,36,100)
        # y_tau = 6 * 100e3 * (0.25**2 - (x2)**2) / (0.5**3 * 0.2)
        # # y_tau = - (5*100e3) / (0.5**2 * 0.5 * 0.2) * x2**2 + 5 * 100e3 / (4*0.5*0.2)
        # y_tau = 100e3 * (1 - 6*x2/0.5**2 + 4*x2**3/0.5**3) / (0.5*0.2)

        # print(max(stresses))

        plt.figure(None, figsize=(5, 5), dpi=600)
        # plt.scatter(x*100, stresses, label='HybriDFEM', marker='.', color='blue')
        plt.bar(
            x,
            stresses,
            label="HybriDFEM",
            facecolor="white",
            edgecolor="blue",
            linewidth=1,
            width=50 / counter,
        )
        # print(x)
        # print(stresses)
        # plt.plot(str,y_sigma,label='Analytical',color='red')
        # elif stress=='tau':
        #     plt.plot(x2*100,y_tau/1e6,label='Analytical',color='red')
        plt.legend(fontsize=12)
        plt.ylabel(r"Stress [MPa]")
        plt.xlabel(r"Height [cm]")
        plt.grid(True, linestyle="--", linewidth=0.3)

        if save:
            plt.savefig(save)


class Structure_FEM(Structure_2D):
    def __init__(self):
        super().__init__()
        self.list_fes: List[FE] = []

    # Construction methods
    def add_fe(self, N1, N2, E, nu, h, b=1, lin_geom=True, rho=0.0):
        self.list_fes.append(
            Timoshenko_FE_2D(N1, N2, E, nu, b, h, lin_geom=lin_geom, rho=rho)
        )

    # Generation methods
    def _make_nodes_fem(self):
        for fe in self.list_fes:
            for j, node in enumerate(fe.nodes):
                index = self._add_node_if_new(node)
                fe.make_connect(index, j)

    def make_nodes(self):
        self._make_nodes_fem()

        self.nb_dofs = 3 * len(self.list_nodes)
        self.U = np.zeros(self.nb_dofs, dtype=float)
        self.P = np.zeros(self.nb_dofs, dtype=float)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=float)
        rot_to_fix = []
        for fe in self.list_fes:
            if isinstance(fe, FE_2D):
                rot_to_fix.extend(fe.rotation_dofs.tolist())

        if rot_to_fix:
            self.dof_fix = np.unique(np.array(rot_to_fix, dtype=int))
            self.dof_free = np.setdiff1d(np.arange(self.nb_dofs, dtype=int), self.dof_fix, assume_unique=False)
        else:
            self.dof_fix = np.array([], dtype=int)
            self.dof_free = np.arange(self.nb_dofs, dtype=int)

        self.nb_dof_fix = len(self.dof_fix)
        self.nb_dof_free = len(self.dof_free)

    # Solving methods
    def _get_P_r_fem(self):
        self.dofs_defined()
        if not hasattr(self, "P_r"):
            self.P_r = np.zeros(self.nb_dofs, dtype=float)

        for fe in self.list_fes:
            q_glob = self.U[fe.dofs]
            p_glob = fe.get_p_glob(q_glob)
            self.P_r[fe.dofs] += p_glob

    def get_P_r(self):
        self.P_r = np.zeros(self.nb_dofs, dtype=float)
        self._get_P_r_fem()

    def _mass_fem(self, no_inertia: bool = False):
        for fe in getattr(self, "list_fes", []):
            mass_fe = fe.get_mass(no_inertia=no_inertia)
            if mass_fe is None:
                continue
            dofs = np.asarray(fe.dofs, dtype=int)
            self.M[np.ix_(dofs, dofs)] += mass_fe

    def get_M_str(self, no_inertia: bool = False):
        self.dofs_defined()
        self.M = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._mass_fem(no_inertia=no_inertia)
        return self.M

    def _stiffness_fem(self):
        for fe in getattr(self, "list_fes", []):
            k_glob = fe.get_k_glob()
            dofs = np.asarray(fe.dofs, dtype=int)
            self.K[np.ix_(dofs, dofs)] += k_glob

    def _stiffness0_fem(self):
        for fe in getattr(self, "list_fes", []):
            k_glob0 = fe.get_k_glob0()
            dofs = np.asarray(fe.dofs, dtype=int)
            self.K0[np.ix_(dofs, dofs)] += k_glob0

    def _stiffness_LG_fem(self):
        for fe in getattr(self, "list_fes", []):
            k_glob_LG = fe.get_k_glob_LG()
            dofs = np.asarray(fe.dofs, dtype=int)
            self.K_LG[np.ix_(dofs, dofs)] += k_glob_LG

    def get_K_str(self):
        self.dofs_defined()
        self.K = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_fem()
        return self.K

    def get_K_str0(self):
        self.dofs_defined()
        self.K0 = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness0_fem()
        return self.K0

    def get_K_str_LG(self):
        self.dofs_defined()
        self.K_LG = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_LG_fem()
        return self.K_LG

    def set_lin_geom(self, lin_geom=True):
        for fe in self.list_fes:
            fe.lin_geom = lin_geom


class Hybrid(Structure_block, Structure_FEM):
    def __init__(self):
        super().__init__()

    def make_nodes(self):
        self.list_nodes = []

        # Call each method separately
        self._make_nodes_block()
        self._make_nodes_fem()

        # Initialize DOFs
        self.nb_dofs = 3 * len(self.list_nodes)
        self.U = np.zeros(self.nb_dofs, dtype=float)
        self.P = np.zeros(self.nb_dofs, dtype=float)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=float)
        self.dof_fix = np.array([], dtype=int)
        self.dof_free = np.arange(self.nb_dofs, dtype=int)
        self.nb_dof_fix = 0
        self.nb_dof_free = self.nb_dofs

    def get_P_r(self):
        self.dofs_defined()
        self.P_r = np.zeros(self.nb_dofs, dtype=float)

        self._get_P_r_block()
        self._get_P_r_fem()

    def get_M_str(self, no_inertia: bool = False):
        self.dofs_defined()
        self.M = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)

        # Compose contributions
        self._mass_block(no_inertia=no_inertia)
        self._mass_fem(no_inertia=no_inertia)

        return self.M

    def get_K_str(self):
        self.dofs_defined()
        self.K = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_block()
        self._stiffness_fem()
        return self.K

    def get_K_str0(self):
        self.dofs_defined()
        self.K0 = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness0_block()
        self._stiffness0_fem()
        return self.K0

    def get_K_str_LG(self):
        self.dofs_defined()
        self.K_LG = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_LG_block()
        self._stiffness_LG_fem()
        return self.K_LG

    def set_lin_geom(self, lin_geom=True):
        for cf in self.list_cfs:
            cf.set_lin_geom(lin_geom)

        for fe in self.list_fes:
            fe.lin_geom = lin_geom

    def plot_def_structure(
            self, scale=0, plot_cf=True, plot_forces=True, plot_supp=True, lighter=False
    ):
        # self.get_P_r()

        for block in self.list_blocks:
            block.disps = self.U[block.dofs]
            block.plot_block(scale=scale, lighter=lighter)

        for fe in self.list_fes:
            if scale == 0:
                fe.PlotUndefShapeElem()
            else:
                defs = self.U[fe.dofs]
                fe.PlotDefShapeElem(defs, scale=scale)

        # for cf in self.list_cfs:
        #     if cf.cps[0].sp1.law.tag == 'CTC':
        #         if cf.cps[0].sp1.law.cracked:
        #             disp1 = self.U[cf.bl_A.dofs[0]]
        #             disp2 = self.U[cf.bl_B.dofs[0]]
        #             cf.plot_cf(scale, disp1, disp2)

        if plot_cf:
            for cf in self.list_cfs:
                cf.plot_cf(scale)

        if plot_forces:
            for i in self.dof_free:
                if self.P[i] != 0:
                    node_id = int(i / 3)
                    dof = i % 3

                    start = (
                            self.list_nodes[node_id]
                            + scale
                            * self.U[
                                3 * node_id * np.ones(2, dtype=int)
                                + np.array([0, 1], dtype=int)
                                ]
                    )
                    arr_len = 0.3

                    if dof == 0:
                        end = arr_len * np.array([1, 0]) * np.sign(self.P[i])
                        plt.arrow(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            head_width=0.05,
                            head_length=0.075,
                            fc="green",
                            ec="green",
                        )
                    elif dof == 1:
                        end = arr_len * np.array([0, 1]) * np.sign(self.P[i])
                        plt.arrow(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            head_width=0.05,
                            head_length=0.075,
                            fc="green",
                            ec="green",
                        )
                    else:
                        if np.sign(self.P[i]) == 1:
                            plt.plot(
                                start[0],
                                start[1],
                                marker="o",
                                markerfacecolor="None",
                                markeredgecolor="green",
                                markersize=10,
                            )
                            plt.plot(
                                start[0],
                                start[1],
                                marker=".",
                                markerfacecolor="green",
                                markeredgecolor="green",
                                markersize=5,
                            )
                        else:
                            plt.plot(
                                start[0],
                                start[1],
                                marker="o",
                                markerfacecolor="None",
                                markeredgecolor="green",
                                markersize=10,
                            )
                            plt.plot(
                                start[0],
                                start[1],
                                marker="x",
                                markerfacecolor="None",
                                markeredgecolor="green",
                                markersize=10,
                            )

                if self.P_fixed[i] != 0:
                    node_id = int(i / 3)
                    dof = i % 3

                    start = (
                            self.list_nodes[node_id]
                            + scale
                            * self.U[
                                3 * node_id * np.ones(2, dtype=int)
                                + np.array([0, 1], dtype=int)
                                ]
                    )
                    arr_len = 0.3

                    if dof == 0:
                        end = arr_len * np.array([1, 0]) * np.sign(self.P_fixed[i])
                        plt.arrow(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            head_width=0.05,
                            head_length=0.075,
                            fc="red",
                            ec="red",
                        )
                    elif dof == 1:
                        end = arr_len * np.array([0, 1]) * np.sign(self.P_fixed[i])
                        plt.arrow(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            head_width=0.05,
                            head_length=0.075,
                            fc="red",
                            ec="red",
                        )
                    else:
                        if np.sign(self.P_fixed[i]) == 1:
                            plt.plot(
                                start[0],
                                start[1],
                                marker="o",
                                markerfacecolor="None",
                                markeredgecolor="red",
                                markersize=10,
                            )
                            plt.plot(
                                start[0],
                                start[1],
                                marker=".",
                                markerfacecolor="red",
                                markeredgecolor="red",
                                markersize=5,
                            )
                        else:
                            plt.plot(
                                start[0],
                                start[1],
                                marker="o",
                                markerfacecolor="None",
                                markeredgecolor="red",
                                markersize=10,
                            )
                            plt.plot(
                                start[0],
                                start[1],
                                marker="x",
                                markerfacecolor="None",
                                markeredgecolor="red",
                                markersize=10,
                            )

        if plot_supp:
            for fix in self.dof_fix:
                node_id = int(fix / 3)
                dof = fix % 3

                node = (
                        self.list_nodes[node_id]
                        + scale
                        * self.U[
                            3 * node_id * np.ones(2, dtype=int)
                            + np.array([0, 1], dtype=int)
                            ]
                )

                import matplotlib as mpl

                if dof == 0:
                    mark = mpl.markers.MarkerStyle(marker=5)
                elif dof == 1:
                    mark = mpl.markers.MarkerStyle(marker=6)
                else:
                    mark = mpl.markers.MarkerStyle(marker="x")

                plt.plot(node[0], node[1], marker=mark, color="blue", markersize=8)

    def plot_modes(
            self, modes=None, scale=1, save=False, lims=None, folder=None, show=True
    ):
        if not hasattr(self, "eig_modes"):
            warnings.warn("Eigen modes were not determined yet")

        if modes is None:
            modes = self.nb_dof_free

        if len(self.eig_vals) < modes:
            warnings.warn("Asking for too many modes, fewer were computed")

        for i in range(modes):
            self.U[self.dof_free] = self.eig_modes.T[i]

            if lims is None:
                plt.figure(None, dpi=400, figsize=(6, 6))
            else:
                x_len = lims[0][1] - lims[0][0]
                y_len = lims[1][1] - lims[1][0]
                if x_len > y_len:
                    plt.figure(None, dpi=400, figsize=(6, 6 * y_len / x_len))
                else:
                    plt.figure(None, dpi=400, figsize=(6 * x_len / y_len, 6))

            plt.axis("equal")
            plt.axis("off")

            self.plot_def_structure(
                scale=scale, plot_cf=False, plot_forces=False, plot_supp=False
            )
            self.plot_def_structure(
                scale=0, plot_cf=False, plot_forces=False, plot_supp=False, lighter=True
            )

            if lims is not None:
                plt.xlim(lims[0][0], lims[0][1])
                plt.ylim(lims[1][0], lims[1][1])

            w = np.around(self.eig_vals[i], 3)
            f = np.around(self.eig_vals[i] / (2 * np.pi), 3)
            if not w == 0:
                T = np.around(2 * np.pi / w, 3)
            else:
                T = float("inf")
            plt.title(
                rf"$\omega_{{{i + 1}}} = {w}$ rad/s - $T_{{{i + 1}}} = {T}$ s - $f_{{{i + 1}}} = {f}$ "
            )
            if save:
                if folder is not None:
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    plt.savefig(folder + f"/Mode_{i + 1}.eps")
                else:
                    plt.savefig(f"Mode_{i + 1}.eps")

            if not show:
                # print('Closing figure...')
                plt.close()
            else:
                plt.show()

    def plot_structure(self, scale=0, plot_cf=True, plot_forces=True, plot_supp=True, show=True, save=None, lims=None):
        desired_aspect = 1.0

        if lims is not None:
            x0, x1 = lims[0][0], lims[0][1]
            xrange = x1 - x0
            y0, y1 = lims[1][0], lims[1][1]
            yrange = y1 - y0
            aspect = xrange / yrange

            if aspect > desired_aspect:
                center_y = (y0 + y1) / 2
                yrange_new = xrange
                y0 = center_y - yrange_new / 2
                y1 = center_y + yrange_new / 2
            else:
                center_x = (x0 + x1) / 2
                xrange_new = yrange
                x0 = center_x - xrange_new / 2
                x1 = center_x + xrange_new / 2

        plt.figure(None, dpi=400, figsize=(6, 6))

        # plt.axis('equal')
        plt.axis("off")

        self.plot_def_structure(
            scale=scale, plot_cf=plot_cf, plot_forces=plot_forces, plot_supp=plot_supp
        )

        if lims is not None:
            plt.xlim((x0, x1))
            plt.ylim((y0, y1))

        if save is not None:
            plt.savefig(save)

        if not show:
            # print('Closing figure...')
            plt.close()
        else:
            plt.show()
