# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:02:25 2024

@author: ibouckaert
"""

# Standart imports
import math
import os
import warnings
from typing import List, Union

import numpy as np
import scipy as sc
from scipy.spatial import cKDTree

from .Block import Block_2D
from .ContactFace import CF_2D
from .FE import FE, Timoshenko, Element2D


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"

warnings.formatwarning = custom_warning_format

from abc import ABC, abstractmethod
class Structure_2D(ABC):
    DOF_PER_NODE = 3  # [ux, uy, rz]

    def __init__(self):
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

    @classmethod
    @abstractmethod
    def from_Rhino(cls):
        pass

    @abstractmethod
    def make_nodes(self):
        pass

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

    @abstractmethod
    def set_lin_geom(self, lin_geom=True):
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

class Structure_block(Structure_2D):
    def __init__(self, listBlocks: Union[List[Block_2D], None] = None):
        super().__init__()
        self.list_blocks = listBlocks or []
        self.list_cfs: List[CF_2D] = []

    @classmethod
    @abstractmethod
    def from_Rhino(cls):
        pass
    
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

    def add_tapered_beam(self, N1, N2, n_blocks, h1, h2, rho, b=1, material=None, contact=None, end_1=True, end_2=True):
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

    def add_arch(self, c, a1, a2, R, n_blocks, h, rho, b=1, material=None, contact=None):
        d_a = (a2 - a1) / n_blocks
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

    def add_wall(self, c1, l_block, h_block, pattern, rho, b=1, material=None, orientation=None):
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

class Structure_FEM(Structure_2D):
    def __init__(self):
        super().__init__()
        self.list_fes: List[FE] = []

    @classmethod
    @abstractmethod
    def from_Rhino(cls):
        pass
    
    # Construction methods
    def add_fe(self, nodes, mat, geom):  # TODO complete with Element2D
        self.list_fes.append(Timoshenko(nodes, mat, geom))

    # Generation methods
    def _make_nodes_fem(self):
        for fe in self.list_fes:
            for j, node in enumerate(fe.nodes):
                index = self._add_node_if_new(node)  # new or existing index of the node of the element in Structure_2D
                fe.make_connect(index, j)  # create the connection vector of the element

    def make_nodes(self):
        self._make_nodes_fem()

        self.nb_dofs = 3 * len(self.list_nodes)
        self.U = np.zeros(self.nb_dofs, dtype=float)
        self.P = np.zeros(self.nb_dofs, dtype=float)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=float)
        rot_to_fix = []
        for fe in self.list_fes:
            if isinstance(fe, Element2D):
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

    @classmethod
    @abstractmethod
    def from_Rhino(cls):
        pass
    
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


class Structure_2D(ABC):
    def __init__(self):
        self.list_nodes = []

    @abstractmethod
    def make_nodes(self):
        pass


class Structure_block(Structure_2D):
    def __init__(self, listBlocks: Union[List[Block_2D], None] = None):
        super().__init__()
        self.list_blocks = listBlocks or []
        self.list_cfs: List[CF_2D] = []

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


class Structure_FEM(Structure_2D):
    def __init__(self):
        super().__init__()
        self.list_fes: List[FE] = []

        def _make_nodes_fem(self):
            for fe in self.list_fes:
                for j, node in enumerate(fe.nodes):
                    index = self._add_node_if_new(
                        node)  # new or existing index of the node of the element in Structure_2D
                    fe.make_connect(index, j)  # create the connection vector of the element

    def make_nodes(self):
        self._make_nodes_fem()

        self.nb_dofs = 3 * len(self.list_nodes)
        self.U = np.zeros(self.nb_dofs, dtype=float)
        self.P = np.zeros(self.nb_dofs, dtype=float)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=float)
        rot_to_fix = []  # no rotation dof for Element2D TODO
        for fe in self.list_fes:
            if isinstance(fe, Element2D):
                rot_to_fix.extend(fe.rotation_dofs.tolist())
        if rot_to_fix:
            self.dof_fix = np.unique(np.array(rot_to_fix, dtype=int))
            self.dof_free = np.setdiff1d(np.arange(self.nb_dofs, dtype=int), self.dof_fix, assume_unique=False)
        else:
            self.dof_fix = np.array([], dtype=int)
            self.dof_free = np.arange(self.nb_dofs, dtype=int)

        self.nb_dof_fix = len(self.dof_fix)
        self.nb_dof_free = len(self.dof_free)


class Hybrid(Structure_block, Structure_FEM):
    def __init__(self):
        super().__init__()

    def make_nodes(self):
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
