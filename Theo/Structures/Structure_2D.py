# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:02:25 2024

@author: ibouckaert
"""

# Standart imports
import math
import os
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import cKDTree


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"

warnings.formatwarning = custom_warning_format

class Structure_2D(ABC):
    DOF_PER_NODE = 3  # [ux, uy, rz] - Default for backward compatibility

    def __init__(self):
        self.list_nodes = []

        # VARIABLE DOF SUPPORT
        # Track DOFs per node and cumulative offsets for flexible DOF management
        self.node_dof_counts = []  # DOFs for each node (e.g., [3, 3, 2, 2, 3])
        self.node_dof_offsets = [0]  # Cumulative DOF offsets (e.g., [0, 3, 6, 8, 10, 13])

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

    # ============================================================
    # ---------------------- CORE HELPERS ------------------------
    # ============================================================
    def compute_nb_dofs(self):
        """
        Compute total number of DOFs from node DOF tracking.

        Returns
        -------
        int
            Total number of DOFs in the structure.

        Notes
        -----
        Uses node_dof_offsets for variable-DOF structures, or falls back to
        DOF_PER_NODE * num_nodes for fixed-DOF structures.
        """
        if len(self.node_dof_offsets) > 1:
            # Variable DOF mode: use the last cumulative offset
            return self.node_dof_offsets[-1]
        else:
            # Fallback: use fixed DOF_PER_NODE
            return self.DOF_PER_NODE * len(self.list_nodes)

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

    def _add_node_if_new(self, node, tol: float = 1e-9, optimized: bool = True, use_hash: bool = False,
                         dof_count: int = None):
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
        dof_count : int, optional
            Number of DOFs for this node. If None, uses self.DOF_PER_NODE (default 3).
            For variable-DOF structures, specify the actual DOF count per element type.
        """
        # Default DOF count if not specified
        if dof_count is None:
            dof_count = self.DOF_PER_NODE
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
                # Node exists - validate DOF count consistency
                if self.node_dof_counts and self.node_dof_counts[idx] != dof_count:
                    raise ValueError(
                        f"DOF count mismatch: node {idx} has {self.node_dof_counts[idx]} DOFs but {dof_count} was requested")
                # ensure hash maps to the found index
                self._node_hash[key] = int(idx)
                return int(idx)
            # Not found: append and update hash + DOF tracking
            self.list_nodes.append(q.copy())
            new_idx = len(self.list_nodes) - 1
            self._node_hash[key] = new_idx
            # Update DOF tracking
            self.node_dof_counts.append(dof_count)
            self.node_dof_offsets.append(self.node_dof_offsets[-1] + dof_count)
            # Invalidate KD-tree (will rebuild lazily)
            self._kdtree = None
            self._kdtree_n = 0
            return new_idx

        # 2) No-hash path: KD-tree (optimized) or vectorized fallback
        idx = self.get_node_id(q, tol=tol, optimized=optimized)
        if idx is not None:
            # Node exists - validate DOF count consistency
            if self.node_dof_counts and self.node_dof_counts[idx] != dof_count:
                raise ValueError(
                    f"DOF count mismatch: node {idx} has {self.node_dof_counts[idx]} DOFs but {dof_count} was requested")
            return int(idx)

        # Not found: append and invalidate KD-tree cache
        self.list_nodes.append(q.copy())
        new_idx = len(self.list_nodes) - 1
        # Update DOF tracking
        self.node_dof_counts.append(dof_count)
        self.node_dof_offsets.append(self.node_dof_offsets[-1] + dof_count)
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
        """
        Map (node_id, local_dof) -> global dof index.

        Uses node_dof_offsets for flexible DOF-per-node support.
        Falls back to DOF_PER_NODE arithmetic if offsets not initialized.
        """
        if len(self.node_dof_offsets) > node_id + 1:
            # Variable DOF mode: use lookup
            return self.node_dof_offsets[node_id] + int(local_dof)
        else:
            # Fallback for backward compatibility
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
