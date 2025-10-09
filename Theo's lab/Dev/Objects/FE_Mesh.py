from typing import List, Tuple, Dict, Optional

import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.collections import LineCollection


class Mesh:
    @staticmethod
    def compute_physical_node_groups(mesh: meshio.Mesh) -> Dict[int, List[int]]:
        """
        Extracts mapping from physical group tags to node indices for 1D groups (lines).
        Returns:
            {physical_tag: [node_idx, ...]}
        """
        groups: Dict[int, List[int]] = {}
        if 'line' in mesh.cells_dict and 'gmsh:physical' in mesh.cell_data_dict:
            line_cells = mesh.cells_dict['line']
            phys_data = mesh.cell_data_dict['gmsh:physical']['line']
            for cell, tag in zip(line_cells, phys_data):
                for nid in cell:
                    groups.setdefault(tag, []).append(int(nid))
        for tag, nodes in groups.items():
            groups[tag] = sorted(set(nodes))
        return groups

    @classmethod
    def from_json(cls, filename: str):
        def readJSON():
            return None

        param = readJSON()
        return cls(param)

    def __init__(
            self,
            points: Optional[List[Tuple[float, float]]] = None,
            mesh_file: Optional[str] = None,
            element_type: str = 'triangle',
            element_size: float = 0.1,
            name: str = 'myMesh',
            # new: mapping from boundary‐group name to list of line‐entity tags
            boundary_line_groups: Optional[Dict[str, List[int]]] = None,
            # optional explicit tags to assign (int)
            group_tags: Optional[Dict[str, int]] = None
    ):
        """
        Either provide `points` to generate, or `mesh_file` to load.
        boundary_line_groups: e.g. {"clamped":[0,3], "loaded":[1,2]}
            where the ints are the gmsh curve tags in self._lines.
        group_tags: optional dict of name->physical tag (int).
        """
        if points is None and mesh_file is None:
            raise ValueError("Provide either `points` or `mesh_file`.")
        self.points_list = points
        self.mesh_file = mesh_file
        self.element_type = 'triangle' if element_type in ('triangle', 'tri') else 'quad'
        self.element_size = element_size
        self.name = name
        self.boundary_line_groups = boundary_line_groups
        self.group_tags = group_tags or {}
        self._mesh: Optional[meshio.Mesh] = None
        self.generated = False

        # geometry placeholders
        self._surface: Optional[int] = None
        self._lines: Optional[List[int]] = None

        if self.points_list is not None:
            gmsh.initialize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.points_list is not None:
            gmsh.finalize()

    def generate_mesh(self) -> None:
        if self.points_list is None:
            raise RuntimeError("Cannot generate: no geometry defined.")
        gmsh.model.add(self.name)
        gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        # create points & lines
        pts = [gmsh.model.geo.addPoint(x, y, 0, self.element_size)
               for x, y in self.points_list]
        self._lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)])
                       for i in range(len(pts))]
        cl = gmsh.model.geo.addCurveLoop(self._lines)
        self._surface = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()

        # now create physical groups, with names
        self.create_physical_groups()

        if self.element_type == 'quad':
            gmsh.model.mesh.setRecombine(2, self._surface)
        gmsh.model.mesh.generate(2)

        filename = self.mesh_file or f"{self.name}.msh"
        gmsh.write(filename)
        self.mesh_file = filename
        self.generated = True
        self._mesh = meshio.read(self.mesh_file)

    def create_physical_groups(self) -> None:
        """
        Creates:
          - a 2D surface group named 'domain' (or self.name)
          - one or more 1D line groups named per self.boundary_line_groups
        """
        # --- surface group ---
        surf_tag = self.group_tags.get('domain', 1)
        gmsh.model.addPhysicalGroup(2, [self._surface], tag=surf_tag)
        gmsh.model.setPhysicalName(2, surf_tag, 'domain')

        # --- line groups ---
        if self.boundary_line_groups is None:
            # default: lump all lines into one group 'boundary'
            line_tag = self.group_tags.get('boundary', 2)
            gmsh.model.addPhysicalGroup(1, self._lines, tag=line_tag)
            gmsh.model.setPhysicalName(1, line_tag, 'boundary')
        else:
            # user‐specified groups
            used_tags = {surf_tag}
            for name, curve_list in self.boundary_line_groups.items():
                tag = self.group_tags.get(name)
                # pick a fresh integer tag if none given
                if tag is None:
                    tag = max(used_tags) + 1
                used_tags.add(tag)
                gmsh.model.addPhysicalGroup(1, curve_list, tag=tag)
                gmsh.model.setPhysicalName(1, tag, name)

    def read_mesh(self) -> meshio.Mesh:
        if self._mesh is None:
            if self.mesh_file is None:
                raise RuntimeError("No mesh available.")
            self._mesh = meshio.read(self.mesh_file)
        return self._mesh

    def nodes(self) -> np.ndarray:
        return self.read_mesh().points[:, :2]

    def elements(self) -> np.ndarray:
        key = 'quad' if self.element_type == 'quad' else 'triangle'
        return self.read_mesh().cells_dict.get(key, np.empty((0,)))

    def physical_line_groups(self) -> Dict[int, List[int]]:
        return self.compute_physical_node_groups(self.read_mesh())

    def change_size(self, new_size: float, regen: bool = False) -> None:
        """
        Change the mesh element size. Optionally regenerate the mesh.
        """
        self.element_size = new_size
        if self.generated:
            self._mesh = None
            gmsh.model.remove()
            self.generated = False
        if regen:
            self.generate_mesh()

    def change_points(
            self,
            new_points: List[Tuple[float, float]],
            regen: bool = False
    ) -> None:
        """
        Replace the geometry points. Optionally regenerate the mesh.
        """
        if self.generated:
            self._mesh = None
            gmsh.model.remove()
            self.generated = False
        self.points_list = new_points
        if regen:
            self.generate_mesh()

    def change_type(self, new_type: str, regen: bool = False) -> None:
        """
        Change the element type ('triangle' or 'quad'). Optionally regenerate the mesh.
        """
        if new_type not in ('triangle', 'tri', 'quad', 'quadrilateral'):
            raise ValueError("element_type must be 'triangle'/'tri' or 'quad'/'quadrilateral'.")
        updated = 'triangle' if new_type in ('triangle', 'tri') else 'quad'
        if self.generated and updated != self.element_type:
            self._mesh = None
            gmsh.model.remove()
            self.generated = False
        self.element_type = updated
        if regen:
            self.generate_mesh()

    def plot(self, save_path: Optional[str] = None) -> None:
        mesh = self.read_mesh()
        pts = mesh.points
        segments = []
        for ctype in ('line', 'triangle', 'quad'):
            for cell in mesh.cells_dict.get(ctype, []):
                for i in range(len(cell)):
                    a, b = cell[i], cell[(i + 1) % len(cell)]
                    segments.append([pts[a][:2], pts[b][:2]])
        lc = LineCollection(segments, linewidths=0.5, colors='k')
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect('equal')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


# Example usage of the Mesh class
if __name__ == '__main__':
    pts = [(0.0, 0.0), (10, 0.0), (10, 1.0), (0.0, 1.0)]
    # We know lines 0→1: bottom, 1→2: right, 2→3: top, 3→0: left.
    # We want left edge clamped, top edge loaded.

    boundary_groups = {
        "clamped": [3],  # gmsh curve tag for segment 3→0
        "loaded": [2]  # gmsh curve tag for segment 2→3
    }
    group_tags = {"clamped": 10, "loaded": 11}  # optional custom tags

    with Mesh(points=pts,
              element_size=0.5,
              element_type='triangle',
              name='beam',
              boundary_line_groups=boundary_groups,
              group_tags=group_tags) as m:
        m.generate_mesh()
        mesh = m.read_mesh()
        print("Physical groups:", mesh.field_data)
        # now you can pass "clamped" and "loaded" into your FE class
        m.plot()
