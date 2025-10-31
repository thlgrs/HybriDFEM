from typing import Dict, Optional
from typing import Tuple, List

import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.collections import LineCollection


class Mesh:
    """
    Create or read a 2D mesh (triangles or quads, linear or quadratic),
    expose nodes/elements/physical-edge groups, quick plot, and VTK export.
    """

    def __init__(self,
                 points: Optional[List[Tuple[float, float]]] = None,  # Boxing points
                 mesh_file: Optional[str] = None,
                 element_type: str = "triangle",  # 'triangle'/'tri' or 'quad'
                 element_size: float = 0.1,
                 order: int = 2,  # 1=linear, 2=quadratic
                 name: str = "myMesh",
                 edge_groups: Optional[Dict[str, List[int]]] = None,  # indices into boundary edges (CCW)
                 ):
        if points is None and mesh_file is None:
            raise ValueError("Provide either `points` or `mesh_file`.")
        self.points_list = points
        self.mesh_file = mesh_file
        self.element_type = (
            "triangle" if element_type in ("tri", "triangle") else "quad"
        )
        self.element_size = float(element_size)
        self.order = int(order)
        self.name = str(name)
        self.edge_groups = edge_groups or {}
        self._mesh: Optional[meshio.Mesh] = None
        self.generated = False

    # -- Mesh generation -----------------------------------------------------
    def generate_mesh(self) -> None:
        """
        Build a polygon from `points_list`, mesh it with Gmsh, create
        physical groups: 'domain' (surface) and named line groups in edge_groups.
        """
        if self.points_list is None:
            raise RuntimeError(
                "Cannot generate: no geometry defined (points_list is None)."
            )

        gmsh_init_here = not gmsh.isInitialized()
        if gmsh_init_here:
            gmsh.initialize()
        try:
            gmsh.model.add(self.name)

            # Points + boundary lines
            pts = [
                gmsh.model.geo.addPoint(x, y, 0.0, self.element_size)
                for x, y in self.points_list
            ]
            lines = [
                gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)])
                for i in range(len(pts))
            ]
            loop = gmsh.model.geo.addCurveLoop(lines)
            surface = gmsh.model.geo.addPlaneSurface([loop])
            gmsh.model.geo.synchronize()

            # Physical groups
            dom_tag = gmsh.model.addPhysicalGroup(2, [surface])
            gmsh.model.setPhysicalName(2, dom_tag, "domain")
            for name, line_indices in (self.edge_groups or {}).items():
                try:
                    phys = gmsh.model.addPhysicalGroup(
                        1, [lines[i] for i in line_indices]
                    )
                    gmsh.model.setPhysicalName(1, phys, name)
                except Exception as e:
                    print(f"[warn] failed creating physical group '{name}': {e}")

            # Meshing options
            if self.element_type == "quad":
                gmsh.model.mesh.setRecombine(2, surface)
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
            gmsh.option.setNumber("Mesh.ElementOrder", self.order)

            gmsh.model.mesh.generate(2)

            filename = self.mesh_file or f"{self.name}.msh"
            gmsh.write(filename)
            self.mesh_file = filename
            self.generated = True

            self._mesh = meshio.read(self.mesh_file)

            if self._mesh.field_data:
                print("\nMeshio Physical Groups:")
                for name, (tag, dim) in self._mesh.field_data.items():
                    print(f"  '{name}': tag={tag}, dim={dim}")
        finally:
            if gmsh_init_here:
                gmsh.finalize()

    def read_mesh(self) -> meshio.Mesh:
        if self._mesh is None:
            if self.mesh_file is None:
                raise RuntimeError("No mesh available to read.")
            self._mesh = meshio.read(self.mesh_file)
        return self._mesh

    def nodes(self) -> np.ndarray:
        return self.read_mesh().points[:, :2].copy()

    def elements(self) -> np.ndarray:
        """
        Element connectivities for chosen family/order.
        MeshIO names:
          triangle: 'triangle' (3), 'triangle6' (6)
          quad    : 'quad' (4), 'quad8' (8)
        """
        md = self.read_mesh().cells_dict
        if self.element_type == "triangle":
            key = "triangle6" if self.order == 2 else "triangle"
        else:
            key = "quad8" if self.order == 2 else "quad"
        return md.get(key, np.empty((0, 0), dtype=int))

    def plot(
            self, save_path: Optional[str] = None, title: Optional[str] = None
    ) -> None:
        mesh = self.read_mesh()
        pts = mesh.points[:, :2]
        segs: List[Tuple[np.ndarray, np.ndarray]] = []

        for cb in mesh.cells:
            t = cb.type
            data = cb.data
            if t == "line":
                for e in data:
                    segs.append((pts[e[0]], pts[e[1]]))
            elif t == "line3":
                for e in data:
                    segs.append((pts[e[0]], pts[e[2]]))
                    segs.append((pts[e[2]], pts[e[1]]))
            elif t == "triangle":
                for e in data:
                    cyc = [0, 1, 2, 0]
                    for i in range(3):
                        segs.append((pts[e[cyc[i]]], pts[e[cyc[i + 1]]]))
            elif t == "triangle6":
                for e in data:
                    segs += [(pts[e[0]], pts[e[3]]), (pts[e[3]], pts[e[1]])]
                    segs += [(pts[e[1]], pts[e[4]]), (pts[e[4]], pts[e[2]])]
                    segs += [(pts[e[2]], pts[e[5]]), (pts[e[5]], pts[e[0]])]
            elif t == "quad":
                for e in data:
                    cyc = [0, 1, 2, 3, 0]
                    for i in range(4):
                        segs.append((pts[e[cyc[i]]], pts[e[cyc[i + 1]]]))
            elif t == "quad8":
                for e in data:
                    segs += [(pts[e[0]], pts[e[4]]), (pts[e[4]], pts[e[1]])]
                    segs += [(pts[e[1]], pts[e[5]]), (pts[e[5]], pts[e[2]])]
                    segs += [(pts[e[2]], pts[e[6]]), (pts[e[6]], pts[e[3]])]
                    segs += [(pts[e[3]], pts[e[7]]), (pts[e[7]], pts[e[0]])]

        lc = LineCollection(segs, linewidths=0.5, colors="k")
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(title or f"{self.name} ({self.element_type}, order={self.order})")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=160)
            plt.close(fig)
        else:
            plt.show()
