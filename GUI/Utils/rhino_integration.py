"""
Rhino Integration Module for HybridFEM
Handles import/export between Rhino 3D and HybridFEM structures
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional


class RhinoImporter:
    """
    Import geometry from Rhino files or text-based geometry definitions
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.geometry_data = {
            'blocks': [],
            'beams': [],
            'materials': {},
            'layers': {}
        }

    def import_from_3dm(self, layer_filter: str = "*"):
        """
        Import from Rhino .3dm file
        
        Note: Requires rhino3dm library
        Install with: pip install rhino3dm
        """
        try:
            import rhino3dm as r3dm
        except ImportError:
            raise ImportError(
                "rhino3dm library not found. "
                "Install it with: pip install rhino3dm"
            )

        file3dm = r3dm.File3dm.Read(str(self.file_path))

        # Extract geometry by layer
        for obj in file3dm.Objects:
            layer = file3dm.Layers.FindIndex(obj.Attributes.LayerIndex)
            layer_name = layer.Name if layer else "Default"

            # Apply layer filter
            if layer_filter != "*" and layer_filter not in layer_name:
                continue

            # Process based on geometry type
            if obj.Geometry.ObjectType == r3dm.ObjectType.Curve:
                self._process_curve(obj.Geometry, layer_name)
            elif obj.Geometry.ObjectType == r3dm.ObjectType.Brep:
                self._process_brep(obj.Geometry, layer_name)
            elif obj.Geometry.ObjectType == r3dm.ObjectType.Mesh:
                self._process_mesh(obj.Geometry, layer_name)

        return self.geometry_data

    def import_from_text(self):
        """
        Import from text file with geometry definitions
        Format follows the structure of your existing text import
        """
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        self._parse_text_format(lines)
        return self.geometry_data

    def _parse_text_format(self, lines: List[str]):
        """Parse text-based geometry format"""
        i = 0
        while i < len(lines):
            line = lines[i].strip().lower()

            # Parse blocks
            if line.startswith("box"):
                block_data = self._parse_block(lines, i)
                if block_data:
                    self.geometry_data['blocks'].append(block_data)
                i += 4  # Skip block definition lines

            # Parse beams/FE elements
            elif line.startswith("fe"):
                beam_data = self._parse_beam(lines, i)
                if beam_data:
                    self.geometry_data['beams'].append(beam_data)
                i += 12  # Skip beam definition lines

            else:
                i += 1

    def _parse_block(self, lines: List[str], start_idx: int) -> Optional[Dict]:
        """Parse a block definition from text"""
        try:
            # Expected format:
            # Box i:
            # vertices line
            # density line
            # material line

            if start_idx + 3 >= len(lines):
                return None

            vertices_line = lines[start_idx + 1].strip()
            density_line = lines[start_idx + 2].strip()
            material_line = lines[start_idx + 3].strip()

            # Parse vertices (example: "[x1,y1],[x2,y2],[x3,y3],[x4,y4]")
            vertices = self._parse_vertices(vertices_line)

            # Parse density (example: "rho = 2400.0")
            density = self._parse_parameter(density_line, "rho")

            # Parse material (example: "material = 0")
            material_id = self._parse_parameter(material_line, "material")

            return {
                'type': 'block',
                'vertices': vertices,
                'density': density,
                'material_id': material_id,
                'b': 1.0  # Default thickness
            }

        except Exception as e:
            print(f"Error parsing block at line {start_idx}: {e}")
            return None

    def _parse_beam(self, lines: List[str], start_idx: int) -> Optional[Dict]:
        """Parse a beam/FE element definition from text"""
        try:
            # Expected format includes nodes, material properties, boundary conditions
            if start_idx + 11 >= len(lines):
                return None

            # Parse node coordinates
            n1_line = lines[start_idx + 1].strip()
            n2_line = lines[start_idx + 2].strip()

            n1 = self._parse_node(n1_line)
            n2 = self._parse_node(n2_line)

            # Parse material properties
            E = self._parse_parameter(lines[start_idx + 3].strip(), "E")
            nu = self._parse_parameter(lines[start_idx + 4].strip(), "nu")
            h = self._parse_parameter(lines[start_idx + 5].strip(), "h")
            b = self._parse_parameter(lines[start_idx + 6].strip(), "b")

            # Parse boundary conditions
            bc_n1 = self._parse_bc(lines[start_idx + 8].strip())
            bc_n2 = self._parse_bc(lines[start_idx + 9].strip())

            return {
                'type': 'beam',
                'nodes': [n1, n2],
                'E': E,
                'nu': nu,
                'h': h,
                'b': b,
                'bc_n1': bc_n1,
                'bc_n2': bc_n2
            }

        except Exception as e:
            print(f"Error parsing beam at line {start_idx}: {e}")
            return None

    def _parse_vertices(self, line: str) -> List[Tuple[float, float]]:
        """Parse vertex coordinates from string"""
        # Remove brackets and split
        line = line.replace('[', '').replace(']', '')
        coords = []

        parts = line.split(',')
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                x = float(parts[i].strip())
                y = float(parts[i + 1].strip())
                coords.append((x, y))

        return coords

    def _parse_node(self, line: str) -> Tuple[float, float]:
        """Parse node coordinates"""
        # Format: "N1 = [x, y]"
        if '=' in line:
            coord_str = line.split('=')[1].strip()
            coord_str = coord_str.replace('[', '').replace(']', '')
            parts = coord_str.split(',')
            return (float(parts[0].strip()), float(parts[1].strip()))
        return (0.0, 0.0)

    def _parse_parameter(self, line: str, param_name: str) -> float:
        """Parse a numeric parameter from line"""
        if '=' in line:
            value_str = line.split('=')[1].strip()
            return float(value_str)
        return 0.0

    def _parse_bc(self, line: str) -> Optional[str]:
        """Parse boundary condition"""
        if "no bc" in line.lower():
            return None
        elif "hinge" in line.lower():
            return "hinge"
        elif "fixed" in line.lower():
            return "fixed"
        elif "roller_x" in line.lower():
            return "roller_x"
        elif "roller_y" in line.lower():
            return "roller_y"
        elif "slider_x" in line.lower():
            return "slider_x"
        elif "slider_y" in line.lower():
            return "slider_y"
        return None

    def _process_curve(self, curve, layer_name: str):
        """Process Rhino curve as beam element"""
        # Extract curve points and create beam
        if hasattr(curve, 'PointAtStart') and hasattr(curve, 'PointAtEnd'):
            start = curve.PointAtStart
            end = curve.PointAtEnd

            beam_data = {
                'type': 'beam',
                'nodes': [(start.X, start.Y), (end.X, end.Y)],
                'layer': layer_name,
                'E': 30e9,  # Default values
                'nu': 0.2,
                'h': 0.2,
                'b': 0.2
            }
            self.geometry_data['beams'].append(beam_data)

    def _process_brep(self, brep, layer_name: str):
        """Process Rhino Brep as block or shell"""
        # For now, treat closed planar Breps as blocks
        # This is simplified - real implementation would be more complex
        pass

    def _process_mesh(self, mesh, layer_name: str):
        """Process Rhino mesh"""
        pass


class RhinoExporter:
    """
    Export HybridFEM results to Rhino-compatible formats
    """

    def __init__(self, structure):
        self.structure = structure

    def export_to_text(self, file_path: str, include_displacements: bool = True):
        """
        Export results to text format that can be read by Grasshopper/Python in Rhino
        """
        with open(file_path, 'w') as f:
            f.write("# HybridFEM Analysis Results\n")
            f.write(f"# Structure Type: {type(self.structure).__name__}\n")
            f.write(f"# Total DOFs: {self.structure.nb_dofs}\n")
            f.write(f"# Number of Nodes: {len(self.structure.list_nodes)}\n\n")

            if include_displacements and self.structure.U is not None:
                f.write("# DISPLACEMENTS\n")
                f.write("# Format: node_id, x_original, y_original, ux, uy, rotation_z\n")

                for i, node in enumerate(self.structure.list_nodes):
                    ux = self.structure.U[3 * i]
                    uy = self.structure.U[3 * i + 1]
                    rz = self.structure.U[3 * i + 2]

                    f.write(f"{i}, {node[0]:.6f}, {node[1]:.6f}, "
                            f"{ux:.6e}, {uy:.6e}, {rz:.6e}\n")

            # Export element information
            if hasattr(self.structure, 'list_blocks'):
                f.write("\n# BLOCKS\n")
                f.write("# Format: block_id, ref_point_x, ref_point_y, "
                        "ux, uy, rotation\n")

                for i, block in enumerate(self.structure.list_blocks):
                    node_idx = block.connect
                    ux = self.structure.U[3 * node_idx] if self.structure.U is not None else 0.0
                    uy = self.structure.U[3 * node_idx + 1] if self.structure.U is not None else 0.0
                    rz = self.structure.U[3 * node_idx + 2] if self.structure.U is not None else 0.0

                    f.write(f"{i}, {block.ref_point[0]:.6f}, {block.ref_point[1]:.6f}, "
                            f"{ux:.6e}, {uy:.6e}, {rz:.6e}\n")

            if hasattr(self.structure, 'list_fes'):
                f.write("\n# FE ELEMENTS\n")
                f.write("# Format: element_id, node1_id, node2_id, "
                        "node1_x, node1_y, node2_x, node2_y\n")

                for i, fe in enumerate(self.structure.list_fes):
                    n1_idx = fe.connect[0]
                    n2_idx = fe.connect[1]
                    n1 = self.structure.list_nodes[n1_idx]
                    n2 = self.structure.list_nodes[n2_idx]

                    f.write(f"{i}, {n1_idx}, {n2_idx}, "
                            f"{n1[0]:.6f}, {n1[1]:.6f}, {n2[0]:.6f}, {n2[1]:.6f}\n")

    def export_to_3dm(self, file_path: str):
        """
        Export to Rhino .3dm file with deformed geometry
        
        Requires rhino3dm library
        """
        try:
            import rhino3dm as r3dm
        except ImportError:
            raise ImportError(
                "rhino3dm library not found. "
                "Install it with: pip install rhino3dm"
            )

        file3dm = r3dm.File3dm()

        # Create layers
        undeformed_layer = r3dm.Layer()
        undeformed_layer.Name = "Undeformed"
        undeformed_layer.Color = (0, 0, 255, 255)  # Blue
        file3dm.Layers.Add(undeformed_layer)

        deformed_layer = r3dm.Layer()
        deformed_layer.Name = "Deformed"
        deformed_layer.Color = (255, 0, 0, 255)  # Red
        file3dm.Layers.Add(deformed_layer)

        # Add geometry
        if hasattr(self.structure, 'list_blocks'):
            for block in self.structure.list_blocks:
                # Add block geometry
                pass  # Implement based on your block representation

        if hasattr(self.structure, 'list_fes'):
            for fe in self.structure.list_fes:
                # Add undeformed beam
                n1 = self.structure.list_nodes[fe.connect[0]]
                n2 = self.structure.list_nodes[fe.connect[1]]

                line = r3dm.LineCurve(
                    r3dm.Point3d(n1[0], n1[1], 0),
                    r3dm.Point3d(n2[0], n2[1], 0)
                )

                attrs = r3dm.ObjectAttributes()
                attrs.LayerIndex = 0
                file3dm.Objects.AddCurve(line, attrs)

                # Add deformed beam if displacements exist
                if self.structure.U is not None:
                    u1 = self.structure.U[fe.dofs[:2]]
                    u2 = self.structure.U[fe.dofs[3:5]]

                    def_line = r3dm.LineCurve(
                        r3dm.Point3d(n1[0] + u1[0], n1[1] + u1[1], 0),
                        r3dm.Point3d(n2[0] + u2[0], n2[1] + u2[1], 0)
                    )

                    attrs = r3dm.ObjectAttributes()
                    attrs.LayerIndex = 1
                    file3dm.Objects.AddCurve(def_line, attrs)

        file3dm.Write(file_path, 7)  # Version 7

    def export_deformed_coordinates(self, file_path: str, scale: float = 1.0):
        """
        Export deformed node coordinates in simple CSV format
        Easy to import into Rhino with Python/Grasshopper
        """
        with open(file_path, 'w') as f:
            f.write("node_id,x_def,y_def,ux,uy,rz\n")

            for i, node in enumerate(self.structure.list_nodes):
                if self.structure.U is not None:
                    ux = self.structure.U[3 * i] * scale
                    uy = self.structure.U[3 * i + 1] * scale
                    rz = self.structure.U[3 * i + 2]

                    x_def = node[0] + ux
                    y_def = node[1] + uy
                else:
                    x_def = node[0]
                    y_def = node[1]
                    ux = uy = rz = 0.0

                f.write(f"{i},{x_def},{y_def},{ux},{uy},{rz}\n")


def create_structure_from_rhino(file_path: str, structure_type: str = "Hybrid",
                                layer_filter: str = "*"):
    """
    Convenience function to create a HybridFEM structure from Rhino file
    
    Args:
        file_path: Path to Rhino file (.3dm or .txt)
        structure_type: Type of structure ("Hybrid", "Structure_FEM", "Structure_block")
        layer_filter: Layer filter for import
    
    Returns:
        Initialized structure with geometry
    """
    from Theo.Objects.Structure_2D import Hybrid, Structure_FEM, Structure_block

    # Import geometry
    importer = RhinoImporter(file_path)

    if file_path.endswith('.3dm'):
        geometry_data = importer.import_from_3dm(layer_filter)
    else:
        geometry_data = importer.import_from_text()

    # Create structure based on type
    if structure_type == "Hybrid":
        structure = Hybrid()
    elif structure_type == "Structure_FEM":
        structure = Structure_FEM()
    elif structure_type == "Structure_block":
        structure = Structure_block()
    else:
        raise ValueError(f"Unknown structure type: {structure_type}")

    # Add blocks
    for block_data in geometry_data['blocks']:
        # Convert to your block addition method
        # structure.add_block(...)
        pass

    # Add beams
    for beam_data in geometry_data['beams']:
        # Convert to your FE addition method
        # structure.add_fe(...)
        pass

    return structure


# Grasshopper/Python script example for Rhino
GRASSHOPPER_SCRIPT_TEMPLATE = """
# Grasshopper Python script to import HybridFEM results
# Copy this into a Python component in Grasshopper

import rhinoscriptsyntax as rs

# Read results file
file_path = r"C:/path/to/results.txt"

nodes = []
displacements = []

with open(file_path, 'r') as f:
    lines = f.readlines()
    
    reading_displacements = False
    for line in lines:
        if line.startswith('# DISPLACEMENTS'):
            reading_displacements = True
            continue
        
        if reading_displacements and not line.startswith('#'):
            parts = line.strip().split(',')
            if len(parts) >= 6:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                ux = float(parts[3])
                uy = float(parts[4])
                rz = float(parts[5])
                
                nodes.append([x, y, 0])
                displacements.append([ux, uy, 0])

# Create points in Rhino
scale = 100  # Scale factor for visualization

original_points = [rs.AddPoint(n) for n in nodes]
deformed_points = [rs.AddPoint([n[0] + d[0]*scale, n[1] + d[1]*scale, 0]) 
                   for n, d in zip(nodes, displacements)]

# Output to Grasshopper
a = original_points
b = deformed_points
"""
