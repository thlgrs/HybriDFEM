import Rhino
import numpy as np
import rhinoscriptsyntax as rs
import scriptcontext as sc


def export_brep_vertices(filepath):
    # Export blocks (Brep objects)
    guids = rs.AllObjects()
    boxes = []

    for guid in guids:
        geo = rs.coercegeometry(guid)
        if geo is None:
            continue

        brep = None
        if isinstance(geo, Rhino.Geometry.Brep):
            brep = geo
        else:
            brep = Rhino.Geometry.Brep.TryConvertBrep(geo)

        if brep is None:
            continue

        corners = [v.Location for v in brep.Vertices]

        if len(corners) < 4:
            continue

        base_pts = sorted(corners, key=lambda pt: pt.Z)[:4]
        avg_x = sum(pt.X for pt in base_pts) / 4.0
        avg_z = sum(pt.Z for pt in base_pts) / 4.0
        boxes.append(((avg_z, avg_x), corners, guid))

    boxes.sort(key=lambda b: (round(b[0][0], 6), b[0][1]))

    with open(filepath, "w") as f:
        for i, (_, corners, guid) in enumerate(boxes):
            # Enumerate the blocks in the .txt file
            f.write("Box {}\n".format(i + 1))
            for pt in corners:
                f.write("{:.6f} {:.6f} {:.6f}\n".format(pt.X, pt.Y, pt.Z))

            bc = rs.GetUserText(guid, "bc")
            if bc:
                f.write("bc: {}\n".format(bc))
            else:
                f.write("No bc applied\n")

            load = rs.GetUserText(guid, "load")
            if load:
                f.write("load: {}\n".format(load))
            else:
                f.write("No load applied\n")

            f.write("\n")

            # Compute the point in the lower face (the one with lower Y)
            min_y = min(pt.Y for pt in corners)
            face_pts = [pt for pt in corners if abs(pt.Y - min_y) < 1e-6]
            avg_x = sum(pt.X for pt in face_pts) / len(face_pts)
            avg_z = sum(pt.Z for pt in face_pts) / len(face_pts)
            point = Rhino.Geometry.Point3d(avg_x, min_y, avg_z)

            rs.AddText(str(i + 1), point, height=0.2)

    # FE export
    curve_guids = rs.AllObjects()
    isolated_curves = []

    for guid in curve_guids:
        if rs.IsCurve(guid) and not rs.IsObjectInGroup(guid):
            isolated_curves.append(guid)

    if isolated_curves:
        with open(filepath, "a") as f:
            for curve in isolated_curves:
                points = rs.CurveEditPoints(curve)
                if len(points) == 2:
                    points = sorted(points, key=lambda pt: (pt[2], pt[0]))
                    area = rs.GetUserText(curve, "Area")
                    inertia = rs.GetUserText(curve, "Inertia")
                    E = rs.GetUserText(curve, "El_modulus")
                    nu = rs.GetUserText(curve, "Poisson_coeff")

                    f.write("Line:\n")
                    for pt in points:
                        f.write("{:.6f} {:.6f} {:.6f}\n".format(pt[0], pt[1], pt[2]))

                    if area:
                        f.write("A: {}\n".format(area))
                    else:
                        f.write("A: Not Assigned\n")

                    if inertia:
                        f.write("I: {}\n".format(inertia))
                    else:
                        f.write("I: Not Assigned\n")

                    if E:
                        f.write("E: {}\n".format(E))
                    else:
                        f.write("E: Not Assigned\n")

                    if nu:
                        f.write("nu: {}\n".format(nu))
                    else:
                        f.write("nu: Not Assigned\n")

                    bcN1 = rs.GetUserText(curve, "bcN1")
                    if bcN1:
                        f.write("bcN1: {}\n".format(bcN1))
                    else:
                        f.write("No bc applied to N1\n")

                    bcN2 = rs.GetUserText(curve, "bcN2")
                    if bcN2:
                        f.write("bcN2: {}\n".format(bcN2))
                    else:
                        f.write("No bc applied to N2\n")

                    loadN1 = rs.GetUserText(curve, "loadN1")
                    if loadN1:
                        f.write("loadN1: {}\n".format(loadN1))
                    else:
                        f.write("No load applied to N1\n")

                    loadN2 = rs.GetUserText(curve, "loadN2")
                    if loadN2:
                        f.write("loadN2: {}\n".format(loadN2))
                    else:
                        f.write("No load applied to N2\n")

                    f.write("\n")


def add_geometry(self, filepath=r"", rho=2000, material=None):

    def read_geometry_from_txt(filepath):
        boxes = []
        lines = []
        current_box = []

        with open(filepath, 'r') as file:
            lines_list = file.readlines()

        i = 0
        while i < len(lines_list):
            line = lines_list[i].strip()

            if line.lower().startswith("box:"):
                if current_box:
                    boxes.append(current_box)
                    current_box = []
                i += 1
                continue

            if line.lower().startswith("line:"):
                try:
                    # Read 2 points
                    p1 = tuple(map(float, lines_list[i + 1].strip().split()))
                    p2 = tuple(map(float, lines_list[i + 2].strip().split()))
                    # Read parameters
                    A = float(lines_list[i + 3].strip().split(":")[1])
                    I = float(lines_list[i + 4].strip().split(":")[1])
                    E = float(lines_list[i + 5].strip().split(":")[1])
                    nu = float(lines_list[i + 6].strip().split(":")[1])

                    lines.append({
                        "N1": (p1[0], p1[2]),  # (X, Z)
                        "N2": (p2[0], p2[2]),  # (X, Z)
                        "A": A,
                        "I": I,
                        "E": E,
                        "nu": nu
                    })
                    i += 7
                except Exception as e:
                    print(f"Error parsing line at index {i}: {e}")
                    i += 1
                continue

            # Points of a box - reads only if there are exactly 3 numbers
            if line:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        x, y, z = map(float, parts)
                        current_box.append((x, y, z))
                    except ValueError:
                        pass
            i += 1

        if current_box:
            boxes.append(current_box)

        return boxes, lines

    boxes, line_elements = read_geometry_from_txt(filepath)

    # Create the blocks
    for i, box in enumerate(boxes):
        if len(box) != 8:
            print(f"Box {i} has {len(box)} points, expected 8.")
            continue

        min_y = min(abs(pt[1]) for pt in box)
        front_pts = [(x, z) for (x, y, z) in box if abs(y) - min_y < 1e-6]

        if len(front_pts) != 4:
            print(f"Box {i} has {len(front_pts)} frontal points, expected 4.")
            continue

        front_pts_sorted = sorted(front_pts, key=lambda pt: (pt[1], pt[0]))

        bottom = sorted(front_pts_sorted[:2], key=lambda pt: pt[0])
        top = sorted(front_pts_sorted[2:], key=lambda pt: pt[0])
        ordered = [bottom[0], bottom[1], top[1], top[0]]

        vertices = np.array(ordered)
        self.add_block(vertices, rho, b=1, material=None, ref_point=None)

    self.make_nodes()

    # Create the FEs
    for i, fe in enumerate(line_elements):
        N1 = fe["N1"]
        N2 = fe["N2"]
        A = fe["A"]
        I = fe["I"]
        E = fe["E"]
        nu = fe["nu"]

        b = 1.0  # default value
        h = (12 * I / b) ** (1 / 3)

        N1 = np.array(fe["N1"], dtype=float)
        N2 = np.array(fe["N2"], dtype=float)

        fe["nodes"] = [N1, N2]

        self.add_fe(N1, N2, E, nu, h, b=1, lin_geom=True, rho=0.)

    self.make_nodes()

    def add_boundary_conditions(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Blocks
        box_index = 0
        i = 0
        while i < len(lines):
            if lines[i].strip().lower().startswith("box:") and box_index < len(self.list_blocks):
                box = self.list_blocks[box_index]

                # Verifica la riga per il BC
                if i + 9 < len(lines):
                    bc_line = lines[i + 9].strip()  # La linea BC per il blocco è alla riga i + 9

                    # Applica il BC del blocco usando il ref_point come nodo
                    node_coords = box.ref_point
                    match (bc_line):
                        case "No bc applied":
                            pass  # Non fare nulla se non ci sono BC applicati
                        case "bc: hinge":
                            self.fixNode(node_coords, [0, 1])
                        case "bc: fixed":
                            self.fixNode(node_coords, [0, 1, 2])
                        case "bc: roller_x":
                            self.fixNode(node_coords, [0])
                        case "bc: roller_y":
                            self.fixNode(node_coords, [1])
                        case "bc: slider_x":
                            self.fixNode(node_coords, [0, 2])
                        case "bc: slider_y":
                            self.fixNode(node_coords, [1, 2])
                        case _:
                            print(f"Unrecognized bc on block {box_index}: {bc_line}")
                box_index += 1
                i += 11  # Vai alla prossima sezione di blocco
            else:
                i += 1  # Se non è un blocco, continua a scorrere le righe

        # FEs
        fe_index = 0
        i = 0
        while i < len(lines):
            if lines[i].strip().lower().startswith("line:") and fe_index < len(self.list_fes):
                fe = self.list_fes[fe_index]
                if i + 6 < len(lines):
                    bcN1_line = lines[i + 7].strip()
                    match (bcN1_line):
                        case "No bc applied to N1":
                            pass
                        case "bcN1: hinge":
                            self.fixNode(np.array(fe.nodes[0]), [0, 1])
                        case "bcN1: fixed":
                            self.fixNode(np.array(fe.nodes[0]), [0, 1, 2])
                        case "bcN1: roller_x":
                            self.fixNode(np.array(fe.nodes[0]), [0])
                        case "bcN1: roller_y":
                            self.fixNode(np.array(fe.nodes[0]), [1])
                        case "bcN1: slider_x":
                            self.fixNode(np.array(fe.nodes[0]), [0, 2])
                        case "bcN1: slider_y":
                            self.fixNode(np.array(fe.nodes[0]), [1, 2])
                        case _:
                            print(f"Unrecognized bcN1 on line {i + 6}: {bcN1_line}")

                if i + 7 < len(lines):
                    bcN2_line = lines[i + 8].strip()
                    match (bcN2_line):
                        case "No bc applied to N2":
                            pass
                        case "bcN2: hinge":
                            self.fixNode(np.array(fe.nodes[1]), [0, 1])
                        case "bcN2: fixed":
                            self.fixNode(np.array(fe.nodes[1]), [0, 1, 2])
                        case "bcN2: roller_x":
                            self.fixNode(np.array(fe.nodes[1]), [0])
                        case "bcN2: roller_y":
                            self.fixNode(np.array(fe.nodes[1]), [1])
                        case "bcN2: slider_x":
                            self.fixNode(np.array(fe.nodes[1]), [0, 2])
                        case "bcN2: slider_y":
                            self.fixNode(np.array(fe.nodes[1]), [1, 2])
                        case _:
                            print(f"Unrecognized bcN2 on line {i + 7}: {bcN2_line}")
                fe_index += 1
                i += 11
            else:
                i += 1

    add_boundary_conditions(filepath)

    def add_loads(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # Blocks
        box_index = 0
        i = 0
        while i < len(lines):
            if lines[i].strip().lower().startswith("box:") and box_index < len(self.list_blocks):
                box = self.list_blocks[box_index]

                # Verifica la riga per il Load
                if i + 10 < len(lines):
                    load_line = lines[i + 10].strip()

                    node_coords = box.ref_point

                    if load_line == "No load applied":
                        pass
                    elif load_line.lower().startswith("load:"):
                        try:
                            load_info = load_line[5:].strip()  # "load" takes the first 4 caracters
                            load_type, force_str = load_info.split(";")
                            load_type = load_type.strip().lower()
                            force_value = float(force_str.strip())
                            match (load_type):
                                case "horizontaldead":
                                    self.loadNode(node_coords, [0], force_value, True)
                                case "horizontallive":
                                    self.loadNode(node_coords, [0], force_value, False)
                                case "verticaldead":
                                    self.loadNode(node_coords, [1], force_value, True)
                                case "verticallive":
                                    self.loadNode(node_coords, [1], force_value, False)
                                case _:
                                    print(f"Unrecognized load type on block {box_index}: {load_type}")
                                    box_index += 1
                                    i += 11
                        except Exception as e:
                            print(f"Error parsing load on block {box_index}: {e}")

                    else:
                        print(f"Unrecognized load line on block {box_index}: {load_line}")

                box_index += 1
                i += 11
            else:
                i += 1

                # FEs
        fe_index = 0
        i = 0
        while i < len(lines):
            if lines[i].strip().lower().startswith("line:") and fe_index < len(self.list_fes):
                fe = self.list_fes[fe_index]

                # Verifica la riga per il Load per N1
                if i + 8 < len(lines):  # La riga 9 è per loadN1
                    loadN1_line = lines[i + 9].strip()

                    if loadN1_line == "No load applied to N1":
                        pass
                    elif loadN1_line.lower().startswith("loadn1:"):
                        try:
                            loadN1_info = loadN1_line[7:].strip()  # "loadN1" takes 6 caracters
                            load_type, force_str = loadN1_info.split(";")
                            load_type = load_type.strip().lower()
                            force_value = float(force_str.strip())
                            match (load_type):
                                case "horizontaldead":
                                    self.loadNode(np.array(fe.nodes[0]), [0], force_value, True)
                                case "horizontallive":
                                    self.loadNode(np.array(fe.nodes[0]), [0], force_value, False)
                                case "verticaldead":
                                    self.loadNode(np.array(fe.nodes[0]), [1], force_value, True)
                                case "verticallive":
                                    self.loadNode(np.array(fe.nodes[0]), [1], force_value, False)
                                case _:
                                    print(f"Unrecognized load type on FE {fe_index}: {load_type}")
                                    fe_index += 1
                                    i += 11

                        except Exception as e:
                            print(f"Error parsing load on FE {fe_index}, N1: {e}")

                    else:
                        print(f"Unrecognized load line on FE {fe_index}, N1: {loadN1_line}")

                # Verifica la riga per il Load per N2
                if i + 9 < len(lines):  # La riga 10 è per loadN2
                    loadN2_line = lines[i + 10].strip()

                    if loadN2_line == "No load applied to N2":
                        pass
                    elif loadN2_line.lower().startswith("loadn2:"):
                        try:
                            loadN2_info = loadN2_line[7:].strip()  # "loadN2" takes 6 caracters
                            load_type, force_str = loadN2_info.split(";")
                            load_type = load_type.strip().lower()
                            force_value = float(force_str.strip())
                            match (load_type):
                                case "horizontaldead":
                                    self.loadNode(np.array(fe.nodes[1]), [0], force_value, True)
                                case "horizontallive":
                                    self.loadNode(np.array(fe.nodes[1]), [0], force_value, False)
                                case "verticaldead":
                                    self.loadNode(np.array(fe.nodes[1]), [1], force_value, True)

                                case "verticallive":
                                    self.loadNode(np.array(fe.nodes[1]), [1], force_value, False)

                                case _:
                                    print(f"Unrecognized load type on FE {fe_index}: {load_type}")
                                    fe_index += 1
                                    i += 11

                        except Exception as e:
                            print(f"Error parsing load on FE {fe_index}, N2: {e}")

                    else:
                        print(f"Unrecognized load line on FE {fe_index}, N2: {loadN2_line}")

                fe_index += 1
                i += 11
            else:
                i += 1

    add_loads(filepath)




if __name__ == "__main__":
    path = rs.SaveFileName("Save blocks and lines vertices", "Text Files (*.txt)|*.txt||")
    if path:
        export_brep_vertices(path)
        print("Export procedure completed:", path)