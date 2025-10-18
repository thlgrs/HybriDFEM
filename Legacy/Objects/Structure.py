# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:02:25 2024

@author: ibouckaert
"""

# Standard imports 

import importlib
import os
import time
import warnings
from copy import deepcopy

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc


def reload_modules():
    importlib.reload(bl)
    importlib.reload(cf)
    importlib.reload(mat)
    importlib.reload(tfe)


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"


warnings.formatwarning = custom_warning_format

import Legacy.Objects.Block as bl
import Legacy.Objects.ContactFace as cf
import Legacy.Objects.Material as mat
import Legacy.Objects.Timoshenko_FE as tfe

reload_modules()

default = float


class Structure_2D:
    def __init__(self):
        self.list_blocks = []
        self.list_fes = []
        self.list_nodes = []

    def make_nodes(self):
        def node_exists(node):
            for i, n in enumerate(self.list_nodes):
                if np.all(np.isclose(n, node, rtol=1e-8)): return i, True
            return -1, False

        self.list_nodes = []
        for i, bl in enumerate(self.list_blocks):
            index, exists = node_exists(bl.ref_point)
            if exists:
                bl.make_connect(index)
            else:
                self.list_nodes.append(bl.ref_point.copy())
                bl.make_connect(len(self.list_nodes) - 1)
        for i, fe in enumerate(self.list_fes):
            for j, node in enumerate(fe.nodes):
                index, exists = node_exists(node)
                if exists:
                    fe.make_connect(index, j)
                else:
                    self.list_nodes.append(node)
                    fe.make_connect(len(self.list_nodes) - 1, j)
        self.nb_dofs = 3 * len(self.list_nodes)
        self.U = np.zeros(self.nb_dofs, dtype=default)
        self.P = np.zeros(self.nb_dofs, dtype=default)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=default)
        self.dof_fix = np.array([], dtype=int)
        self.dof_free = np.arange(self.nb_dofs, dtype=int)
        self.nb_dof_fix = 0
        self.nb_dof_free = self.nb_dofs

    def add_fe(self, N1, N2, E, nu, h, b=1, lin_geom=True, rho=0.):

        self.list_fes.append(tfe.Timoshenko_FE_2D(N1, N2, E, nu, b, h, lin_geom=lin_geom, rho=rho))

    def add_block(self, vertices, rho, b=1, material=None, ref_point=None):

        self.list_blocks.append(bl.Block_2D(vertices, rho, b=b, material=material, ref_point=ref_point))

    def add_beam(self, N1, N2, n_blocks, h, rho, b=1, material=None, end_1=True, end_2=True):

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

    def add_wall(self, c1, l_block, h_block, pattern, rho, b=1, material=None, orientation=None):

        if orientation is not None:
            long = orientation
            tran = np.array([-orientation[1], orientation[0]])
        else:
            long = np.array([1, 0], dtype=default)
            tran = np.array([0, 1], dtype=default)

        for j, line in enumerate(pattern):

            ref_point = c1 + .5 * abs(line[0]) * l_block * long + (j + 0.5) * h_block * tran

            for i, brick in enumerate(line):

                if brick > 0:
                    vertices = np.array([ref_point, ref_point, ref_point, ref_point])
                    vertices[0] += brick * l_block / 2 * long - h_block / 2 * tran
                    vertices[1] += brick * l_block / 2 * long + h_block / 2 * tran
                    vertices[2] += -brick * l_block / 2 * long + h_block / 2 * tran
                    vertices[3] += -brick * l_block / 2 * long - h_block / 2 * tran

                    self.add_block(vertices, rho, b=b, material=material)

                if not i == len(line) - 1:
                    ref_point += .5 * l_block * long * (abs(brick) + abs(line[i + 1]))

    def add_geometry(self, filepath=r"", rho=2000, material=None, gravity=True):

        def read_geometry_from_txt(filepath):
            boxes = []
            lines = []
            current_box = []

            with open(filepath, 'r') as file:
                lines_list = file.readlines()

            i = 0
            while i < len(lines_list):
                line = lines_list[i].strip()

                if line.lower().startswith("box"):
                    if current_box:
                        boxes.append(current_box)
                        current_box = []
                    i += 1
                    continue

                if line.lower().startswith("line"):
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
                if lines[i].strip().lower().startswith("box") and box_index < len(self.list_blocks):
                    box = self.list_blocks[box_index]

                    # Verifica la riga per il BC
                    if i + 9 < len(lines):
                        bc_line = lines[i + 9].strip()  # La linea BC per il blocco è alla riga i + 9

                        # Applica il BC del blocco usando il ref_point come nodo
                        node_coords = box.ref_point

                        if bc_line == "No bc applied":
                            pass  # Non fare nulla se non ci sono BC applicati
                        elif bc_line == "bc: hinge":
                            self.fixNode(node_coords, [0, 1])
                        elif bc_line == "bc: fixed":
                            self.fixNode(node_coords, [0, 1, 2])
                        elif bc_line == "bc: roller_x":
                            self.fixNode(node_coords, [0])
                        elif bc_line == "bc: roller_y":
                            self.fixNode(node_coords, [1])
                        elif bc_line == "bc: slider_x":
                            self.fixNode(node_coords, [0, 2])
                        elif bc_line == "bc: slider_y":
                            self.fixNode(node_coords, [1, 2])
                        else:
                            print(f"Unrecognized bc on block {box_index}: {bc_line}")

                    box_index += 1
                    i += 11  # Vai alla prossima sezione di blocco
                else:
                    i += 1  # Se non è un blocco, continua a scorrere le righe

            # FEs
            fe_index = 0
            i = 0
            while i < len(lines):
                if lines[i].strip().lower().startswith("line") and fe_index < len(self.list_fes):
                    fe = self.list_fes[fe_index]

                    if i + 6 < len(lines):
                        bcN1_line = lines[i + 7].strip()
                        if bcN1_line == "No bc applied to N1":
                            pass
                        elif bcN1_line == "bcN1: hinge":
                            self.fixNode(np.array(fe.nodes[0]), [0, 1])
                        elif bcN1_line == "bcN1: fixed":
                            self.fixNode(np.array(fe.nodes[0]), [0, 1, 2])
                        elif bcN1_line == "bcN1: roller_x":
                            self.fixNode(np.array(fe.nodes[0]), [0])
                        elif bcN1_line == "bcN1: roller_y":
                            self.fixNode(np.array(fe.nodes[0]), [1])
                        elif bcN1_line == "bcN1: slider_x":
                            self.fixNode(np.array(fe.nodes[0]), [0, 2])
                        elif bcN1_line == "bcN1: slider_y":
                            self.fixNode(np.array(fe.nodes[0]), [1, 2])
                        else:
                            print(f"Unrecognized bcN1 on line {i + 6}: {bcN1_line}")

                    if i + 7 < len(lines):
                        bcN2_line = lines[i + 8].strip()
                        if bcN2_line == "No bc applied to N2":
                            pass
                        elif bcN2_line == "bcN2: hinge":
                            self.fixNode(np.array(fe.nodes[1]), [0, 1])
                        elif bcN2_line == "bcN2: fixed":
                            self.fixNode(np.array(fe.nodes[1]), [0, 1, 2])
                        elif bcN2_line == "bcN2: roller_x":
                            self.fixNode(np.array(fe.nodes[1]), [0])
                        elif bcN2_line == "bcN2: roller_y":
                            self.fixNode(np.array(fe.nodes[1]), [1])
                        elif bcN2_line == "bcN2: slider_x":
                            self.fixNode(np.array(fe.nodes[1]), [0, 2])
                        elif bcN2_line == "bcN2: slider_y":
                            self.fixNode(np.array(fe.nodes[1]), [1, 2])
                        else:
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
                if lines[i].strip().lower().startswith("box") and box_index < len(self.list_blocks):
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

                                if load_type == "horizontaldead":
                                    dofs = [0]
                                    fixed = True
                                elif load_type == "horizontallive":
                                    dofs = [0]
                                    fixed = False
                                elif load_type == "verticaldead":
                                    dofs = [1]
                                    fixed = True
                                elif load_type == "verticallive":
                                    dofs = [1]
                                    fixed = False
                                else:
                                    print(f"Unrecognized load type on block {box_index}: {load_type}")
                                    box_index += 1
                                    i += 11
                                    continue

                                self.loadNode(node_coords, dofs, force_value, fixed)

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
                if lines[i].strip().lower().startswith("line") and fe_index < len(self.list_fes):
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

                                if load_type == "horizontaldead":
                                    dofs = [0]
                                    fixed = True
                                elif load_type == "horizontallive":
                                    dofs = [0]
                                    fixed = False
                                elif load_type == "verticaldead":
                                    dofs = [1]
                                    fixed = True
                                elif load_type == "verticallive":
                                    dofs = [1]
                                    fixed = False
                                else:
                                    print(f"Unrecognized load type on FE {fe_index}: {load_type}")
                                    fe_index += 1
                                    i += 11
                                    continue

                                self.loadNode(np.array(fe.nodes[0]), dofs, force_value, fixed)

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

                                if load_type == "horizontaldead":
                                    dofs = [0]
                                    fixed = True
                                elif load_type == "horizontallive":
                                    dofs = [0]
                                    fixed = False
                                elif load_type == "verticaldead":
                                    dofs = [1]
                                    fixed = True
                                elif load_type == "verticallive":
                                    dofs = [1]
                                    fixed = False
                                else:
                                    print(f"Unrecognized load type on FE {fe_index}: {load_type}")
                                    fe_index += 1
                                    i += 11
                                    continue

                                self.loadNode(np.array(fe.nodes[1]), dofs, force_value, fixed)

                            except Exception as e:
                                print(f"Error parsing load on FE {fe_index}, N2: {e}")

                        else:
                            print(f"Unrecognized load line on FE {fe_index}, N2: {loadN2_line}")

                    fe_index += 1
                    i += 11
                else:
                    i += 1

        add_loads(filepath)

        def add_gravity():
            if gravity == False:
                return

            self.get_M_str()
            g = 9.81  # m/s^2

            for i, block in enumerate(self.list_blocks):
                M = block.m
                W = g * M
                self.loadNode(i, [1], -W, fixed=True)

        add_gravity()

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
                warnings.warn('Not all points lie on the surface')
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
                                vertices.insert(i + 1, C + np.cross(D - C, A - C) / np.cross(D - C, B - C) * (B - A))
                                vertices.insert(i + 2, D)
                                break

                self.add_block(np.array(vertices), rho, b=b, material=material)

    def make_nodes(self):
        def node_exists(node):
            for i, n in enumerate(self.list_nodes):
                if np.all(np.isclose(n, node, rtol=1e-8)): return i, True
            return -1, False
        self.list_nodes = []
        for i, bl in enumerate(self.list_blocks):
            index, exists = node_exists(bl.ref_point)
            if exists:
                bl.make_connect(index)
            else:
                self.list_nodes.append(bl.ref_point.copy())
                bl.make_connect(len(self.list_nodes) - 1)
        for i, fe in enumerate(self.list_fes):
            for j, node in enumerate(fe.nodes):
                index, exists = node_exists(node)
                if exists:
                    fe.make_connect(index, j)
                else:
                    self.list_nodes.append(node)
                    fe.make_connect(len(self.list_nodes) - 1, j)
        self.nb_dofs = 3 * len(self.list_nodes)
        self.U = np.zeros(self.nb_dofs, dtype=default)
        self.P = np.zeros(self.nb_dofs, dtype=default)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=default)
        self.dof_fix = np.array([], dtype=int)
        self.dof_free = np.arange(self.nb_dofs, dtype=int)
        self.nb_dof_fix = 0
        self.nb_dof_free = self.nb_dofs

    def make_cfs(self, lin_geom, nb_cps=2, offset=-1, contact=None, surface=None, weights=None):

        interfaces = self.detect_interfaces()
        self.list_cfs = []

        for i, contactface in enumerate(interfaces):
            self.list_cfs.append(
                cf.CF_2D(contactface, nb_cps, lin_geom, offset=offset, contact=contact, surface=surface,
                         weights=weights))
            self.list_cfs[-1].bl_A.cfs.append(i)
            self.list_cfs[-1].bl_B.cfs.append(i)

    def get_P_r(self):

        if not hasattr(self, 'nb_dofs'): warnings.warn('The DoFs of the structure were not defined')

        self.P_r = np.zeros(self.nb_dofs, dtype=default)

        for CF in self.list_cfs:
            qf_glob = np.zeros(6)
            # print(CF.bl_A.dofs)
            qf_glob[:3] = self.U[CF.bl_A.dofs]
            qf_glob[3:] = self.U[CF.bl_B.dofs]

            pf_glob = CF.get_pf_glob(qf_glob)

            self.P_r[CF.bl_A.dofs] += pf_glob[:3]
            self.P_r[CF.bl_B.dofs] += pf_glob[3:]

        for FE in self.list_fes:
            q_glob = self.U[FE.dofs]
            p_glob = FE.get_p_glob(q_glob)
            self.P_r[FE.dofs] += p_glob

    def fixNode(self, node_ids, dofs):

        def fix_onenode(Str, index):
            Str.dof_fix = np.append(Str.dof_fix, index)
            Str.dof_free = Str.dof_free[Str.dof_free != index]
            Str.nb_dof_fix = len(Str.dof_fix)
            Str.nb_dof_free = len(Str.dof_free)

        if isinstance(node_ids, int):  # Loading one single node
            if isinstance(dofs, int):
                fix_onenode(self, 3 * node_ids + dofs)
            elif isinstance(dofs, list):
                for i in dofs:
                    fix_onenode(self, 3 * node_ids + i)
            else:
                warnings.warn('DoFs to be fixed is not an int or a list')
        elif isinstance(node_ids, list):
            for j in node_ids:
                if isinstance(dofs, int):
                    fix_onenode(self, 3 * j + dofs)
                elif isinstance(dofs, list):
                    for i in dofs:
                        fix_onenode(self, 3 * j + i)
                else:
                    warnings.warn('DoFs to be fixed is not an int or a list')

        elif isinstance(node_ids, np.ndarray) and node_ids.size == 2:  # With coordinates of Node

            node_fixed = -1
            for index, node in enumerate(self.list_nodes):
                if np.allclose(node, node_ids, rtol=1e-9):
                    node_fixed = index
                    break

            if node_fixed < 0:
                warnings.warn('Input node to be fixed does not exist')
            else:
                if isinstance(dofs, int):
                    fix_onenode(self, 3 * node_fixed + dofs)
                elif isinstance(dofs, list):
                    for i in dofs:
                        fix_onenode(self, 3 * node_fixed + i)

        else:
            warnings.warn('Nodes to be loaded must be int, list of ints or numpy array')

    def loadNode(self, node_ids, dofs, force, fixed=False):

        def load_onenode(Str, node_id, dof, force, fixed=False):

            index = 3 * node_id + dof
            if fixed:
                Str.P_fixed[index] += force
            else:
                Str.P[index] += force

        if isinstance(node_ids, int):
            if isinstance(dofs, int):
                load_onenode(self, node_ids, dofs, force, fixed)
            elif isinstance(dofs, list):
                for i in dofs:
                    load_onenode(self, node_ids, i, force, fixed)
            else:
                warnings.warn('DoFs to be loaded is not an int or a list')
        elif isinstance(node_ids, list):
            for j in node_ids:
                if isinstance(dofs, int):
                    load_onenode(self, j, dofs, force, fixed)
                elif isinstance(dofs, list):
                    for i in dofs:
                        load_onenode(self, j, i, force, fixed)
                else:
                    warnings.warn('DoFs to be loaded is not an int or a list')

        elif isinstance(node_ids, np.ndarray) and node_ids.size == 2:  # With coordinates of Node

            node_loaded = -1
            for index, node in enumerate(self.list_nodes):
                if np.allclose(node, node_ids, rtol=1e-9):
                    node_loaded = index
                    break

            if node_loaded < 0:
                warnings.warn('Input node to be loaded does not exist')

            else:
                if isinstance(dofs, int):
                    load_onenode(self, node_loaded, dofs, force, fixed)
                elif isinstance(dofs, list):
                    for i in dofs:
                        load_onenode(self, node_loaded, i, force, fixed)

        else:
            warnings.warn('Nodes to be loaded must be int, list of ints or numpy array')

    def reset_loading(self):

        self.P_fixed = np.zeros(self.nb_dofs)
        self.P = np.zeros(self.nb_dofs)

    def get_node_id(self, node):

        if node.size != 2: warnings.warn('Input node should be an array of size 2')  # With coordinates of Node

        for index, n in enumerate(self.list_nodes):
            if np.allclose(n, node, rtol=1e-9):
                return index

        warnings.warn('Input node to be loaded does not exist')

    def get_M_str(self, no_inertia=False):

        if not hasattr(self, 'nb_dofs'): warnings.warn('The DoFs of the structure were not defined')

        self.M = np.zeros((self.nb_dofs, self.nb_dofs), dtype=default)

        for block in self.list_blocks:
            self.M[np.ix_(block.dofs, block.dofs)] += block.get_mass(no_inertia=no_inertia)

        for FE in self.list_fes:
            mass_fe = FE.get_mass(no_inertia=no_inertia)
            self.M[np.ix_(FE.dofs[:3], FE.dofs[:3])] += mass_fe[:3, :3]
            self.M[np.ix_(FE.dofs[3:], FE.dofs[3:])] += mass_fe[3:, 3:]

    def get_K_str(self):

        if not hasattr(self, 'nb_dofs'): warnings.warn('The DoFs of the structure were not defined')

        self.K = np.zeros((self.nb_dofs, self.nb_dofs), dtype=default)

        for CF in self.list_cfs:
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob = CF.get_kf_glob()

            self.K[np.ix_(dof1, dof1)] += kf_glob[:3, :3]
            self.K[np.ix_(dof1, dof2)] += kf_glob[:3, 3:]
            self.K[np.ix_(dof2, dof1)] += kf_glob[3:, :3]
            self.K[np.ix_(dof2, dof2)] += kf_glob[3:, 3:]

        for FE in self.list_fes:
            k_glob = FE.get_k_glob()

            self.K[np.ix_(FE.dofs[:3], FE.dofs[:3])] += k_glob[:3, :3]
            self.K[np.ix_(FE.dofs[:3], FE.dofs[3:])] += k_glob[:3, 3:]
            self.K[np.ix_(FE.dofs[3:], FE.dofs[:3])] += k_glob[3:, :3]
            self.K[np.ix_(FE.dofs[3:], FE.dofs[3:])] += k_glob[3:, 3:]

    def get_K_str0(self):

        if not hasattr(self, 'nb_dofs'): warnings.warn('The DoFs of the structure were not defined')

        self.K0 = np.zeros((self.nb_dofs, self.nb_dofs), dtype=default)

        for CF in self.list_cfs:
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob = CF.get_kf_glob0()

            self.K0[np.ix_(dof1, dof1)] += kf_glob[:3, :3]
            self.K0[np.ix_(dof1, dof2)] += kf_glob[:3, 3:]
            self.K0[np.ix_(dof2, dof1)] += kf_glob[3:, :3]
            self.K0[np.ix_(dof2, dof2)] += kf_glob[3:, 3:]

        for FE in self.list_fes:
            k_glob = FE.get_k_glob0()

            self.K0[np.ix_(FE.dofs[:3], FE.dofs[:3])] += k_glob[:3, :3]
            self.K0[np.ix_(FE.dofs[:3], FE.dofs[3:])] += k_glob[:3, 3:]
            self.K0[np.ix_(FE.dofs[3:], FE.dofs[:3])] += k_glob[3:, :3]
            self.K0[np.ix_(FE.dofs[3:], FE.dofs[3:])] += k_glob[3:, 3:]

    def get_K_str_LG(self):

        if not hasattr(self, 'nb_dofs'): warnings.warn('The DoFs of the structure were not defined')

        self.K_LG = np.zeros((self.nb_dofs, self.nb_dofs), dtype=default)

        for CF in self.list_cfs:
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob_LG = CF.get_kf_glob_LG()

            self.K_LG[np.ix_(dof1, dof1)] += kf_glob_LG[:3, :3]
            self.K_LG[np.ix_(dof1, dof2)] += kf_glob_LG[:3, 3:]
            self.K_LG[np.ix_(dof2, dof1)] += kf_glob_LG[3:, :3]
            self.K_LG[np.ix_(dof2, dof2)] += kf_glob_LG[3:, 3:]

        for FE in self.list_fes:
            k_glob_LG = FE.get_k_glob0()

            self.K_LG[np.ix_(FE.dofs[:3], FE.dofs[:3])] += k_glob_LG[:3, :3]
            self.K_LG[np.ix_(FE.dofs[:3], FE.dofs[3:])] += k_glob_LG[:3, 3:]
            self.K_LG[np.ix_(FE.dofs[3:], FE.dofs[:3])] += k_glob_LG[3:, :3]
            self.K_LG[np.ix_(FE.dofs[3:], FE.dofs[3:])] += k_glob_LG[3:, 3:]

    def solve_linear(self):

        self.get_P_r()
        self.get_K_str0()

        K_ff = self.K0[np.ix_(self.dof_free, self.dof_free)]
        K_fr = self.K0[np.ix_(self.dof_free, self.dof_fix)]
        K_rf = self.K0[np.ix_(self.dof_fix, self.dof_free)]
        K_rr = self.K0[np.ix_(self.dof_fix, self.dof_fix)]

        self.U[self.dof_free] = sc.linalg.solve(K_ff,
                                                self.P[self.dof_free] + self.P_fixed[self.dof_free] - K_fr @ self.U[
                                                    self.dof_fix])
        # self.P[self.dof_fix] = K_rf @ self.U[self.dof_free] + K_rr @ self.U[self.dof_fix]
        self.get_P_r()

    def commit(self):

        for CF in self.list_cfs:
            CF.commit()

    def revert_commit(self):

        for CF in self.list_cfs:
            CF.revert_commit()

    def solve_forcecontrol(self, steps, tol=1, stiff='tan', max_iter=25, filename='Results_ForceControl', dir_name=''):

        time_start = time.time()

        if isinstance(steps, list):
            nb_steps = len(steps) - 1
            lam = steps

        elif isinstance(steps, int):
            nb_steps = steps
            lam = np.linspace(0, 1, nb_steps + 1)

        else:
            warnings.warn('Steps of the simulation should be either a list or a number of steps (int)')

        # Displacements, forces and stiffness
        U_conv = np.zeros((self.nb_dofs, nb_steps + 1), dtype=default)
        P_r_conv = np.zeros((self.nb_dofs, nb_steps + 1), dtype=default)
        save_k = False
        if save_k:
            K_conv = np.zeros((self.nb_dofs, self.nb_dofs, nb_steps + 1), dtype=default)

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
                    if np.linalg.cond(self.K[np.ix_(self.dof_free, self.dof_free)]) < 1e12:
                        dU = sc.linalg.solve(self.K[np.ix_(self.dof_free, self.dof_free)], R)
                    else:
                        try:
                            dU = sc.linalg.solve(K_conv[:, :, i - 1][np.ix_(self.dof_free, self.dof_free)], R)
                        except:
                            dU = sc.linalg.solve(self.K0[np.ix_(self.dof_free, self.dof_free)], R)

                except np.linalg.LinAlgError:
                    warnings.warn('The tangent and initial stiffnesses are singular')

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
                    print(f'Method did not converge at step {i}')
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

                print(f'Force increment {i} converged after {iteration + 1} iterations')

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f'Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file')

        filename = filename + '.h5'
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, 'w') as hf:

            hf.create_dataset('U_conv', data=U_conv)
            hf.create_dataset('P_r_conv', data=P_r_conv)
            if save_k:
                hf.create_dataset('K_conv', data=K_conv)
            hf.create_dataset('Residuals', data=res_counter)
            hf.create_dataset('Iterations', data=iter_counter)
            hf.create_dataset('Last_conv', data=last_conv)
            hf.create_dataset('Lambda', data=lam)

            hf.attrs['Descr'] = f'Results of the force_control simulation'
            hf.attrs['Tolerance'] = tol
            # hf.attrs['Lambda'] = lam
            hf.attrs['Simulation_Time'] = total_time

    def solve_dispcontrol(self, steps, disp, node, dof, tol=1, stiff='tan', max_iter=25, filename='Results_DispControl',
                          dir_name=''):

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
            warnings.warn('Steps of the simulation should be either a list or a number of steps (int)')
        # Displacements, forces and stiffness

        U_conv = np.zeros((self.nb_dofs, nb_steps + 1), dtype=default)
        P_r_conv = np.zeros((self.nb_dofs, nb_steps + 1), dtype=default)
        save_k = False
        if save_k:
            K_conv = np.zeros((self.nb_dofs, self.nb_dofs, nb_steps + 1), dtype=default)

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
            for n in node: control_dof.append(3 * n + dof)
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

            R = - self.P_r + lam[i] * self.P + self.P_fixed

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
                        solution = np.append(Rf - dU_c * K_fc_conv, Rc - dU_c * K_cc_conv)

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

                R = - self.P_r + lam[i] * self.P + self.P_fixed
                Rf = R[other_dofs]
                Rc = R[control_dof]

                res = np.linalg.norm(R[self.dof_free])

                if res < tol:
                    converged = True
                    self.commit()

                    list_blocks_yielded = []
                    for cf in self.list_cfs:
                        for cp in cf.cps:
                            if cp.sp1.law.tag == 'STC' and cp.sp2.law.tag == 'STC':
                                if cp.sp1.law.yielded or cp.sp2.law.yielded:
                                    list_blocks_yielded.append(cf.bl_A.connect)
                                    list_blocks_yielded.append(cf.bl_B.connect)

                    for cf in self.list_cfs:
                        for cp in cf.cps:
                            if cp.sp1.law.tag == 'BSTC' and cp.sp2.law.tag == 'BSTC':
                                if cf.bl_A.connect in list_blocks_yielded or cf.bl_B.connect in list_blocks_yielded:
                                    # print('Reducing')
                                    cp.sp1.law.reduced = True
                                    cp.sp2.law.reduced = True



                else:
                    # self.revert_commit()
                    iteration += 1
                    dU_c = 0

                if iteration > max_iter and not converged:
                    non_conv = True
                    print(f'Method did not converge at Increment {i}')
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

                print(f'Disp. Increment {i} converged after {iteration + 1} iterations')

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f'Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file')

        filename = filename + '.h5'
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, 'w') as hf:

            hf.create_dataset('U_conv', data=U_conv)
            hf.create_dataset('P_r_conv', data=P_r_conv)
            if save_k:
                hf.create_dataset('K_conv', data=K_conv)
            hf.create_dataset('Residuals', data=res_counter)
            hf.create_dataset('Iterations', data=iter_counter)
            hf.create_dataset('Last_conv', data=last_conv)
            hf.create_dataset('Control_Disp', data=d_c)
            hf.create_dataset('Lambda', data=lam)

            hf.attrs['Descr'] = f'Results of the force_control simulation'
            hf.attrs['Tolerance'] = tol
            hf.attrs['Simulation_Time'] = total_time

    def set_damping_properties(self, xsi=0., damp_type='RAYLEIGH', stiff_type='INIT'):

        if isinstance(xsi, float):
            self.xsi = [xsi, xsi]

        elif isinstance(xsi, list) and len(xsi) == 2:
            self.xsi = xsi

        self.damp_type = damp_type
        self.stiff_type = stiff_type

    def get_C_str(self):

        if not (hasattr(self, 'K')): self.get_K_str()
        # if not (hasattr(self, 'M')): self.get_M_str()

        if not hasattr(self, 'damp_coeff'):

            # No damping
            if self.xsi[0] == 0 and self.xsi[1] == 0:
                self.damp_coeff = np.zeros(2)

            elif self.damp_type == 'RAYLEIGH':

                try:
                    self.solve_modal(modes=2, save=False, initial=True)
                except:
                    self.solve_modal(save=False, initial=True)

                A = np.array([[1 / self.eig_vals[0], self.eig_vals[0]],
                              [1 / self.eig_vals[1], self.eig_vals[1]]])

                if isinstance(self.xsi, float):
                    self.xsi = [self.xsi, self.xsi]
                    self.damp_coeff = 2 * sc.linalg.solve(A, np.array(self.xsi))

                if isinstance(self.xsi, list) and len(self.xsi) == 2:
                    self.damp_coeff = 2 * sc.linalg.solve(A, np.array(self.xsi))

                else:
                    warnings.warn('Xsi is not a list of two damping ratios for Rayleigh damping')

            elif self.damp_type == 'STIFF':

                if not hasattr(self, 'eig_vals'):
                    try:
                        self.solve_modal(modes=1, save=False, initial=True)
                    except:
                        self.solve_modal(save=False, initial=True)
                self.damp_coeff = np.array([0, 2 * self.xsi[0] / self.eig_vals[0]])

            elif self.damp_type == 'MASS':
                try:
                    self.solve_modal(modes=1, save=False, initial=True)
                except:
                    self.solve_modal(save=False, initial=True)
                self.damp_coeff = np.array([2 * self.xsi[0] * self.eig_vals[0], 0])
                print(self.damp_coeff)

        if self.stiff_type == 'INIT':

            if not (hasattr(self, 'C')):
                self.get_K_str0()
                self.C = self.damp_coeff[0] * self.M + self.damp_coeff[1] * self.K0

        elif self.stiff_type == 'TAN':

            self.get_K_str()
            self.C = self.damp_coeff[0] * self.M + self.damp_coeff[1] * self.K

        elif self.stiff_type == 'TAN_LG':

            self.get_K_str_LG()
            self.C = self.damp_coeff[0] * self.M + self.damp_coeff[1] * self.K_LG

    def ask_method(self, Meth=None):

        if Meth is None:

            Meth = input('Which method do you want to use ? CDM, CAA, LA, NWK, WIL, HHT, WBZ or GEN - Default is CDM ')

            if Meth == 'CDM' or Meth == '':
                return Meth, None
            elif Meth == 'CAA' or Meth == 'NWK':
                return 'NWK', {'g': 1 / 2, 'b': 1 / 4}  # If not specified run CAA by default
            elif Meth == 'LA':
                return 'NWK', {'g': 1 / 2, 'b': 1 / 6}
            elif Meth == 'NWK':

                g = input('Which value for Gamma ? - Default is 1/2')
                b = input('Which value for Beta ? - Default is 1/4')

                if g == '':
                    g = 1 / 2
                else:
                    g = float(g)
                if b == '':
                    b = 1 / 4
                else:
                    b = float(b)

                return 'NWK', {'g': g, 'b': b}

            elif Meth == 'WIL':
                t = input('Which value for Gamma ? - Default is 1.5')
                if t == '':
                    t = 1.5
                else:
                    t = float(t)
                if t < 1:
                    warnings.warn('Theta should be larger or equal to one for Wilson\'s theta method')
                elif t < 1.37:
                    warnings.warn(
                        'Theta should be larger or equal to one for unconditional stability in Wilson\'s theta method')
                return 'WIL', {'t': t}

            elif Meth == 'HHT':

                a = input('Which value for Alpha ? - Default is 1/4')
                g = input('Which value for Gamma ? - Default is (1+2a)/2')
                b = input('Which value for Beta ? - Default is (1+a)^2/4')

                if a == '':
                    a = 1 / 4
                else:
                    a = float(a)
                if a < 0 or a > 1 / 3: warnings.warn(
                    'Alpha should be between 0 and 1/3 for unconditional stability in HHT Method')
                if g == '':
                    g = (1 + 2 * a) / 2
                else:
                    g = float(g)
                if b == '':
                    b = (1 + a) ** 2 / 4
                else:
                    b = float(b)

                return 'GEN', {'am': 0, 'af': a, 'g': g, 'b': b}

            elif Meth == 'WBZ':

                a = input('Which value for Alpha ? - Default is 1/2')
                g = input('Which value for Gamma ? - Default is (1-2a)/2')
                b = input('Which value for Beta ? - Default is 1/4')

                if a == '':
                    a = 1 / 2
                else:
                    a = float(a)
                if a > 1 / 2: warnings.warn(
                    'Alpha should be smaller thann 1/2 for unconditional stability in WBZ Method')
                if g == '':
                    g = (1 - 2 * a) / 2
                else:
                    g = float(g)
                if g < (1 - 2 * a) / 2: warnings.warn(
                    'Gamma should be larger than (1-2a)/2 for unconditional stability in WBZ Method')
                if b == '':
                    b = 1 / 4
                else:
                    b = float(b)
                if b < g / 2: warnings.warn('Beta should be larger than g/2 for unconditional stability in WBZ Method')

                return 'GEN', {'am': a, 'af': 0, 'g': g, 'b': b}

            elif Meth == 'GEN':

                m = input('Which value for Mu ? - Default is 1')

                if m == '':
                    m = 1
                else:
                    m = float(m)
                if m < 0 or m > 1: warnings.warn('Mu should be between 0 and 1 for Generalized-alpha Method')

                return 'GEN', {'am': (2 * m - 1) / (m + 1), 'af': m / (m + 1), 'g': (3 * m - 1) / (2 * (m + 1)),
                               'b': (m / (m + 1)) ** 2}

        elif isinstance(Meth, str):

            if Meth == 'CDM':
                return Meth, {}
            elif Meth == 'CAA' or Meth == 'NWK':
                return 'NWK', {'g': 1 / 2, 'b': 1 / 4}  # If not specified run CAA by default
            elif Meth == 'LA':
                return 'NWK', {'g': 1 / 2, 'b': 1 / 6}
            elif Meth == 'NWK':
                return 'GEN', {'am': 0, 'af': 0, 'g': 1 / 2, 'b': 1 / 4}
            elif Meth == 'WIL':
                return 'WIL', {'t': 1.5}
            elif Meth == 'HHT':
                return 'GEN', {'am': 0, 'af': 0, 'g': 1 / 2, 'b': 1 / 4}
            elif Meth == 'WBZ':
                return 'GEN', {'am': 0, 'af': 0, 'g': 1 / 2, 'b': 1 / 4}
            elif Meth == 'GEN':
                m = 1
                return 'GEN', {'am': (2 * m - 1) / (m + 1), 'af': m / (m + 1), 'g': (3 * m - 1) / (2 * (m + 1)),
                               'b': (m / (m + 1)) ** 2}

        elif isinstance(Meth, list):

            if Meth[0] == 'NWK':

                if len(Meth) != 3: warnings.warn('Requiring 2 parameters for Newmark method')

                g = Meth[1]
                b = Meth[2]

                return 'GEN', {'am': 0, 'af': 0, 'g': g, 'b': b}

            elif Meth[0] == 'WIL':

                if len(Meth) != 2: warnings.warn('Requiring 1 parameters for Wilson\'s theta method')

                t = Meth[1]
                if t < 1:
                    warnings.warn('Theta should be larger or equal to one for Wilson\'s theta method')
                elif t < 1.37:
                    warnings.warn(
                        'Theta should be larger or equal to one for unconditional stability in Wilson\'s theta method')
                return 'WIL', {'t': t}

            elif Meth[0] == 'HHT':

                if len(Meth) == 2:
                    a = Meth[1]
                    g = (1 + 2 * a) / 2
                    b = (1 + a) ** 2 / 4

                elif len(Meth) == 4:

                    a = Meth[1]
                    g = Meth[2]
                    b = Meth[3]

                else:
                    warnings.warn('Requiring 3 parameters for HHT method')

                if a < 0 or a > 1 / 3: warnings.warn(
                    'Alpha should be between 0 and 1/3 for unconditional stability in HHT Method')

                return 'GEN', {'am': 0, 'af': a, 'g': g, 'b': b}

            elif Meth[0] == 'WBZ':

                if len(Meth) != 4: warnings.warn('Requiring 3 parameters for WBZ method')

                a = Meth[1]
                g = Meth[2]
                b = Meth[3]

                if a > 1 / 2: warnings.warn(
                    'Alpha should be smaller thann 1/2 for unconditional stability in WBZ Method')
                if g < (1 - 2 * a) / 2: warnings.warn(
                    'Gamma should be larger than (1-2a)/2 for unconditional stability in WBZ Method')
                if b < g / 2: warnings.warn('Beta should be larger than g/2 for unconditional stability in WBZ Method')

                return 'GEN', {'am': a, 'af': 0, 'g': g, 'b': b}

            elif Meth[0] == 'GEN':

                if len(Meth) != 2: warnings.warn('Requiring 1 parameters for Generalized Alpha method')

                m = Meth[1]

                if m < 0 or m > 1: warnings.warn('Mu should be between 0 and 1 for Generalized-alpha Method')

                return 'GEN', {'am': (2 * m - 1) / (m + 1), 'af': m / (m + 1), 'g': (3 * m - 1) / (2 * (m + 1)),
                               'b': (m / (m + 1)) ** 2}

        return None, None

    def impose_dyn_excitation(self, node, dof, U_app, dt):

        if 3 * node + dof not in self.dof_fix:
            warnings.warn('Excited DoF should be a fixed one')

        if not hasattr(self, 'dof_moving'):
            self.dof_moving = []
            self.disp_histories = []
            self.times = []

        self.dof_moving.append(3 * node + dof)
        self.disp_histories.append(U_app)
        self.times.append(dt)

        # Later, add a function to interpolate when different timesteps are used. 

    def solve_dyn_linear(self, T, dt, U0=None, V0=None, lmbda=None, Meth=None, filename='', dir_name=''):

        time_start = time.time()

        self.get_K_str0()
        self.get_M_str()
        self.get_C_str()

        if U0 is None:
            if np.linalg.norm(self.U) == 0:
                U0 = np.zeros(self.nb_dofs)
            else:
                U0 = deepcopy(self.U)

        if V0 is None: V0 = np.zeros(self.nb_dofs)

        if hasattr(self, 'times'):
            for timestep in self.times:
                if timestep != dt: warnings.warn('Unmatching timesteps between excitation and simulation')
            for i, disp in enumerate(self.disp_histories):
                if U0[self.dof_moving[i]] != disp[0]:
                    warnings.warn('Unmatching initial displacements')
        else:
            self.dof_moving = []
            self.disp_histories = []
            self.times = []

        Time = np.arange(0, T, dt, dtype=default)
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
        A_conv[:, 0] = sc.linalg.solve(self.M, loading[0] * self.P - self.C @ V_conv[:, 0] - self.K0 @ U_conv[:, 0])

        Meth, P = self.ask_method(Meth)

        if Meth == 'CDM':

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

                P_h_f = loading[i - 1] * self.P[self.dof_free] - a_ff @ U_conv[self.dof_free, i - 2] - a_fd @ U_conv[
                    self.dof_moving, i - 2] - b_ff @ U_conv[self.dof_free, i - 1] - b_fd @ U_conv[
                            self.dof_moving, i - 1]

                U_d = np.zeros(len(self.disp_histories))

                for j, disp in enumerate(self.disp_histories):
                    U_d[j] = disp[i]

                U_conv[self.dof_free, i] = np.linalg.solve(k_ff, P_h_f - k_fd @ U_d)
                U_conv[self.dof_moving, i] = U_d

                P_h_d = k_df @ U_conv[self.dof_free, i] + k_dd @ U_d + a_df @ U_conv[self.dof_free, i - 2] + a_dd @ \
                        U_conv[self.dof_moving, i - 2] + b_df @ U_conv[self.dof_free, i - 1] + b_dd @ U_conv[
                            self.dof_moving, i - 1]

                V_conv[self.dof_free, i] = (U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]) / (2 * dt)
                V_conv[self.dof_moving, i] = (U_conv[self.dof_moving, i] - U_conv[self.dof_moving, i - 1]) / (2 * dt)

                A_conv[self.dof_free, i] = (U_conv[self.dof_free, i] - 2 * U_conv[self.dof_free, i - 1] + U_conv[
                    self.dof_free, i - 2]) / (dt ** 2)
                A_conv[self.dof_moving, i] = (U_conv[self.dof_moving, i] - 2 * U_conv[self.dof_moving, i - 1] + U_conv[
                    self.dof_moving, i - 2]) / (dt ** 2)

                P_conv[self.dof_free, i] = P_h_f.copy()
                P_conv[self.dof_moving, i] = P_h_d.copy()

        elif Meth == 'NWK':

            A1 = self.M / (P['b'] * dt ** 2) + P['g'] * self.C / (P['b'] * dt)
            A2 = self.M / (P['b'] * dt) + (P['g'] / P['b'] - 1) * self.C
            A3 = (1 / (2 * P['b']) - 1) * self.M + dt * (P['g'] / (2 * P['b']) - 1) * self.C

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

                P_h_f = loading[i] * self.P[self.dof_free] + self.P_fixed[self.dof_free] + a1_ff @ U_conv[
                    self.dof_free, i - 1] + a2_ff @ V_conv[self.dof_free, i - 1] + a3_ff @ A_conv[
                            self.dof_free, i - 1] + a1_fd @ U_conv[self.dof_moving, i - 1] + a2_fd @ V_conv[
                            self.dof_moving, i - 1] + a3_fd @ A_conv[self.dof_moving, i - 1]

                for j, disp in enumerate(self.disp_histories):
                    U_conv[self.dof_moving[j], i] = disp[i]
                    V_conv[self.dof_moving[j], i] = (U_conv[self.dof_moving[j], i] - U_conv[
                        self.dof_moving[j], i - 1]) / dt
                    A_conv[self.dof_moving[j], i] = (V_conv[self.dof_moving[j], i] - V_conv[
                        self.dof_moving[j], i - 1]) / dt

                U_conv[self.dof_free, i] = sc.linalg.solve(k_ff, P_h_f - k_fd @ U_conv[self.dof_moving, i])

                V_conv[self.dof_free, i] = (P['g'] / (P['b'] * dt)) * (
                        U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]) + (1 - P['g'] / P['b']) * V_conv[
                                               self.dof_free, i - 1] + dt * (1 - P['g'] / (2 * P['b'])) * A_conv[
                                               self.dof_free, i - 1]
                A_conv[self.dof_free, i] = (1 / (P['b'] * dt ** 2)) * (
                        U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]) - V_conv[self.dof_free, i - 1] / (
                                                   P['b'] * dt) - (1 / (2 * P['b']) - 1) * A_conv[
                                               self.dof_free, i - 1]

                P_conv[self.dof_free, i] = P_h_f.copy()
                P_conv[self.dof_moving, i] = k_df @ U_conv[self.dof_free, i] + k_dd @ U_conv[self.dof_moving, i]

        elif Meth == 'WIL':

            A1 = 6 / (P['t'] * dt) * self.M + 3 * self.C
            A2 = 3 * self.M + P['t'] * dt / 2 * self.C

            K_h = self.K0 + 6 / (P['t'] * dt) ** 2 * self.M + 3 / (P['t'] * dt) * self.C

            loading = np.append(loading, loading[-1])

            for i in np.arange(1, nb_steps):
                dp_h = ((P['t'] - 1) * (loading[i + 1] - loading[i]) + loading[i] - loading[i - 1]) * self.P

                dp_h += A1 @ V_conv[:, i - 1] + A2 @ A_conv[:, i - 1]

                d_Uh = (sc.linalg.solve(K_h, dp_h))

                d_A = (6 / (P['t'] * dt) ** 2 * d_Uh - 6 / (P['t'] * dt) * V_conv[:, i - 1] - 3 * A_conv[:, i - 1]) / (
                    P['t'])

                d_V = dt * A_conv[:, i - 1] + dt / 2 * d_A
                d_U = dt * V_conv[:, i - 1] + (dt ** 2) / 2 * A_conv[:, i - 1] + (dt ** 2) / 6 * d_A

                U_conv[self.dof_free, i] = (U_conv[:, i - 1] + d_U)[self.dof_free]
                V_conv[self.dof_free, i] = (V_conv[:, i - 1] + d_V)[self.dof_free]
                A_conv[self.dof_free, i] = (A_conv[:, i - 1] + d_A)[self.dof_free]

        elif Meth == 'GEN':

            am = 0
            b = P['b']
            g = P['g']
            af = P['af']

            A1 = (1 - am) / (b * dt ** 2) * self.M + g * (1 - af) / (b * dt) * self.C
            A2 = (1 - am) / (b * dt) * self.M + (g * (1 - af) / b - 1) * self.C
            A3 = ((1 - am) / (2 * b) - 1) * self.M + dt * (1 - af) * (g / (2 * b) - 1) * self.C

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

                P_h_f = loading[i] * self.P[self.dof_free] + self.P_fixed[self.dof_free] + a1_ff @ U_conv[
                    self.dof_free, i - 1] + a2_ff @ V_conv[self.dof_free, i - 1] + a3_ff @ A_conv[
                            self.dof_free, i - 1] + a1_fd @ U_conv[self.dof_moving, i - 1] + a2_fd @ V_conv[
                            self.dof_moving, i - 1] + a3_fd @ A_conv[self.dof_moving, i - 1] - af * (
                                self.K0[np.ix_(self.dof_free, self.dof_free)] @ U_conv[self.dof_free, i - 1] +
                                self.K0[
                                    np.ix_(self.dof_free, self.dof_moving)] @ U_conv[self.dof_moving, i - 1])

                for j, disp in enumerate(self.disp_histories):
                    U_conv[self.dof_moving[j], i] = disp[i]
                    # V_conv[self.dof_moving[j],i] = (U_conv[self.dof_moving[j],i] - U_conv[self.dof_moving[j],i-1]) / dt
                    # A_conv[self.dof_moving[j],i] = (V_conv[self.dof_moving[j],i] - V_conv[self.dof_moving[j],i-1]) / dt

                U_conv[self.dof_free, i] = sc.linalg.solve(k_ff, P_h_f - k_fd @ U_conv[self.dof_moving, i])

                V_conv[:, i][self.dof_free] = (
                        P['g'] / (P['b'] * dt) * (U_conv[:, i] - U_conv[:, i - 1]) + (1 - P['g'] / P['b']) * V_conv[
                                                                                                             :,
                                                                                                             i - 1] + dt * (
                                1 - P['g'] / (2 * P['b'])) * A_conv[:, i - 1])[self.dof_free]
                A_conv[:, i][self.dof_free] = (
                        1 / (P['b'] * dt ** 2) * (U_conv[:, i] - U_conv[:, i - 1]) - 1 / (dt * P['b']) * V_conv[:,
                                                                                                         i - 1] - (
                                1 / (2 * P['b']) - 1) * A_conv[:, i - 1])[self.dof_free]
                P_conv[self.dof_free, i] = P_h_f.copy()
                P_conv[self.dof_moving, i] = k_df @ U_conv[self.dof_free, i] + k_dd @ U_conv[self.dof_moving, i]



        elif Meth is None:
            print('Method does not exist')

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f'Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file')

        Params = []
        for key, value in P.items():
            Params.append(f'{key}={np.around(value, 2)}')

        Params = "_".join(Params)

        filename = filename + '_' + Meth + '_' + Params + '.h5'
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, 'w') as hf:

            hf.create_dataset('U_conv', data=U_conv)
            hf.create_dataset('V_conv', data=V_conv)
            hf.create_dataset('A_conv', data=A_conv)
            hf.create_dataset('P_ref', data=self.P)
            hf.create_dataset('P_conv', data=P_conv)
            hf.create_dataset('Load_Multiplier', data=loading)
            hf.create_dataset('Time', data=Time)
            hf.create_dataset('Last_conv', data=nb_steps - 1)

            hf.attrs['Descr'] = 'Results of the' + Meth + 'simulation'
            hf.attrs['Method'] = Meth

    def solve_dyn_nonlinear(self, T, dt, U0=None, V0=None, lmbda=None, Meth=None, filename='', dir_name=''):

        time_start = time.time()

        if U0 is None:
            if np.linalg.norm(self.U) == 0:
                U0 = np.zeros(self.nb_dofs)
            else:
                U0 = deepcopy(self.U)

        if V0 is None: V0 = np.zeros(self.nb_dofs)

        Time = np.arange(0, T, dt, dtype=default)
        Time = np.append(Time, T)
        nb_steps = len(Time)

        loading = np.zeros(nb_steps)

        if callable(lmbda):
            for i, t in enumerate(Time):
                loading[i] = lmbda(t)
        elif isinstance(lmbda, list):
            loading = lmbda
            if len(loading) > nb_steps:
                print('Truncate')
                loading = loading[:nb_steps]
            elif len(loading) < nb_steps:
                print('Add 0')
                missing = nb_steps - len(loading)
                for i in range(missing):
                    loading.append(0)

        self.get_P_r()
        self.get_K_str0()
        self.get_M_str()
        self.get_C_str()

        if hasattr(self, 'times'):
            for timestep in self.times:
                if timestep != dt: warnings.warn('Unmatching timesteps between excitation and simulation')
            for i, disp in enumerate(self.disp_histories):
                if U0[self.dof_moving[i]] != disp[0]:
                    warnings.warn('Unmatching initial displacements')

        else:
            self.dof_moving = []
            self.disp_histories = []
            self.times = []

        U_conv = np.zeros((self.nb_dofs, nb_steps), dtype=default)
        V_conv = np.zeros((self.nb_dofs, nb_steps), dtype=default)
        A_conv = np.zeros((self.nb_dofs, nb_steps), dtype=default)
        F_conv = np.zeros((self.nb_dofs, nb_steps), dtype=default)

        U_conv[:, 0] = deepcopy(U0)
        V_conv[:, 0] = deepcopy(V0)
        F_conv[:, 0] = deepcopy(self.P_r)

        self.commit()

        last_sec = 0

        Meth, P = self.ask_method(Meth)

        if Meth == 'CDM':

            self.U = U_conv[:, 0].copy()
            self.get_P_r()
            F_conv[:, 0] = self.P_r.copy()

            A_conv[:, 0] = sc.linalg.solve(self.M,
                                           loading[0] * self.P + self.P_fixed - self.C @ V_conv[:, 0] - F_conv[:, 0])

            U_conv[:, -1] = U_conv[:, 0] - dt * V_conv[:, 0] + dt ** 2 / 2 * A_conv[:, 0]

            K_h = 1 / (dt ** 2) * self.M + 1 / (2 * dt) * self.C
            A = 1 / (dt ** 2) * self.M - 1 / (2 * dt) * self.C
            B = - 2 / (dt ** 2) * self.M

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

                if self.stiff_type[:3] == 'TAN':
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

                P_h_f = loading[i] * self.P[self.dof_free] + self.P_fixed[self.dof_free] - a_ff @ U_conv[
                    self.dof_free, i - 2] - a_fd @ U_conv[self.dof_moving, i - 2] - b_ff @ U_conv[
                            self.dof_free, i - 1] - b_fd @ U_conv[self.dof_moving, i - 1] - F_conv[self.dof_free, i - 1]

                U_d = np.zeros(len(self.disp_histories))

                for j, disp in enumerate(self.disp_histories):
                    U_d[j] = disp[i]

                U_conv[self.dof_free, i] = np.linalg.solve(k_ff, P_h_f - k_fd @ U_d)
                U_conv[self.dof_moving, i] = U_d

                P_h_d = k_df @ U_conv[self.dof_free, i] + k_dd @ U_d + a_df @ U_conv[self.dof_free, i - 2] + a_dd @ \
                        U_conv[self.dof_moving, i - 2] + b_df @ U_conv[self.dof_free, i - 1] + b_dd @ U_conv[
                            self.dof_moving, i - 1]

                V_conv[self.dof_free, i] = (U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]) / (2 * dt)
                V_conv[self.dof_moving, i] = (U_conv[self.dof_moving, i] - U_conv[self.dof_moving, i - 1]) / (2 * dt)

                A_conv[self.dof_free, i] = (U_conv[self.dof_free, i] - 2 * U_conv[self.dof_free, i - 1] + U_conv[
                    self.dof_free, i - 2]) / (dt ** 2)
                A_conv[self.dof_moving, i] = (U_conv[self.dof_moving, i] - 2 * U_conv[self.dof_moving, i - 1] + U_conv[
                    self.dof_moving, i - 2]) / (dt ** 2)

                if i * dt >= last_sec:
                    print(f'reached {np.around(last_sec, 3)} seconds out of {int(Time[-1])} seconds')
                    last_sec += .1

                last_conv = i

                self.commit()




        elif Meth == 'NWK':

            tol = 1
            singular_steps = []
            # tol = np.max(self.M) / np.max(self.K) * 10
            print(f'Tolerance is {tol}')

            self.U = deepcopy(U_conv[:, 0])

            g = P['g']
            b = P['b']

            print(g)
            print(b)
            A_conv[:, 0] = sc.linalg.solve(self.M, loading[0] * self.P + self.P_fixed - self.C @ V0 - F_conv[:, 0])

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
                    V_conv[self.dof_moving[j], i] = a4 * (
                            U_conv[self.dof_moving[j], i] - U_conv[self.dof_moving[j], i - 1]) + a5 * V_conv[
                                                        self.dof_moving[j], i - 1] + a6 * A_conv[
                                                        self.dof_moving[j], i - 1]
                    A_conv[self.dof_moving[j], i] = a1 * (
                            U_conv[self.dof_moving[j], i] - U_conv[self.dof_moving[j], i - 1]) - a2 * V_conv[
                                                        self.dof_moving[j], i - 1] - a3 * A_conv[
                                                        self.dof_moving[j], i - 1]

                P_h_f = loading[i] * self.P[self.dof_free] + self.P_fixed[self.dof_free] + A1[
                    np.ix_(self.dof_free, self.dof_free)] @ U_conv[self.dof_free, i - 1] + A1[
                            np.ix_(self.dof_free, self.dof_moving)] @ U_conv[self.dof_moving, i - 1] + A2[
                            np.ix_(self.dof_free, self.dof_free)] @ V_conv[self.dof_free, i - 1] + A2[
                            np.ix_(self.dof_free, self.dof_moving)] @ V_conv[self.dof_moving, i - 1] + A3[
                            np.ix_(self.dof_free, self.dof_free)] @ A_conv[self.dof_free, i - 1] + A3[
                            np.ix_(self.dof_free, self.dof_moving)] @ A_conv[self.dof_moving, i - 1]
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

                    R = P_h_f - self.P_r[self.dof_free] - A1[np.ix_(self.dof_free, self.dof_free)] @ self.U[
                        self.dof_free] - A1[np.ix_(self.dof_free, self.dof_moving)] @ self.U[self.dof_moving]
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
                    print(f'Step {no_conv} did not converge')
                    break

                dU_step = U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]
                V_conv[self.dof_free, i] = a4 * dU_step + a5 * V_conv[self.dof_free, i - 1] + a6 * A_conv[
                    self.dof_free, i - 1]
                A_conv[self.dof_free, i] = a1 * dU_step - a2 * V_conv[self.dof_free, i - 1] - a3 * A_conv[
                    self.dof_free, i - 1]

                if self.stiff_type[:3] == 'TAN':
                    self.get_C_str()
                    A1 = (1 / (b * dt ** 2)) * self.M + (g / (b * dt)) * self.C
                    A2 = (1 / (b * dt)) * self.M + (g / b - 1) * self.C
                    A3 = (1 / (2 * b) - 1) * self.M + dt * (g / (2 * b) - 1) * self.C

                # if np.min(np.real(np.sqrt(eigvals))) <=  1:
                #     singular_steps.append(i)
                #     print(f'Step{i} is singular - {np.linalg.cond(self.C[np.ix_(self.dof_free, self.dof_free)])}')
                #     # print(f'Step{i} is singular - {np.linalg.cond(self.K[np.ix_(self.dof_free, self.dof_free)])}')

                if i * dt >= last_sec:
                    print(f'reached {np.around(last_sec, 3)} seconds out of {int(Time[-1])} seconds')
                    self.plot_structure(scale=1, plot_forces=False, plot_cf=False, plot_supp=False,
                                        lims=[[-6., 6.], [-1.2, 6.5]])
                    last_sec += .1


        elif Meth == 'WIL':

            pass

        elif Meth == 'GEN':

            tol = 1e-3
            singular_steps = []
            # tol = np.max(self.M) / np.max(self.K) * 10
            print(f'Tolerance is {tol}')

            self.U = deepcopy(U_conv[:, 0])

            g = P['g']
            b = P['b']
            af = P['af']
            am = P['am']

            A_conv[:, 0] = sc.linalg.solve(self.M, loading[0] * self.P + self.P_fixed - self.C @ V0 - F_conv[:, 0])

            A1 = ((1 - am) / (b * dt ** 2)) * self.M + (g * (1 - af) / (b * dt)) * self.C
            A2 = ((1 - am) / (b * dt)) * self.M + (g * (1 - af) / b - 1) * self.C
            A3 = ((1 - am) / (2 * b) - 1) * self.M + dt * (1 - af) * (g / (2 * b) - 1) * self.C

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

                P_h_f = loading[i] * self.P[self.dof_free] + self.P_fixed[self.dof_free] + A1[
                    np.ix_(self.dof_free, self.dof_free)] @ U_conv[self.dof_free, i - 1] + A1[
                            np.ix_(self.dof_free, self.dof_moving)] @ U_conv[self.dof_moving, i - 1] + A2[
                            np.ix_(self.dof_free, self.dof_free)] @ V_conv[self.dof_free, i - 1] + A2[
                            np.ix_(self.dof_free, self.dof_moving)] @ V_conv[self.dof_moving, i - 1] + A3[
                            np.ix_(self.dof_free, self.dof_free)] @ A_conv[self.dof_free, i - 1] + A3[
                            np.ix_(self.dof_free, self.dof_moving)] @ A_conv[self.dof_moving, i - 1] - af * F_conv[
                            self.dof_free, i - 1]

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

                    R = P_h_f - self.P_r[self.dof_free] - A1[np.ix_(self.dof_free, self.dof_free)] @ self.U[
                        self.dof_free] - A1[np.ix_(self.dof_free, self.dof_moving)] @ self.U[self.dof_moving]
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
                    print(f'Step {no_conv} did not converge')
                    break

                dU_step = U_conv[self.dof_free, i] - U_conv[self.dof_free, i - 1]
                V_conv[self.dof_free, i] = a4 * dU_step + a5 * V_conv[self.dof_free, i - 1] + a6 * A_conv[
                    self.dof_free, i - 1]
                A_conv[self.dof_free, i] = a1 * dU_step - a2 * V_conv[self.dof_free, i - 1] - a3 * A_conv[
                    self.dof_free, i - 1]

                if self.stiff_type[:3] == 'TAN':
                    self.get_C_str()
                    A1 = ((1 - am) / (b * dt ** 2)) * self.M + (g * (1 - af) / (b * dt)) * self.C
                    A2 = ((1 - am) / (b * dt)) * self.M + (g * (1 - af) / b - 1) * self.C
                    A3 = ((1 - am) / (2 * b) - 1) * self.M + dt * (1 - af) * (g / (2 * b) - 1) * self.C

                # if np.min(np.real(np.sqrt(eigvals))) <=  1:
                #     singular_steps.append(i)
                #     print(f'Step{i} is singular - {np.linalg.cond(self.C[np.ix_(self.dof_free, self.dof_free)])}')
                #     # print(f'Step{i} is singular - {np.linalg.cond(self.K[np.ix_(self.dof_free, self.dof_free)])}')

                if i * dt >= last_sec:
                    print(f'reached {np.around(last_sec, 3)} seconds out of {int(Time[-1])} seconds')
                    self.plot_structure(scale=1, plot_forces=False, plot_cf=False, plot_supp=False,
                                        lims=[[-6., 6.], [-1.2, 6.5]])
                    last_sec += .1


        elif Meth is None:
            print('Method does not exist')

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f'Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file')

        Params = []
        for key, value in P.items():
            Params.append(f'{key}={np.around(value, 2)}')

        Params = "_".join(Params)

        filename = filename + '_' + Meth + '_' + Params + '.h5'
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, 'w') as hf:

            hf.create_dataset('U_conv', data=U_conv)
            hf.create_dataset('V_conv', data=V_conv)
            hf.create_dataset('A_conv', data=A_conv)
            hf.create_dataset('F_conv', data=F_conv)
            hf.create_dataset('P_ref', data=self.P)
            hf.create_dataset('Load_Multiplier', data=loading)
            hf.create_dataset('Time', data=Time)
            hf.create_dataset('Last_conv', data=last_conv)
            # hf.create_dataset('Singular_steps', data=singular_steps)

            hf.attrs['Descr'] = f'Results of the' + Meth + 'simulation'
            hf.attrs['Method'] = Meth

    def save_structure(self, filename):

        import pickle

        with open(filename + '.pkl', 'wb') as file:
            pickle.dump(self, file)

    def set_lin_geom(self, lin_geom=True):

        for cf in self.list_cfs:
            cf.set_lin_geom(lin_geom)

        for fe in self.list_fes:
            fe.lin_geom = lin_geom

    def solve_modal(self, modes=None, no_inertia=False, filename='Results_Modal', dir_name='', save=True,
                    initial=False):

        time_start = time.time()

        self.get_P_r()
        self.get_M_str(no_inertia=no_inertia)

        if not initial:
            if not hasattr(self, 'K'): self.get_K_str()

            if modes is None:
                # print('HEllo')
                # self.K = np.around(self.K,6)
                # self.M = np.around(self.M,8)
                omega, phi = sc.linalg.eig(self.K[np.ix_(self.dof_free, self.dof_free)],
                                           self.M[np.ix_(self.dof_free, self.dof_free)])

            elif isinstance(modes, int):
                if np.linalg.det(self.M) == 0: warnings.warn(
                    'Might need to use linalg.eig if the matrix M is non-invertible')
                omega, phi = sc.sparse.linalg.eigsh(self.K[np.ix_(self.dof_free, self.dof_free)], modes,
                                                    self.M[np.ix_(self.dof_free, self.dof_free)], which='SM')

            else:
                warnings.warn("Required modes should be either int or None")
        else:
            self.get_K_str0()
            if modes is None:
                omega, phi = sc.linalg.eigh(self.K0[np.ix_(self.dof_free, self.dof_free)],
                                            self.M[np.ix_(self.dof_free, self.dof_free)])
            elif isinstance(modes, int):
                if np.linalg.det(self.M) == 0: warnings.warn(
                    'Might need to use linalg.eig if the matrix M is non-invertible')
                omega, phi = sc.sparse.linalg.eigsh(self.K0[np.ix_(self.dof_free, self.dof_free)], modes,
                                                    self.M[np.ix_(self.dof_free, self.dof_free)], which='SM')

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
            print('Simulation done... writing results to file')

            filename = filename + '.h5'
            file_path = os.path.join(dir_name, filename)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('eig_vals', data=self.eig_vals)
                hf.create_dataset('eig_modes', data=self.eig_modes)

                hf.attrs['Simulation_Time'] = total_time

    def detect_interfaces(self):

        self.interf_counter = 0

        def detect_overlap(pair1, pair2, factor=None):

            if not factor == 1:
                pair1[0][1] *= factor
                pair1[1][1] *= factor
                pair2[0][1] *= factor
                pair2[1][1] *= factor

            sorted1 = sorted(pair1, key=lambda point: point[0] + point[1])
            sorted2 = sorted(pair2, key=lambda point: point[0] + point[1])

            if np.sum(sorted1[0]) - np.sum(sorted2[1]) >= -1e-8 or np.sum(sorted2[0]) - np.sum(sorted1[1]) >= -1e-8:
                return False, None

            else:
                edge1 = sorted([sorted1[0], sorted2[0]], key=lambda point: point[0] + point[1])
                edge2 = sorted([sorted1[1], sorted2[1]], key=lambda point: point[0] + point[1])

                if not factor == 1:
                    edge1[0][1] /= factor
                    edge1[1][1] /= factor
                    edge2[0][1] /= factor
                    edge2[1][1] /= factor

                return True, [edge1[1], edge2[0]]

        def detect_interface_2blocks(cand, anta):

            interfaces = []

            triplets_cand = cand.compute_triplets()
            triplets_anta = anta.compute_triplets()

            for triplet1 in triplets_cand:
                for triplet2 in triplets_anta:

                    if np.all(np.isclose(triplet1['ABC'], triplet2['ABC'], rtol=1e-8)):

                        # Handle the case where x+y = constant
                        if triplet1['ABC'][0] == triplet1['ABC'][1]:
                            factor = 2
                        else:
                            factor = 1
                        overlap, nodes = detect_overlap(triplet1['Vertices'], triplet2['Vertices'], factor=factor)

                        if overlap:
                            interface = {}
                            unit_vector = (nodes[1] - nodes[0]) / np.linalg.norm(nodes[1] - nodes[0])
                            normal_vector = np.array([[0, -1], [1, 0]]) @ unit_vector
                            if np.dot(cand.ref_point - nodes[0], normal_vector) > 0:
                                interface['Block A'] = cand
                                interface['Block B'] = anta
                            else:
                                interface['Block A'] = anta
                                interface['Block B'] = cand
                            interface['x_e1'] = nodes[0]
                            interface['x_e2'] = nodes[1]

                            interfaces.append(interface)

            if len(interfaces) == 0:
                return False, None
            else:
                return True, interfaces

        def distance(a, b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        interfaces = []

        for i, cand in enumerate(self.list_blocks):

            for j, anta in enumerate(self.list_blocks[i + 1:]):

                # Check if blocks have same dofs
                if cand.connect == anta.connect: continue
                # Check if influence circles intersect 
                if distance(cand.circle_center, anta.circle_center) >= (
                        cand.circle_radius + anta.circle_radius) * 1.01: continue

                contact, interface = detect_interface_2blocks(cand, anta)
                self.interf_counter += 1

                if not contact: continue

                interfaces.extend(interface)

        print(f'{len(interfaces)} interface{"s" if len(interfaces) != 1 else ""} detected')

        return interfaces

    def plot_stiffness(self, save=None):

        E = []
        vertices = []

        for j, CF in enumerate(self.list_cfs):

            for i, CP in enumerate(CF.cps):
                E.append(np.around(CP.sp1.law.stiff['E'], 3))
                E.append(np.around(CP.sp2.law.stiff['E'], 3))
                vertices.append(CP.vertices_fibA)
                vertices.append(CP.vertices_fibB)

        from matplotlib.colors import Normalize
        from matplotlib import cm

        def normalize(smax, smin):

            if (smax - smin) == 0 and smax < 0:
                return Normalize(vmin=1.1 * smin / 1e9, vmax=0.9 * smax / 1e9, clip=False)
            elif (smax - smin) == 0 and smax == 0:
                return Normalize(vmin=-1e-6, vmax=1e-6, clip=False)
            elif (smax - smin) == 0:
                return Normalize(vmin=0.9 * smin / 1e9, vmax=1.1 * smax / 1e9, clip=False)
            else:
                return Normalize(vmin=smin / 1e9, vmax=smax / 1e9, clip=False)

        def plot(stiff, vertex):
            smax = np.max(stiff)
            smin = np.min(stiff)

            plt.axis('equal')
            plt.axis('off')
            plt.title(f"Axial stiffness [GPa]")

            norm = normalize(smax, smin)
            cmap = cm.get_cmap('coolwarm', 200)

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

        if save is not None: plt.savefig(save)

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
                            eps = np.append(eps, np.around(CP.sp1.law.strain['e'], 12))
                            # print(np.around(CP.sp1.law.strain['e'],12))
                            sigma = np.append(sigma, np.around(CP.sp1.law.stress['s'], 12))
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
                            tau.append(np.around(CP.sp1.law.stress['t'], 12))
                            tau.append(np.around(CP.sp2.law.stress['t'], 12))
                            sigma.append(np.around(CP.sp1.law.stress['s'], 12))
                            sigma.append(np.around(CP.sp2.law.stress['s'], 12))
                            vertices.append(CP.vertices_fibA)
                            vertices.append(CP.vertices_fibB)

        from matplotlib.colors import Normalize
        from matplotlib import cm

        def normalize(smax, smin):

            if (smax - smin) == 0 and smax < 0:
                return Normalize(vmin=1.1 * smin / 1e6, vmax=0.9 * smax / 1e6, clip=False)
            elif (smax - smin) == 0 and smax == 0:
                return Normalize(vmin=-1e-6, vmax=1e-6, clip=False)
            elif (smax - smin) == 0:
                return Normalize(vmin=0.9 * smin / 1e6, vmax=1.1 * smax / 1e6, clip=False)
            else:
                return Normalize(vmin=smin / 1e6, vmax=smax / 1e6, clip=False)

        def plot(stress, vertex, name_stress=None):
            smax = np.max(stress)
            smin = np.min(stress)

            print(f"Maximal {'axial' if name_stress == 'sigma' else 'shear'} stress is {np.around(smax / 1e6, 3)} MPa")
            print(f"Minimum {'axial' if name_stress == 'sigma' else 'shear'} stress is {np.around(smin / 1e6, 3)} MPa")
            # Plot sigmas 

            plt.axis('equal')
            plt.axis('off')
            plt.title(f"{'Axial' if name_stress == 'sigma' else 'Shear'} stresses [MPa]")

            norm = normalize(smax, smin)
            cmap = cm.get_cmap('viridis', 200)

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

            cax = divider.append_axes("right", size="10%", pad=.2)
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

        plt.figure()

        plt.subplot(2, 1, 1)
        plot(sigma, vertices, name_stress='sigma')
        plt.subplot(2, 1, 2)
        plot(tau, vertices, name_stress='tau')

        if save is not None: plt.savefig(save)

    def plot_modes(self, modes=None, scale=1, save=False, lims=None, folder=None, show=True):

        if not hasattr(self, 'eig_modes'): warnings.warn('Eigen modes were not determined yet')

        if modes is None:
            modes = self.nb_dof_free

        if len(self.eig_vals) < modes: warnings.warn('Asking for too many modes, fewer were computed')

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

            plt.axis('equal')
            plt.axis('off')

            self.plot_def_structure(scale=scale, plot_cf=False, plot_forces=False, plot_supp=False)
            self.plot_def_structure(scale=0, plot_cf=False, plot_forces=False, plot_supp=False, lighter=True)

            if lims is not None:
                plt.xlim(lims[0][0], lims[0][1])
                plt.ylim(lims[1][0], lims[1][1])

            w = np.around(self.eig_vals[i], 3)
            f = np.around(self.eig_vals[i] / (2 * np.pi), 3)
            if not w == 0:
                T = np.around(2 * np.pi / w, 3)
            else:
                T = float('inf')
            plt.title(fr'$\omega_{{{i + 1}}} = {w}$ rad/s - $T_{{{i + 1}}} = {T}$ s - $f_{{{i + 1}}} = {f}$ ')
            if save:
                if folder is not None:
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    plt.savefig(folder + f'/Mode_{i + 1}.eps')
                else:
                    plt.savefig(f'Mode_{i + 1}.eps')

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
        plt.axis('off')

        self.plot_def_structure(scale=scale, plot_cf=plot_cf, plot_forces=plot_forces, plot_supp=plot_supp)

        if lims is not None:
            plt.xlim((x0, x1))
            plt.ylim((y0, y1))

        if save is not None: plt.savefig(save)

        if not show:
            # print('Closing figure...')
            plt.close()
        else:
            plt.show()

    def plot_def_structure(self, scale=0, plot_cf=True, plot_forces=True, plot_supp=True, lighter=False):

        # self.get_P_r()

        for bl in self.list_blocks:
            bl.disps = self.U[bl.dofs]
            bl.plot_block(scale=scale, lighter=lighter)

        for FE in self.list_fes:
            if scale == 0:
                FE.PlotUndefShapeElem()
            else:
                defs = self.U[FE.dofs]
                FE.PlotDefShapeElem(defs, scale=scale)

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

                    start = self.list_nodes[node_id] + scale * self.U[
                        3 * node_id * np.ones(2, dtype=int) + np.array([0, 1], dtype=int)]
                    arr_len = .3

                    if dof == 0:
                        end = arr_len * np.array([1, 0]) * np.sign(self.P[i])
                        plt.arrow(start[0], start[1], end[0], end[1], head_width=0.05, head_length=0.075, fc='green',
                                  ec='green')
                    elif dof == 1:
                        end = arr_len * np.array([0, 1]) * np.sign(self.P[i])
                        plt.arrow(start[0], start[1], end[0], end[1], head_width=0.05, head_length=0.075, fc='green',
                                  ec='green')
                    else:
                        if np.sign(self.P[i]) == 1:
                            plt.plot(start[0], start[1], marker='o', markerfacecolor='None', markeredgecolor='green',
                                     markersize=10)
                            plt.plot(start[0], start[1], marker='.', markerfacecolor='green', markeredgecolor='green',
                                     markersize=5)
                        else:
                            plt.plot(start[0], start[1], marker='o', markerfacecolor='None', markeredgecolor='green',
                                     markersize=10)
                            plt.plot(start[0], start[1], marker='x', markerfacecolor='None', markeredgecolor='green',
                                     markersize=10)

                if self.P_fixed[i] != 0:

                    node_id = int(i / 3)
                    dof = i % 3

                    start = self.list_nodes[node_id] + scale * self.U[
                        3 * node_id * np.ones(2, dtype=int) + np.array([0, 1], dtype=int)]
                    arr_len = .3

                    if dof == 0:
                        end = arr_len * np.array([1, 0]) * np.sign(self.P_fixed[i])
                        plt.arrow(start[0], start[1], end[0], end[1], head_width=0.05, head_length=0.075, fc='red',
                                  ec='red')
                    elif dof == 1:
                        end = arr_len * np.array([0, 1]) * np.sign(self.P_fixed[i])
                        plt.arrow(start[0], start[1], end[0], end[1], head_width=0.05, head_length=0.075, fc='red',
                                  ec='red')
                    else:
                        if np.sign(self.P_fixed[i]) == 1:
                            plt.plot(start[0], start[1], marker='o', markerfacecolor='None', markeredgecolor='red',
                                     markersize=10)
                            plt.plot(start[0], start[1], marker='.', markerfacecolor='red', markeredgecolor='red',
                                     markersize=5)
                        else:
                            plt.plot(start[0], start[1], marker='o', markerfacecolor='None', markeredgecolor='red',
                                     markersize=10)
                            plt.plot(start[0], start[1], marker='x', markerfacecolor='None', markeredgecolor='red',
                                     markersize=10)

        if plot_supp:

            for fix in self.dof_fix:

                node_id = int(fix / 3)
                dof = fix % 3

                node = self.list_nodes[node_id] + scale * self.U[
                    3 * node_id * np.ones(2, dtype=int) + np.array([0, 1], dtype=int)]

                import matplotlib as mpl

                if dof == 0:
                    mark = mpl.markers.MarkerStyle(marker=5)
                elif dof == 1:
                    mark = mpl.markers.MarkerStyle(marker=6)
                else:
                    mark = mpl.markers.MarkerStyle(marker="x")

                plt.plot(node[0], node[1], marker=mark, color='blue', markersize=8)

    def plot_stress_profile(self, cf_index=0, save=None):

        stresses = []
        x = []
        counter = 0
        for cp in self.list_cfs[cf_index].cps:
            counter += 1
            if not cp.to_ommit():
                stresses.append(cp.sp1.law.stress['s'] / 1e6)
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
        plt.bar(x, stresses, label='HybriDFEM', facecolor='white', edgecolor='blue', linewidth=1, width=50 / counter)
        # print(x)
        # print(stresses)
        # plt.plot(str,y_sigma,label='Analytical',color='red')
        # elif stress=='tau':    
        #     plt.plot(x2*100,y_tau/1e6,label='Analytical',color='red')
        plt.legend(fontsize=12)
        plt.ylabel(r'Stress [MPa]')
        plt.xlabel(r'Height [cm]')
        plt.grid(True, linestyle='--', linewidth=0.3)

        if save:
            plt.savefig(save)
