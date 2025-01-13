import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QFormLayout, QPushButton, QMessageBox, QFileDialog, QHBoxLayout, QGridLayout, QTextEdit, QListWidget
)


class JsonCreatorWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Create JSON Input File")
        self.setGeometry(200, 200, 1000, 600)

        # Main layout
        layout = QVBoxLayout(self)

        # Grid layout for input fields
        self.grid_layout = QGridLayout()

        # Node input fields
        self.node_id_input = QLineEdit()
        self.node_coordinates_input = QLineEdit()
        self.add_node_button = QPushButton("Add Node")
        self.add_node_button.clicked.connect(self.add_node)
        self.node_list = QListWidget()

        self.grid_layout.addWidget(QLabel("Node ID:"), 0, 0)
        self.grid_layout.addWidget(self.node_id_input, 0, 1)
        self.grid_layout.addWidget(QLabel("Coordinates (x, y):"), 1, 0)
        self.grid_layout.addWidget(self.node_coordinates_input, 1, 1)
        self.grid_layout.addWidget(self.add_node_button, 2, 1)
        self.grid_layout.addWidget(QLabel("Created Nodes:"), 3, 0, 1, 2)
        self.grid_layout.addWidget(self.node_list, 4, 0, 1, 2)

        # Element input fields
        self.element_id_input = QLineEdit()
        self.element_type_input = QLineEdit()
        self.element_connectivity_input = QLineEdit()
        self.add_element_button = QPushButton("Add Element")
        self.add_element_button.clicked.connect(self.add_element)
        self.element_list = QListWidget()

        self.grid_layout.addWidget(QLabel("Element ID:"), 0, 2)
        self.grid_layout.addWidget(self.element_id_input, 0, 3)
        self.grid_layout.addWidget(QLabel("Type:"), 1, 2)
        self.grid_layout.addWidget(self.element_type_input, 1, 3)
        self.grid_layout.addWidget(QLabel("Connectivity:"), 2, 2)
        self.grid_layout.addWidget(self.element_connectivity_input, 2, 3)
        self.grid_layout.addWidget(self.add_element_button, 3, 3)
        self.grid_layout.addWidget(QLabel("Created Elements:"), 4, 2, 1, 2)
        self.grid_layout.addWidget(self.element_list, 5, 2, 1, 2)

        # Material input fields
        self.material_name_input = QLineEdit()
        self.material_E_input = QLineEdit()
        self.material_nu_input = QLineEdit()
        self.material_density_input = QLineEdit()
        self.add_material_button = QPushButton("Add Material")
        self.add_material_button.clicked.connect(self.add_material)
        self.material_list = QListWidget()

        self.grid_layout.addWidget(QLabel("Material Name:"), 0, 4)
        self.grid_layout.addWidget(self.material_name_input, 0, 5)
        self.grid_layout.addWidget(QLabel("Young's Modulus:"), 1, 4)
        self.grid_layout.addWidget(self.material_E_input, 1, 5)
        self.grid_layout.addWidget(QLabel("Poisson's Ratio:"), 2, 4)
        self.grid_layout.addWidget(self.material_nu_input, 2, 5)
        self.grid_layout.addWidget(QLabel("Density:"), 3, 4)
        self.grid_layout.addWidget(self.material_density_input, 3, 5)
        self.grid_layout.addWidget(self.add_material_button, 4, 5)
        self.grid_layout.addWidget(QLabel("Created Materials:"), 5, 4, 1, 2)
        self.grid_layout.addWidget(self.material_list, 6, 4, 1, 2)

        # Solver settings
        self.solver_type_input = QLineEdit()
        self.solver_tolerance_input = QLineEdit()
        self.solver_max_iterations_input = QLineEdit()
        self.add_solver_button = QPushButton("Add Solver")
        self.add_solver_button.clicked.connect(self.add_solver)
        self.solver_display = QTextEdit()
        self.solver_display.setReadOnly(True)

        self.grid_layout.addWidget(QLabel("Solver Type:"), 0, 6)
        self.grid_layout.addWidget(self.solver_type_input, 0, 7)
        self.grid_layout.addWidget(QLabel("Tolerance:"), 1, 6)
        self.grid_layout.addWidget(self.solver_tolerance_input, 1, 7)
        self.grid_layout.addWidget(QLabel("Max Iterations:"), 2, 6)
        self.grid_layout.addWidget(self.solver_max_iterations_input, 2, 7)
        self.grid_layout.addWidget(self.add_solver_button, 3, 7)
        self.grid_layout.addWidget(QLabel("Solver Settings:"), 4, 6, 1, 2)
        self.grid_layout.addWidget(self.solver_display, 5, 6, 2, 2)

        # Add the grid layout to the main layout
        layout.addLayout(self.grid_layout)

        # Save button
        self.save_button = QPushButton("Save JSON File")
        self.save_button.clicked.connect(self.save_json_file)
        layout.addWidget(self.save_button)

        # Data storage
        self.nodes = []
        self.elements = []
        self.materials = []
        self.solver = {}

        self.setLayout(layout)

    def add_node(self):
        """Add a node to the nodes list."""
        try:
            node_id = int(self.node_id_input.text())
            coordinates = list(map(float, self.node_coordinates_input.text().split(",")))
            if len(coordinates) != 2:
                raise ValueError("Coordinates must have exactly 2 values (x, y).")
            node = {"id": node_id, "coordinates": coordinates}
            self.nodes.append(node)
            self.node_list.addItem(f"ID: {node_id}, Coordinates: {coordinates}")
            self.node_id_input.clear()
            self.node_coordinates_input.clear()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))

    def add_element(self):
        """Add an element to the elements list."""
        try:
            element_id = int(self.element_id_input.text())
            element_type = self.element_type_input.text()
            connectivity = list(map(int, self.element_connectivity_input.text().split(",")))
            element = {"id": element_id, "type": element_type, "connectivity": connectivity}
            self.elements.append(element)
            self.element_list.addItem(f"ID: {element_id}, Type: {element_type}, Connectivity: {connectivity}")
            self.element_id_input.clear()
            self.element_type_input.clear()
            self.element_connectivity_input.clear()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))

    def add_material(self):
        """Add a material to the materials list."""
        try:
            name = self.material_name_input.text()
            E = float(self.material_E_input.text())
            nu = float(self.material_nu_input.text())
            density = float(self.material_density_input.text())
            material = {"name": name, "E": E, "nu": nu, "density": density}
            self.materials.append(material)
            self.material_list.addItem(f"Name: {name}, E: {E}, nu: {nu}, Density: {density}")
            self.material_name_input.clear()
            self.material_E_input.clear()
            self.material_nu_input.clear()
            self.material_density_input.clear()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))

    def add_solver(self):
        """Set the solver settings."""
        try:
            self.solver = {
                "type": self.solver_type_input.text(),
                "tolerance": float(self.solver_tolerance_input.text()),
                "max_iterations": int(self.solver_max_iterations_input.text())
            }
            self.solver_display.setText(f"Type: {self.solver['type']}\n"
                                        f"Tolerance: {self.solver['tolerance']}\n"
                                        f"Max Iterations: {self.solver['max_iterations']}")
            self.solver_type_input.clear()
            self.solver_tolerance_input.clear()
            self.solver_max_iterations_input.clear()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))

    def save_json_file(self):
        """Save all collected data to a JSON file."""
        try:
            data = {
                "nodes": self.nodes,
                "elements": self.elements,
                "materials": self.materials,
                "solver": self.solver
            }
            options = QFileDialog.Options()
            filepath, _ = QFileDialog.getSaveFileName(self, "Save JSON File", "", "JSON Files (*.json);;All Files (*)", options=options)
            if filepath:
                with open(filepath, "w") as file:
                    json.dump(data, file, indent=4)
                QMessageBox.information(self, "File Saved", f"JSON file saved to: {filepath}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"An error occurred: {e}")
