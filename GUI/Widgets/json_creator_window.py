from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QFormLayout, QPushButton, QMessageBox, QFileDialog,
    QGridLayout, QListWidget, QComboBox, QStackedWidget, QHBoxLayout, QGroupBox
)
import json


class JsonCreatorWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Create JSON Input File")
        self.setGeometry(200, 200, 1000, 600)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Predefined lists for dropdown menus
        self.element_types = ["truss", "beam", "shell", "solid"]
        self.material_types = ["linear", "bilinear", "spring"]
        self.solver_types = ["linear", "nonlinear", "dynamic"]

        # Node Input Section
        node_group = QGroupBox("Nodes")
        node_layout = QFormLayout(node_group)
        self.node_id_input = QLineEdit()
        self.node_coordinates_input = QLineEdit()
        node_layout.addRow("Node ID:", self.node_id_input)
        node_layout.addRow("Coordinates (x, y):", self.node_coordinates_input)

        # Element Input Section
        element_group = QGroupBox("Elements")
        element_layout = QFormLayout(element_group)
        self.element_type_input = QComboBox()
        self.element_type_input.addItems(self.element_types)
        self.element_type_input.currentIndexChanged.connect(self.update_element_parameters)
        self.element_id_input = QLineEdit()
        self.element_parameters = QStackedWidget()
        self.create_element_parameter_forms()
        element_layout.addRow("Element Type:", self.element_type_input)
        element_layout.addRow("Element ID:", self.element_id_input)
        element_layout.addRow(self.element_parameters)

        # Material Input Section
        material_group = QGroupBox("Materials")
        material_layout = QFormLayout(material_group)
        self.material_type_input = QComboBox()
        self.material_type_input.addItems(self.material_types)
        self.material_type_input.currentIndexChanged.connect(self.update_material_parameters)
        self.material_name_input = QLineEdit()
        self.material_parameters = QStackedWidget()
        self.create_material_parameter_forms()
        material_layout.addRow("Material Type:", self.material_type_input)
        material_layout.addRow("Material Name:", self.material_name_input)
        material_layout.addRow(self.material_parameters)

        # Solver Input Section
        solver_group = QGroupBox("Solver")
        solver_layout = QFormLayout(solver_group)
        self.solver_type_input = QComboBox()
        self.solver_type_input.addItems(self.solver_types)
        self.solver_type_input.currentIndexChanged.connect(self.update_solver_parameters)
        self.solver_parameters = QStackedWidget()
        self.create_solver_parameter_forms()
        solver_layout.addRow("Solver Type:", self.solver_type_input)
        solver_layout.addRow(self.solver_parameters)

        # Input Sections Layout
        input_sections_layout = QGridLayout()
        input_sections_layout.addWidget(node_group, 0, 0)
        input_sections_layout.addWidget(element_group, 0, 1)
        input_sections_layout.addWidget(material_group, 0, 2)
        input_sections_layout.addWidget(solver_group, 0, 3)

        main_layout.addLayout(input_sections_layout)

        # Buttons and Created Lists
        buttons_and_lists_layout = QVBoxLayout()

        # Add Buttons
        add_buttons_layout = QHBoxLayout()
        self.add_node_button = QPushButton("Add Node")
        self.add_node_button.clicked.connect(self.add_node)
        self.add_element_button = QPushButton("Add Element")
        self.add_element_button.clicked.connect(self.add_element)
        self.add_material_button = QPushButton("Add Material")
        self.add_material_button.clicked.connect(self.add_material)
        self.add_solver_button = QPushButton("Add Solver")
        self.add_solver_button.clicked.connect(self.add_solver)

        add_buttons_layout.addWidget(self.add_node_button)
        add_buttons_layout.addWidget(self.add_element_button)
        add_buttons_layout.addWidget(self.add_material_button)
        add_buttons_layout.addWidget(self.add_solver_button)

        # Created Lists
        lists_layout = QHBoxLayout()
        self.node_list = QListWidget()
        self.element_list = QListWidget()
        self.material_list = QListWidget()
        self.solver_display = QListWidget()

        lists_layout.addWidget(self.node_list)
        lists_layout.addWidget(self.element_list)
        lists_layout.addWidget(self.material_list)
        lists_layout.addWidget(self.solver_display)

        # Combine Buttons and Lists
        buttons_and_lists_layout.addLayout(add_buttons_layout)
        buttons_and_lists_layout.addLayout(lists_layout)
        main_layout.addLayout(buttons_and_lists_layout)

        # Save Button
        self.save_button = QPushButton("Save JSON File")
        self.save_button.clicked.connect(self.save_json_file)
        main_layout.addWidget(self.save_button)

        # Data Storage
        self.nodes = []
        self.elements = []
        self.materials = []
        self.solver = {}

        self.setLayout(main_layout)

    def create_element_parameter_forms(self):
        """Create parameter forms for each element type."""
        for element_type in self.element_types:
            form = QWidget()
            form_layout = QFormLayout(form)
            if element_type == "truss":
                form_layout.addRow("Length:", QLineEdit())
            elif element_type == "beam":
                form_layout.addRow("Moment of Inertia:", QLineEdit())
            elif element_type == "shell":
                form_layout.addRow("Thickness:", QLineEdit())
            elif element_type == "solid":
                form_layout.addRow("Volume:", QLineEdit())
            self.element_parameters.addWidget(form)

    def update_element_parameters(self):
        """Update the displayed parameters based on the selected element type."""
        self.element_parameters.setCurrentIndex(self.element_type_input.currentIndex())

    def create_material_parameter_forms(self):
        """Create parameter forms for each material type."""
        for material_type in self.material_types:
            form = QWidget()
            form_layout = QFormLayout(form)
            if material_type == "linear":
                form_layout.addRow("Young's Modulus (E):", QLineEdit())
                form_layout.addRow("Poisson's Ratio (nu):", QLineEdit())
            elif material_type == "bilinear":
                form_layout.addRow("Yield Stress:", QLineEdit())
                form_layout.addRow("Hardening Modulus:", QLineEdit())
            elif material_type == "spring":
                form_layout.addRow("Spring Constant (k):", QLineEdit())
            self.material_parameters.addWidget(form)

    def update_material_parameters(self):
        """Update the displayed parameters based on the selected material type."""
        self.material_parameters.setCurrentIndex(self.material_type_input.currentIndex())

    def create_solver_parameter_forms(self):
        """Create parameter forms for each solver type."""
        for solver_type in self.solver_types:
            form = QWidget()
            form_layout = QFormLayout(form)
            if solver_type == "linear":
                form_layout.addRow("Convergence Tolerance:", QLineEdit())
            elif solver_type == "nonlinear":
                form_layout.addRow("Max Load Steps:", QLineEdit())
            elif solver_type == "dynamic":
                form_layout.addRow("Time Step:", QLineEdit())
                form_layout.addRow("Duration:", QLineEdit())
            self.solver_parameters.addWidget(form)

    def update_solver_parameters(self):
        """Update the displayed parameters based on the selected solver type."""
        self.solver_parameters.setCurrentIndex(self.solver_type_input.currentIndex())

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
            element_type = self.element_type_input.currentText()
            element = {"id": element_id, "type": element_type}
            self.elements.append(element)
            self.element_list.addItem(f"ID: {element_id}, Type: {element_type}")
            self.element_id_input.clear()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))

    def add_material(self):
        """Add a material to the materials list."""
        try:
            material_type = self.material_type_input.currentText()
            name = self.material_name_input.text()
            material = {"type": material_type, "name": name}
            self.materials.append(material)
            self.material_list.addItem(f"Type: {material_type}, Name: {name}")
            self.material_name_input.clear()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))

    def add_solver(self):
        """Add solver settings to the solver data."""
        try:
            solver_type = self.solver_type_input.currentText()
            self.solver = {"type": solver_type}
            self.solver_display.clear()
            self.solver_display.addItem(f"Solver Type: {solver_type}")
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
