from PyQt6.QtWidgets import (QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel,
                             QTextEdit,
                             QGroupBox, QFormLayout, QDoubleSpinBox)


# Matplotlib for visualization

class MaterialPanel(QWidget):
    """Panel for material properties and boundary conditions"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Material Properties
        material_group = QGroupBox("Material Properties")
        material_layout = QFormLayout()

        self.E_input = QDoubleSpinBox()
        self.E_input.setRange(1e6, 1e12)
        self.E_input.setValue(30e9)
        self.E_input.setSuffix(" Pa")
        self.E_input.setDecimals(2)
        material_layout.addRow("Young's Modulus (E):", self.E_input)

        self.nu_input = QDoubleSpinBox()
        self.nu_input.setRange(0.0, 0.5)
        self.nu_input.setValue(0.2)
        self.nu_input.setSingleStep(0.01)
        material_layout.addRow("Poisson's Ratio (ν):", self.nu_input)

        self.rho_input = QDoubleSpinBox()
        self.rho_input.setRange(0, 10000)
        self.rho_input.setValue(2400)
        self.rho_input.setSuffix(" kg/m³")
        material_layout.addRow("Density (ρ):", self.rho_input)

        material_group.setLayout(material_layout)
        layout.addWidget(material_group)

        # Boundary Conditions
        bc_group = QGroupBox("Boundary Conditions")
        bc_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.btn_add_support = QPushButton("Add Support")
        self.btn_add_support.clicked.connect(self.add_support)
        btn_layout.addWidget(self.btn_add_support)

        self.btn_add_load = QPushButton("Add Load")
        self.btn_add_load.clicked.connect(self.add_load)
        btn_layout.addWidget(self.btn_add_load)

        bc_layout.addLayout(btn_layout)

        self.bc_list = QTextEdit()
        self.bc_list.setReadOnly(True)
        self.bc_list.setMaximumHeight(200)
        bc_layout.addWidget(QLabel("Applied Conditions:"))
        bc_layout.addWidget(self.bc_list)

        bc_group.setLayout(bc_layout)
        layout.addWidget(bc_group)

        layout.addStretch()
        self.setLayout(layout)

    def add_support(self):
        self.main_window.log_message("Add Support dialog (to be implemented)")
        # TODO: Open dialog to select nodes and constraint directions

    def add_load(self):
        self.main_window.log_message("Add Load dialog (to be implemented)")
        # TODO: Open dialog to apply loads
