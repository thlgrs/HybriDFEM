import os

from PyQt6.QtWidgets import (QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QMessageBox, QTextEdit,
                             QGroupBox, QFormLayout, QLineEdit, QComboBox)


# Matplotlib for visualization

class GeometryPanel(QWidget):
    """Panel for geometry creation and Rhino import"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Rhino Import Section
        rhino_group = QGroupBox("Rhino Import")
        rhino_layout = QVBoxLayout()

        import_layout = QHBoxLayout()
        self.rhino_path_edit = QLineEdit()
        self.rhino_path_edit.setPlaceholderText("Select Rhino file or geometry...")
        import_layout.addWidget(self.rhino_path_edit)

        self.btn_browse_rhino = QPushButton("Browse")
        self.btn_browse_rhino.clicked.connect(self.browse_rhino_file)
        import_layout.addWidget(self.btn_browse_rhino)

        self.btn_import_rhino = QPushButton("Import from Rhino")
        self.btn_import_rhino.clicked.connect(self.import_from_rhino)
        import_layout.addWidget(self.btn_import_rhino)

        rhino_layout.addLayout(import_layout)

        # Import options
        options_layout = QFormLayout()
        self.layer_filter = QLineEdit("*")
        self.layer_filter.setPlaceholderText("Layer filter (* for all)")
        options_layout.addRow("Layer Filter:", self.layer_filter)

        self.import_type = QComboBox()
        self.import_type.addItems(["Auto-detect", "Blocks", "Beams", "Shells", "Hybrid"])
        options_layout.addRow("Import as:", self.import_type)

        rhino_layout.addLayout(options_layout)
        rhino_group.setLayout(rhino_layout)
        layout.addWidget(rhino_group)

        # Manual Geometry Creation
        manual_group = QGroupBox("Manual Geometry Creation")
        manual_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.btn_add_block = QPushButton("Add Block")
        self.btn_add_block.clicked.connect(self.add_block)
        btn_layout.addWidget(self.btn_add_block)

        self.btn_add_beam = QPushButton("Add Beam")
        self.btn_add_beam.clicked.connect(self.add_beam)
        btn_layout.addWidget(self.btn_add_beam)

        self.btn_add_arch = QPushButton("Add Arch")
        self.btn_add_arch.clicked.connect(self.add_arch)
        btn_layout.addWidget(self.btn_add_arch)

        manual_layout.addLayout(btn_layout)
        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)

        # Geometry Info
        info_group = QGroupBox("Geometry Information")
        info_layout = QVBoxLayout()
        self.geometry_info = QTextEdit()
        self.geometry_info.setReadOnly(True)
        self.geometry_info.setMaximumHeight(150)
        info_layout.addWidget(self.geometry_info)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        self.setLayout(layout)

    def browse_rhino_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Rhino File", "",
            "Rhino Files (*.3dm);;Text Files (*.txt);;All Files (*.*)"
        )
        if file_path:
            self.rhino_path_edit.setText(file_path)

    def import_from_rhino(self):
        file_path = self.rhino_path_edit.text()
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please select a file first")
            return

        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Warning", "File does not exist")
            return

        # Log the import
        self.main_window.log_message(f"Importing from: {file_path}")

        try:
            # Call the from_Rhino class method based on structure type
            import_type = self.import_type.currentText()
            self.main_window.log_message(f"Import type: {import_type}")

            # TODO: Implement actual Rhino import
            # structure = Structure_Type.from_Rhino(file_path, ...)

            self.main_window.log_message("Import completed successfully!")
            self.update_geometry_info()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Import failed: {str(e)}")
            self.main_window.log_message(f"Import error: {str(e)}")

    def add_block(self):
        self.main_window.log_message("Add Block dialog (to be implemented)")
        # TODO: Open dialog for block parameters

    def add_beam(self):
        self.main_window.log_message("Add Beam dialog (to be implemented)")
        # TODO: Open dialog for beam parameters

    def add_arch(self):
        self.main_window.log_message("Add Arch dialog (to be implemented)")
        # TODO: Open dialog for arch parameters

    def update_geometry_info(self):
        if hasattr(self.main_window, 'structure') and self.main_window.structure:
            structure = self.main_window.structure
            info = []
            info.append(f"Structure Type: {type(structure).__name__}")
            info.append(f"Number of Nodes: {len(structure.list_nodes)}")

            if hasattr(structure, 'list_blocks'):
                info.append(f"Number of Blocks: {len(structure.list_blocks)}")
            if hasattr(structure, 'list_fes'):
                info.append(f"Number of FE Elements: {len(structure.list_fes)}")
            if hasattr(structure, 'list_cfs'):
                info.append(f"Number of Contact Faces: {len(structure.list_cfs)}")

            info.append(f"Total DOFs: {structure.nb_dofs if structure.nb_dofs else 'Not initialized'}")

            self.geometry_info.setText("\n".join(info))
        else:
            self.geometry_info.setText("No structure loaded")
