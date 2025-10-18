from PyQt6.QtWidgets import (QWidget, QVBoxLayout,
                             QPushButton, QMessageBox, QGroupBox, QFormLayout, QComboBox,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar)


# Matplotlib for visualization

class AnalysisPanel(QWidget):
    """Panel for analysis configuration and execution"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Analysis Type
        type_group = QGroupBox("Analysis Type")
        type_layout = QVBoxLayout()

        self.analysis_type = QComboBox()
        self.analysis_type.addItems([
            "Linear Static",
            "Nonlinear Static (Force Control)",
            "Nonlinear Static (Displacement Control)",
            "Dynamic Linear",
            "Dynamic Nonlinear",
            "Modal Analysis"
        ])
        self.analysis_type.currentTextChanged.connect(self.update_analysis_options)
        type_layout.addWidget(self.analysis_type)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Analysis Parameters
        self.params_group = QGroupBox("Analysis Parameters")
        self.params_layout = QFormLayout()

        # Static analysis parameters
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 10000)
        self.steps_input.setValue(100)
        self.params_layout.addRow("Number of Steps:", self.steps_input)

        self.tolerance_input = QDoubleSpinBox()
        self.tolerance_input.setRange(1e-10, 1.0)
        self.tolerance_input.setValue(1e-6)
        self.tolerance_input.setDecimals(10)
        self.params_layout.addRow("Tolerance:", self.tolerance_input)

        # Dynamic analysis parameters
        self.time_input = QDoubleSpinBox()
        self.time_input.setRange(0.001, 1000)
        self.time_input.setValue(1.0)
        self.time_input.setSuffix(" s")
        self.params_layout.addRow("Total Time (T):", self.time_input)

        self.dt_input = QDoubleSpinBox()
        self.dt_input.setRange(1e-6, 1.0)
        self.dt_input.setValue(0.001)
        self.dt_input.setSuffix(" s")
        self.dt_input.setDecimals(6)
        self.params_layout.addRow("Time Step (dt):", self.dt_input)

        # Modal analysis parameters
        self.modes_input = QSpinBox()
        self.modes_input.setRange(1, 100)
        self.modes_input.setValue(10)
        self.params_layout.addRow("Number of Modes:", self.modes_input)

        # Geometry options
        self.linear_geom = QCheckBox("Linear Geometry")
        self.linear_geom.setChecked(True)
        self.params_layout.addRow("", self.linear_geom)

        self.params_group.setLayout(self.params_layout)
        layout.addWidget(self.params_group)

        # Run Analysis Button
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn_run)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        self.setLayout(layout)

        self.update_analysis_options()

    def update_analysis_options(self):
        """Show/hide parameters based on analysis type"""
        analysis = self.analysis_type.currentText()

        # Hide all first
        for i in range(self.params_layout.rowCount()):
            label = self.params_layout.itemAt(i, QFormLayout.ItemRole.LabelRole)
            field = self.params_layout.itemAt(i, QFormLayout.ItemRole.FieldRole)
            if label and label.widget():
                label.widget().setVisible(False)
            if field and field.widget():
                field.widget().setVisible(False)

        # Show relevant parameters
        if "Static" in analysis:
            self.show_row("Number of Steps:")
            self.show_row("Tolerance:")
            self.show_row("Linear Geometry")

        if "Dynamic" in analysis:
            self.show_row("Total Time (T):")
            self.show_row("Time Step (dt):")
            self.show_row("Linear Geometry")

        if "Modal" in analysis:
            self.show_row("Number of Modes:")

    def show_row(self, label_text):
        """Helper to show a specific row in form layout"""
        for i in range(self.params_layout.rowCount()):
            label = self.params_layout.itemAt(i, QFormLayout.ItemRole.LabelRole)
            field = self.params_layout.itemAt(i, QFormLayout.ItemRole.FieldRole)
            if label and label.widget() and label.widget().text() == label_text:
                label.widget().setVisible(True)
                if field and field.widget():
                    field.widget().setVisible(True)
                break

    def run_analysis(self):
        if not hasattr(self.main_window, 'structure') or not self.main_window.structure:
            QMessageBox.warning(self, "Warning", "No structure loaded. Please create or import geometry first.")
            return

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        analysis_type = self.analysis_type.currentText()
        self.main_window.log_message(f"Starting {analysis_type}...")

        try:
            # Get the structure
            structure = self.main_window.structure

            # Set geometry linearity
            structure.set_lin_geom(self.linear_geom.isChecked())

            # Initialize nodes if not done
            if not structure.nb_dofs:
                structure.make_nodes()
                self.main_window.log_message(f"Structure initialized with {structure.nb_dofs} DOFs")

            # Run appropriate analysis
            if analysis_type == "Linear Static":
                self.run_linear_static(structure)
            elif "Nonlinear Static" in analysis_type:
                self.run_nonlinear_static(structure)
            elif "Dynamic Linear" in analysis_type:
                self.run_dynamic_linear(structure)
            elif "Dynamic Nonlinear" in analysis_type:
                self.run_dynamic_nonlinear(structure)
            elif "Modal" in analysis_type:
                self.run_modal(structure)

            self.progress_bar.setValue(100)
            self.main_window.log_message("Analysis completed!")
            self.main_window.update_results()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            self.main_window.log_message(f"Analysis error: {str(e)}")

        finally:
            self.btn_run.setEnabled(True)
            self.progress_bar.setVisible(False)

    def run_linear_static(self, structure):
        """Run linear static analysis"""
        from Theo.Objects.Solver import Static
        self.progress_bar.setValue(50)
        Static.solve_linear(structure)

    def run_nonlinear_static(self, structure):
        """Run nonlinear static analysis"""
        from Theo.Objects.Solver import Static
        steps = self.steps_input.value()
        tol = self.tolerance_input.value()
        self.progress_bar.setValue(50)
        Static.solve_forcecontrol(structure, steps=steps, tol=tol)

    def run_dynamic_linear(self, structure):
        """Run dynamic linear analysis"""
        from Theo.Objects.Solver import Static
        T = self.time_input.value()
        dt = self.dt_input.value()
        self.progress_bar.setValue(50)
        Static.solve_dyn_linear(structure, T=T, dt=dt)

    def run_dynamic_nonlinear(self, structure):
        """Run dynamic nonlinear analysis"""
        from Theo.Objects.Solver import Static
        T = self.time_input.value()
        dt = self.dt_input.value()
        self.progress_bar.setValue(50)
        Static.solve_dyn_nonlinear(structure, T=T, dt=dt)

    def run_modal(self, structure):
        """Run modal analysis"""
        from Theo.Objects.Solver import Modal
        modes = self.modes_input.value()
        self.progress_bar.setValue(50)
        modal_solver = Modal(modes=modes)
        modal_solver.modal(structure)
