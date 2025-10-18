import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QMessageBox, QTextEdit,
                             QGroupBox, QFormLayout, QDoubleSpinBox, QCheckBox)
# Matplotlib for visualization
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ResultsPanel(QWidget):
    """Panel for results visualization"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Export options
        export_group = QGroupBox("Export Results")
        export_layout = QHBoxLayout()

        self.btn_export_vtk = QPushButton("Export VTK")
        self.btn_export_vtk.clicked.connect(self.export_vtk)
        export_layout.addWidget(self.btn_export_vtk)

        self.btn_export_rhino = QPushButton("Export for Rhino")
        self.btn_export_rhino.clicked.connect(self.export_rhino)
        export_layout.addWidget(self.btn_export_rhino)

        self.btn_export_plot = QPushButton("Export Plot")
        self.btn_export_plot.clicked.connect(self.export_plot)
        export_layout.addWidget(self.btn_export_plot)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Visualization options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()

        self.deformation_scale = QDoubleSpinBox()
        self.deformation_scale.setRange(0, 1000)
        self.deformation_scale.setValue(1.0)
        self.deformation_scale.valueChanged.connect(self.update_plot)
        viz_layout.addRow("Deformation Scale:", self.deformation_scale)

        self.show_undeformed = QCheckBox("Show Undeformed")
        self.show_undeformed.setChecked(True)
        self.show_undeformed.stateChanged.connect(self.update_plot)
        viz_layout.addRow("", self.show_undeformed)

        self.show_forces = QCheckBox("Show Forces")
        self.show_forces.setChecked(False)
        self.show_forces.stateChanged.connect(self.update_plot)
        viz_layout.addRow("", self.show_forces)

        self.show_supports = QCheckBox("Show Supports")
        self.show_supports.setChecked(True)
        self.show_supports.stateChanged.connect(self.update_plot)
        viz_layout.addRow("", self.show_supports)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Matplotlib canvas for visualization
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Results summary
        summary_group = QGroupBox("Results Summary")
        summary_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        summary_layout.addWidget(self.results_text)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        self.setLayout(layout)

    def update_plot(self):
        """Update the visualization plot"""
        if not hasattr(self.main_window, 'structure') or not self.main_window.structure:
            return

        structure = self.main_window.structure
        if structure.U is None or not structure.U.any():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        scale = self.deformation_scale.value()

        # Plot based on structure type
        # This is a placeholder - you'll need to integrate with your plotting methods
        try:
            # For blocks
            if hasattr(structure, 'list_blocks'):
                for block in structure.list_blocks:
                    # Get displacements
                    block.disps = structure.U[block.dofs] if hasattr(block, 'dofs') else np.zeros(3)
                    # Plot block (simplified - you'll use your actual plot_block method)

            # For FE elements
            if hasattr(structure, 'list_fes'):
                for fe in structure.list_fes:
                    if hasattr(fe, 'PlotDefShapeElem'):
                        defs = structure.U[fe.dofs] if hasattr(fe, 'dofs') else np.zeros(len(fe.dofs))
                        # Plot element (you'll use your actual plotting method)

            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Deformed Structure')

            self.canvas.draw()

        except Exception as e:
            self.main_window.log_message(f"Plot error: {str(e)}")

    def update_results_summary(self):
        """Update the results summary text"""
        if not hasattr(self.main_window, 'structure') or not self.main_window.structure:
            self.results_text.setText("No results available")
            return

        structure = self.main_window.structure

        summary = []
        summary.append("=== Analysis Results ===\n")

        if structure.U is not None and structure.U.any():
            max_disp = np.max(np.abs(structure.U))
            summary.append(f"Maximum Displacement: {max_disp:.6e} m")

        if hasattr(structure, 'P_r') and structure.P_r is not None:
            max_reaction = np.max(np.abs(structure.P_r))
            summary.append(f"Maximum Reaction: {max_reaction:.6e} N")

        if hasattr(structure, 'eig_vals'):
            summary.append(f"\nFirst {len(structure.eig_vals)} Natural Frequencies:")
            for i, freq in enumerate(structure.eig_vals[:10]):
                summary.append(f"  Mode {i + 1}: {np.sqrt(freq) / (2 * np.pi):.3f} Hz")

        self.results_text.setText("\n".join(summary))

    def export_vtk(self):
        """Export results to VTK format"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save VTK File", "", "VTK Files (*.vtk);;All Files (*.*)"
        )
        if file_path:
            try:
                # TODO: Implement VTK export using your structure's export_to_vtk method
                self.main_window.log_message(f"Exported to VTK: {file_path}")
                QMessageBox.information(self, "Success", "VTK export completed")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def export_rhino(self):
        """Export results for Rhino"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Rhino Export File", "", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)"
        )
        if file_path:
            try:
                structure = self.main_window.structure
                with open(file_path, 'w') as f:
                    f.write("# HybridFEM Results Export for Rhino\n")
                    f.write(f"# Structure Type: {type(structure).__name__}\n")
                    f.write(f"# Total DOFs: {structure.nb_dofs}\n\n")

                    # Export node displacements
                    f.write("# Node Displacements (node_id, x, y, ux, uy, rz)\n")
                    for i, node in enumerate(structure.list_nodes):
                        ux = structure.U[3 * i] if structure.U is not None else 0.0
                        uy = structure.U[3 * i + 1] if structure.U is not None else 0.0
                        rz = structure.U[3 * i + 2] if structure.U is not None else 0.0
                        f.write(f"{i},{node[0]},{node[1]},{ux},{uy},{rz}\n")

                self.main_window.log_message(f"Exported for Rhino: {file_path}")
                QMessageBox.information(self, "Success", "Rhino export completed")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def export_plot(self):
        """Export current plot as image"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*.*)"
        )
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.main_window.log_message(f"Plot saved: {file_path}")
                QMessageBox.information(self, "Success", "Plot exported")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
