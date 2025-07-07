import sys
from dataclasses import dataclass
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QGroupBox, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QLabel
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

@dataclass
class SimulationParameters:
    simulation_type: str = "Static"
    static_load: float = 0.0  # N
    modal_modes: int = 1
    transient_dt: float = 0.001  # s
    transient_tend: float = 1.0  # s

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HybriDFEM GUI concept")
        self.sim_params = SimulationParameters()
        self.param_widgets = {}
        self._create_ui()

    def _create_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout()

        # 1. Import
        file_group = QGroupBox("Import Files")
        file_layout = QHBoxLayout()
        self.btn_import = QPushButton("Import Mesh/Input File")
        self.btn_import.clicked.connect(self.import_file)
        file_layout.addWidget(self.btn_import)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # 2. Parameters
        self.param_group = QGroupBox("Simulation Parameters")
        self.param_layout = QFormLayout()
        self.param_group.setLayout(self.param_layout)
        self.combo_type = QComboBox()
        types = ["Static", "Modal", "Transient"]
        self.combo_type.addItems(types)
        self.combo_type.currentTextChanged.connect(self._on_type_changed)
        self.param_layout.addRow(QLabel("Simulation type:"), self.combo_type)
        self._update_parameters(self.sim_params.simulation_type)
        main_layout.addWidget(self.param_group)

        # 3. Controls
        run_group = QGroupBox("Run Simulation")
        run_layout = QHBoxLayout()
        self.btn_run = QPushButton("Start")
        self.btn_run.clicked.connect(self.run_simulation)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_simulation)
        run_layout.addWidget(self.btn_run)
        run_layout.addWidget(self.btn_stop)
        run_group.setLayout(run_layout)
        main_layout.addWidget(run_group)

        # 4. Preview plot
        preview_group = QGroupBox("Results Preview")
        preview_layout = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        preview_layout.addWidget(self.canvas)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        # 5. Export
        export_group = QGroupBox("Export Results")
        export_layout = QHBoxLayout()
        self.btn_export = QPushButton("Export CSV/VTK")
        self.btn_export.clicked.connect(self.export_results)
        export_layout.addWidget(self.btn_export)
        export_group.setLayout(export_layout)
        main_layout.addWidget(export_group)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _on_type_changed(self, new_type: str):
        self.sim_params.simulation_type = new_type
        self._update_parameters(new_type)

    def _update_parameters(self, sim_type: str):
        # clear
        for w in self.param_widgets.values(): w.deleteLater()
        self.param_widgets.clear()
        while self.param_layout.rowCount() > 1:
            self.param_layout.removeRow(1)

        if sim_type == "Static":
            sb = QDoubleSpinBox()
            sb.setSuffix(" N"); sb.setRange(0,1e6)
            sb.setValue(self.sim_params.static_load)
            sb.valueChanged.connect(lambda v: setattr(self.sim_params, 'static_load', v))
            self.param_layout.addRow(QLabel("Load magnitude:"), sb)
            self.param_widgets['static_load'] = sb
        elif sim_type == "Modal":
            sb = QSpinBox(); sb.setRange(1,100)
            sb.setValue(self.sim_params.modal_modes)
            sb.valueChanged.connect(lambda v: setattr(self.sim_params, 'modal_modes', v))
            self.param_layout.addRow(QLabel("Number of modes:"), sb)
            self.param_widgets['modal_modes'] = sb
        else:  # Transient
            sb1 = QDoubleSpinBox(); sb1.setSuffix(" s"); sb1.setDecimals(4)
            sb1.setRange(1e-6,10); sb1.setValue(self.sim_params.transient_dt)
            sb1.valueChanged.connect(lambda v: setattr(self.sim_params, 'transient_dt', v))
            self.param_layout.addRow(QLabel("Time step:"), sb1)
            self.param_widgets['transient_dt'] = sb1
            sb2 = QDoubleSpinBox(); sb2.setSuffix(" s"); sb2.setRange(0,1e4)
            sb2.setValue(self.sim_params.transient_tend)
            sb2.valueChanged.connect(lambda v: setattr(self.sim_params, 'transient_tend', v))
            self.param_layout.addRow(QLabel("Total time:"), sb2)
            self.param_widgets['transient_tend'] = sb2

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select input file", "", "All Files (*)")
        if path: print(f"Imported: {path}")

    def run_simulation(self):
        params = self.sim_params
        print("Running with", params)
        # TODO: call solver; here simulate some data
        import numpy as np
        t = np.linspace(0, params.transient_tend if params.simulation_type=='Transient' else 1, 100)
        y = np.sin(2*np.pi*t) * (params.static_load if params.simulation_type=='Static' else 1)
        self.update_preview(t, y)

    def update_preview(self, x, y):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('response')
        ax.set_title(f"{self.sim_params.simulation_type} Preview")
        self.canvas.draw()

    def stop_simulation(self):
        print("Simulation stopped")

    def export_results(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save results as", "results.csv", "CSV Files (*.csv);;VTK Files (*.vtk)")
        if path: print(f"Exported to: {path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
