"""
HybridFEM GUI - Main Application
Structural analysis tool with Rhino integration
"""

import sys

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget)

from GUI.Panels.AnalysisPanel import *
from GUI.Panels.GeometryPanel import *
from GUI.Panels.MaterialPanel import *
from GUI.Panels.ResultsPanel import *


class AnalysisWorker(QThread):
    """Worker thread for running analysis without freezing GUI"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    log = pyqtSignal(str)

    def __init__(self, analysis_func, *args, **kwargs):
        super().__init__()
        self.analysis_func = analysis_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.log.emit("Starting analysis...")
            result = self.analysis_func(*self.args, **self.kwargs)
            self.log.emit("Analysis completed successfully!")
            self.finished.emit(True, "Analysis completed")
        except Exception as e:
            self.log.emit(f"Error: {str(e)}")
            self.finished.emit(False, str(e))


class HybridFEMMainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("HybridFEM")

        # Get the screen geometry
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Define window size as half of screen
        window_width = int(screen_width * 0.5)
        window_height = int(screen_height * 0.5)

        # Compute centered position
        x = int((screen_width - window_width) / 2)
        y = int((screen_height - window_height) / 3)

        # Apply geometry
        self.setGeometry(x, y, window_width, window_height)

        # Rest of your UI setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()

        # Add panels as tabs
        self.geometry_panel = GeometryPanel(self)
        self.tabs.addTab(self.geometry_panel, "Geometry")

        self.material_panel = MaterialPanel(self)
        self.tabs.addTab(self.material_panel, "Material & BC")

        self.analysis_panel = AnalysisPanel(self)
        self.tabs.addTab(self.analysis_panel, "Analysis")

        self.results_panel = ResultsPanel(self)
        self.tabs.addTab(self.results_panel, "Results")

        main_layout.addWidget(self.tabs)

        # Console log at bottom
        log_group = QGroupBox("Console Log")
        log_layout = QVBoxLayout()
        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setMaximumHeight(150)
        log_layout.addWidget(self.console_log)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        central_widget.setLayout(main_layout)

        # Menu bar
        self.menu_bar()

        # Initial log message
        self.log_message("HybridFEM initialized. Ready to load geometry.")

    def mousePressEvent(self, event):
        """Record initial position when user clicks to drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        """Move the window while holding left mouse button."""
        if event.buttons() == Qt.MouseButton.LeftButton:
            diff = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + diff)
            self._drag_pos = event.globalPosition().toPoint()

    def menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        new_action = file_menu.addAction("New Project")
        new_action.triggered.connect(self.new_project)

        open_action = file_menu.addAction("Open Project")
        open_action.triggered.connect(self.open_project)

        save_action = file_menu.addAction("Save Project")
        save_action.triggered.connect(self.save_project)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)

    def new_project(self):
        self.structure = None
        self.log_message("New project created")
        self.geometry_panel.geometry_info.clear()
        self.results_panel.results_text.clear()
        self.results_panel.figure.clear()
        self.results_panel.canvas.draw()

    def open_project(self):
        self.log_message("Open project (to be implemented)")
        # TODO: Implement project loading

    def save_project(self):
        self.log_message("Save project (to be implemented)")
        # TODO: Implement project saving

    def show_about(self):
        QMessageBox.about(
            self, "About HybridFEM",
            "HybridFEM - Structural Analysis Tool\n\n"
            "A hybrid finite element method program for\n"
            "analyzing structures with blocks and beam elements.\n\n"
            "Features Rhino integration for geometry import/export."
        )

    def log_message(self, message):
        """Add message to console log"""
        self.console_log.append(f"> {message}")
        self.console_log.verticalScrollBar().setValue(
            self.console_log.verticalScrollBar().maximum()
        )

    def update_results(self):
        """Update results after analysis"""
        self.results_panel.update_plot()
        self.results_panel.update_results_summary()
        self.tabs.setCurrentWidget(self.results_panel)


def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Windows11')

    # Create and show main window
    window = HybridFEMMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
