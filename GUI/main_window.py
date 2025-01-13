from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
from preprocessing_page import PreprocessingPage
from processing_page import ProcessingPage
from postprocessing_page import PostprocessingPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("HybridFEM")

        # Adjust window size to screen
        self.set_initial_window_size()

        # Create the central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Create QTabWidget for tab layout
        self.tabs = QTabWidget()

        # Add tabs
        self.tabs.addTab(PreprocessingPage(), "Preprocessing")
        self.tabs.addTab(ProcessingPage(), "Processing")
        self.tabs.addTab(PostprocessingPage(), "Postprocessing")

        # Add QTabWidget to the main layout
        main_layout.addWidget(self.tabs)
        self.setCentralWidget(central_widget)

    def set_initial_window_size(self):
        """
        Dynamically adjust the initial window size based on the screen resolution.
        """
        screen = QApplication.primaryScreen().geometry()  # Get screen geometry
        width, height = screen.width(), screen.height()
        self.setGeometry(width // 10, height // 10, width * 4 // 5, height * 4 // 5)

# Run the application
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
