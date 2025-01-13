from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class ProcessingPage(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        layout = QVBoxLayout(self)

        # Add widgets
        layout.addWidget(QLabel("Processing Page"))
        layout.addWidget(QLabel("Run FEM computations here."))
        layout.addWidget(QPushButton("Start Processing"))
        layout.addWidget(QPushButton("Stop Processing"))
