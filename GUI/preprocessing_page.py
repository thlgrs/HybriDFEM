import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QFileDialog, QMessageBox
)
from Widgets.json_creator_window import JsonCreatorWindow


class PreprocessingPage(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        layout = QVBoxLayout(self)

        # Create JSON File Option
        self.create_json_label = QLabel("Create a New Input File")
        self.create_json_button = QPushButton("Create Input File")
        self.create_json_button.clicked.connect(self.open_json_creator_window)
        layout.addWidget(self.create_json_label)
        layout.addWidget(self.create_json_button)

        # Select JSON File Option
        self.select_json_label = QLabel("Select an Existing JSON File")
        self.select_json_button = QPushButton("Select JSON File")
        self.select_json_button.clicked.connect(self.select_json_file)
        layout.addWidget(self.select_json_label)
        layout.addWidget(self.select_json_button)

        # Show JSON File Content
        self.show_json_label = QLabel("Selected JSON File Content")
        self.json_content_display = QTextEdit()
        self.json_content_display.setReadOnly(True)
        layout.addWidget(self.show_json_label)
        layout.addWidget(self.json_content_display)

        # Store the selected file path
        self.selected_json_path = None

    def open_json_creator_window(self):
        """Open the interactive JSON creator window."""
        self.json_creator_window = JsonCreatorWindow()
        self.json_creator_window.show()

    def select_json_file(self):
        """Open a file dialog to select a JSON file and display its content."""
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Select JSON File", "", "JSON Files (*.json);;All Files (*)", options=options)

        if filepath:
            self.selected_json_path = filepath
            self.display_json_content(filepath)

    def display_json_content(self, filepath):
        """Display the content of the selected JSON file."""
        try:
            with open(filepath, "r") as file:
                content = json.load(file)
                formatted_content = json.dumps(content, indent=4)
                self.json_content_display.setText(formatted_content)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read file: {e}")
