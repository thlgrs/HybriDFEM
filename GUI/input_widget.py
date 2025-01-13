from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QTextEdit, QWidget, QMessageBox
)


class NodeInputForm2D(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Add Multiple 2D Nodes")
        self.setGeometry(100, 100, 400, 400)

        # Initialize node storage and ID counter
        self.nodes = []
        self.node_counter = 1  # Counter for default Node IDs (starts at 1)

        # Main layout
        layout = QVBoxLayout()

        # Node ID input
        self.node_id_label = QLabel("Node ID (default: n1, n2, ...):")
        self.node_id_input = QLineEdit()
        layout.addWidget(self.node_id_label)
        layout.addWidget(self.node_id_input)

        # Coordinate inputs (x and y)
        self.coordinates_layout = QHBoxLayout()

        self.x_label = QLabel("X:")
        self.x_input = QLineEdit()
        self.coordinates_layout.addWidget(self.x_label)
        self.coordinates_layout.addWidget(self.x_input)

        self.y_label = QLabel("Y:")
        self.y_input = QLineEdit()
        self.coordinates_layout.addWidget(self.y_label)
        self.coordinates_layout.addWidget(self.y_input)

        layout.addLayout(self.coordinates_layout)

        # Add node button
        self.add_button = QPushButton("Add Node")
        self.add_button.clicked.connect(self.add_node)
        layout.addWidget(self.add_button)

        # Display nodes
        self.node_display = QTextEdit()
        self.node_display.setReadOnly(True)
        layout.addWidget(self.node_display)

        # Save nodes button
        self.save_button = QPushButton("Save Nodes")
        self.save_button.clicked.connect(self.save_nodes)
        layout.addWidget(self.save_button)

        # Container for the layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def add_node(self):
        try:
            # Generate a default Node ID if none is provided
            node_id = self.node_id_input.text().strip() or f"n{self.node_counter}"

            # Check if the Node ID is already used
            if any(node["id"] == node_id for node in self.nodes):
                raise ValueError(f"Node ID '{node_id}' is already in use.")

            # Get x and y coordinates
            x = float(self.x_input.text())
            y = float(self.y_input.text())

            # Add node to the list
            node = {"id": node_id, "coordinates": [x, y]}
            self.nodes.append(node)

            # Update the display
            self.node_display.append(f"Node ID: {node_id}, Coordinates: ({x}, {y})")

            # Clear input fields and increment the node counter
            self.node_id_input.clear()
            self.x_input.clear()
            self.y_input.clear()
            self.node_counter += 1

        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))

    def save_nodes(self):
        if not self.nodes:
            QMessageBox.warning(self, "No Nodes", "No nodes to save!")
            return

        # Save nodes to a file or perform other actions
        QMessageBox.information(self, "Nodes Saved", f"Saved {len(self.nodes)} nodes!")

        # For demonstration, print the nodes
        print("Nodes:", self.nodes)


# Run the application
if __name__ == "__main__":
    app = QApplication([])
    window = NodeInputForm2D()
    window.show()
    app.exec()
