from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QPushButton, QLabel, QLineEdit, QTextEdit, QSizePolicy)
from PyQt5.QtCore import Qt

class ResponsiveGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Main Window')
        
        self.grid = QGridLayout(self)
        
        # Set spacing and margins
        self.grid.setSpacing(10)
        self.grid.setContentsMargins(15, 15, 15, 15)
        
        # Create widgets with different sizes
        self.create_widgets()
        
        # Make layout responsive
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def create_widgets(self):
        # Widget 1 - Tall button (span 2 rows)
        btn1 = QPushButton("Tall Button")
        btn1.setMinimumHeight(150)
        self.grid.addWidget(btn1, 0, 0, 1, 4)  # row, col, rowSpan, colSpan
        
        # Widget 2 - Wide text field
        text1 = QLineEdit()
        text1.setPlaceholderText("Wide field")
        self.grid.addWidget(text1, 0, 5, 1, 2)  # spans 2 columns
        
        # Widget 3 - Small square button
        text1 = QLineEdit()
        text1.setPlaceholderText("Wide field")
        self.grid.addWidget(text1, 1, 5, 1, 2)  # spans 2 columns

if __name__ == '__main__':
    app = QApplication([])
    window = ResponsiveGrid()
    window.show()
    app.exec_()