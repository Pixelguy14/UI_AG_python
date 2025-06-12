import sys
import random
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QPushButton,
                            QSizePolicy, QVBoxLayout, QLabel)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_data(self, data):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(data, '*-')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def clear_plot(self):
        self.figure.clear()
        self.ax = None
        self.canvas.draw()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Multi-Plot Grid Application')

        self.grid = QGridLayout(self)
        self.grid.setSpacing(10)
        self.grid.setContentsMargins(15, 15, 15, 15)

        self.plot_widgets = []
        self.num_plots = 5  # Let's say we want 5 plots

        self.create_widgets()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def create_widgets(self):
        # Main "Plot All" button
        self.plot_all_button = QPushButton('Generate All Plots')
        self.plot_all_button.clicked.connect(self.generate_all_plots)
        self.grid.addWidget(self.plot_all_button, 0, 0, 1, 2)  # Spans 2 columns

        # Create multiple plot widgets and add them to the grid
        plot_definitions = [
            {'row': 1, 'col': 0, 'rowSpan': 1, 'colSpan': 1}, # Plot 1
            {'row': 1, 'col': 1, 'rowSpan': 1, 'colSpan': 1}, # Plot 2
            {'row': 2, 'col': 0, 'rowSpan': 1, 'colSpan': 1}, # Plot 3
            {'row': 2, 'col': 1, 'rowSpan': 1, 'colSpan': 1}, # Plot 4
            {'row': 1, 'col': 2, 'rowSpan': 2, 'colSpan': 1}  # Plot 5 (spans 2 rows)
        ]
        for i in range(self.num_plots):
            plot_widget = PlotWidget(self)
            self.plot_widgets.append(plot_widget)
            definition = plot_definitions[i]
            self.grid.addWidget(plot_widget, definition['row'], definition['col'],
                                definition['rowSpan'], definition['colSpan'])

        # You can add other widgets here if you wish, similar to your original ResponsiveGrid
        # For example, a status label
        self.status_label = QLabel("Ready to plot!")
        self.grid.addWidget(self.status_label, 0, 2, 1, 1) # Example position

    def generate_all_plots(self):
        self.status_label.setText("Generating plots...")
        for plot_widget in self.plot_widgets:
            plot_widget.clear_plot()
            data = [random.random() for _ in range(10)]
            plot_widget.plot_data(data)
        self.status_label.setText("Plots generated!")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())