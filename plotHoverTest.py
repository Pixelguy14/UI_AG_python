import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QToolTip
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QBuffer, QIODevice
import io

class PlotTooltipWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.layout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.layout.addWidget(self.label)
        self.setFixedSize(200, 150) # Adjust size as needed
        self.setStyleSheet("background-color: lightyellow; border: 1px solid gray;")

    def set_plot_data(self, plot_figure):
        buf = io.BytesIO()
        plot_figure.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        image = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matplotlib Plot in Tooltip")
        self.setGeometry(100, 100, 400, 300)

        self.button = QPushButton("Hover over me for a plot", self)
        self.button.move(100, 100)
        self.button.setFixedSize(200, 50)
        self.button.setMouseTracking(True) # Enable mouse tracking for hover events

        self.tooltip_widget = PlotTooltipWidget()
        self.button.installEventFilter(self) # Install event filter on the button

        self.plot_figure = self.create_plot()

    def create_plot(self):
        fig, ax = plt.subplots(figsize=(2, 1.5)) # Smaller figure for tooltip
        ax.plot([0, 1, 2, 3], [0, 1, 4, 9])
        ax.set_title("Mini Plot")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.tight_layout()
        return fig

    def eventFilter(self, obj, event):
        if obj == self.button:
            if event.type() == event.HoverEnter:
                self.tooltip_widget.set_plot_data(self.plot_figure)
                # Position the tooltip relative to the button
                pos = self.button.mapToGlobal(self.button.rect().bottomLeft())
                self.tooltip_widget.move(pos)
                self.tooltip_widget.show()
                return True
            elif event.type() == event.HoverLeave:
                self.tooltip_widget.hide()
                return True
        return super().eventFilter(obj, event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())