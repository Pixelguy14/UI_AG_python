from PyQt5.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
                            QTableView, QHeaderView, QToolBar, QAction, QDialog, QFormLayout, 
                            QLineEdit, QDateEdit, QComboBox, QMessageBox, QLabel, QSplitter,
                            QDialogButtonBox, QApplication, QFrame, QGridLayout, QSizePolicy, 
                            QStackedWidget, QSpinBox)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QDate, QSize
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel, QSqlQueryModel
from datetime import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore
import pandas as pd
from src.functions.exploratory_data import *

# PyQT model that manages columns and rows in a pandas Dataframe
class PandasModel(QAbstractTableModel): 
    def __init__(self, df = pd.DataFrame(), parent=None, decimals=4): 
        QAbstractTableModel.__init__(self, parent=parent)
        self._df = df.copy()
        self._italic_columns = set()
        self._decimal_places = decimals

    def rename_columns(self, rename_dict, italic_cols=None):
        # Check if all old column names exist in the DataFrame
        missing_cols = [col for col in rename_dict.keys() if col not in self._df.columns]
        if missing_cols:
            print(f"Warning: Columns not found in DataFrame: {missing_cols}")
            return False
        # Rename columns in the DataFrame
        self.beginResetModel()  # Notify Qt that the model is about to change
        self._df.rename(columns=rename_dict, inplace=True)
        self.endResetModel()    # Notify Qt that the model has changed
        # Track italic columns
        if italic_cols:
            self._italic_columns.update(italic_cols)
        
        return True

    def toDataFrame(self):
        return self._df.copy()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                # The crucial check for horizontal headers
                if section < len(self._df.columns): # Ensure the section index is within bounds
                    return self._df.columns[section]
                return None # Return None if out of bounds, so Qt doesn't display it
            elif orientation == Qt.Vertical:
                # For vertical headers, display the DataFrame's index
                if section < len(self._df.index): # Ensure the section index is within bounds
                    return str(self._df.index[section]) # Convert index label to string
                return None
        elif role == Qt.FontRole:
            if orientation == Qt.Horizontal:
                col_name = self._df.columns[section]
                if col_name in self._italic_columns:
                    font = QFont()
                    font.setItalic(True)
                    return font
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QtCore.QVariant()

        if role == Qt.DisplayRole:
            try:
                value = self._df.iloc[index.row(), index.column()]

                # Check if the value is a number (integer or float)
                if isinstance(value, (int, float)):
                    # Threshold for applying scientific notation
                    if (value != 0 and (abs(value) < 0.001 or abs(value) >= 10000)):
                        return QtCore.QVariant(f"{value:.{self._decimal_places}e}")
                    else:
                        return QtCore.QVariant(f"{value:.{self._decimal_places}f}")
                else:
                    return QtCore.QVariant(str(value))

            except Exception as e:
                print(f"Error getting table data: {e}")
                return QtCore.QVariant()
        return QtCore.QVariant()

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt5 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QModelIndex()): 
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()): 
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

# Plot prototype for Quality Control model
class PlotWidgetQC(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clear_plot(self):
        self.figure.clear()
        self.ax = None
        self.canvas.draw()

    def plot_bar_chart(self, mean_tic_data, column_names, N):
        ax = self.figure.add_subplot(111)
        ax.clear()

        ind = np.arange(N)
        plot_values = mean_tic_data[:N]
        plot_labels = column_names[:N]

        ax.bar(ind, plot_values, color="#3156A1", alpha=0.4)
        ax.set_ylabel("Mean log2 intensity")
        ax.set_title(f"{N} samples")
        ax.set_xticks(ind)
        ax.set_xticklabels(plot_labels, rotation=-90, ha='left') # 'ha' for horizontal alignment
        ax.tick_params(axis='x', labelsize=8)

        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_line_data(self, data):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(data, '*-')
        self.figure.tight_layout()
        self.canvas.draw()
