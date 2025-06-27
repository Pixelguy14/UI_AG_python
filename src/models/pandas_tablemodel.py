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
from matplotlib.colors import ListedColormap
from PyQt5 import QtCore
import pandas as pd
from src.functions.exploratory_data import *

# PyQT model that manages columns and rows in a pandas Dataframe
class PandasModel(QAbstractTableModel): 
    def __init__(self, df=pd.DataFrame(), parent=None, decimals=4, italic_cols=None):
        QAbstractTableModel.__init__(self, parent=parent)
        self._df = df.copy()
        # Initialize _italic_columns from constructor argument or as an empty set
        self._italic_columns = set(italic_cols) if italic_cols else set()
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
        # Get current dimensions
        col_count = self.columnCount()
        row_count = self.rowCount()

        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if section < col_count:
                    return self._df.columns[section]
                return None
            elif orientation == Qt.Vertical:
                if section < row_count:
                    return str(self._df.index[section])
                return None
        elif role == Qt.FontRole:
            if orientation == Qt.Horizontal:
                if section < col_count:
                    col_name = self._df.columns[section]
                    if col_name in self._italic_columns:
                        font = QFont()
                        font.setItalic(True)
                        return font
        return None
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._df.empty: # Use .empty for clearer check
            return QtCore.QVariant()

        row = index.row()
        col = index.column()

        # Check bounds
        if row >= self.rowCount() or col >= self.columnCount():
            return None

        if role == Qt.DisplayRole:
            try:
                value = self._df.iloc[row, col] # Use row and col directly

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
                try: # Add a try-except for type conversion errors
                    value = None if value == '' else dtype.type(value)
                except ValueError:
                    print(f"Warning: Could not convert '{value}' to {dtype} for column '{col}'")
                    return False # Indicate that data was not set
        self._df.loc[row, col] = value # Use .loc for setting by label
        self.dataChanged.emit(index, index, [role]) # Emit dataChanged signal
        return True
    
    def rowCount(self, parent=QModelIndex()): 
        return len(self._df.index) if self._df is not None else 0

    def columnCount(self, parent=QModelIndex()): 
        return len(self._df.columns) if self._df is not None else 0

    def sort(self, column, order):
        if self._df.empty: # Check if DataFrame is empty before sorting
            return
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending=order == Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

    def reset_model(self, new_data):
        """Completely reset the model with new data"""
        self.beginResetModel()
        self._df = new_data.copy()
        # Ensure _italic_columns is reset or re-initialized based on new_data if needed
        self._italic_columns = set() # Or derive from new_data if applicable
        self.endResetModel()

    def remove_columns(self, columns_to_remove):
        """Safely remove columns with proper model notifications"""
        if not columns_to_remove:
            return

        # Find positions of columns to remove
        col_positions = []
        current_columns = self._df.columns.tolist() # Get current columns as a list
        for col in columns_to_remove:
            if col in current_columns:
                pos = current_columns.index(col)
                col_positions.append(pos)

        if not col_positions:
            return

        # Sort in reverse order to remove from highest index first,
        # which prevents index shifting issues during removal
        col_positions.sort(reverse=True)

        # Notify view of column removal and remove from DataFrame
        for pos in col_positions:
            self.beginRemoveColumns(QModelIndex(), pos, pos)
            # Remove by position using .iloc to avoid issues with duplicate column names
            self._df = self._df.drop(self._df.columns[pos], axis=1)
            self.endRemoveColumns()

        # Emit layoutChanged to ensure views are fully updated after multiple removals
        self.layoutChanged.emit()
