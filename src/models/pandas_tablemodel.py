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

# Plot prototype for Quality Control models
class PlotWidgetQC(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(layout='constrained')
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.last_width = 0
        self.last_height = 0

    def calculate_font_sizes(self, num_items=0):
        """Calculate dynamic font sizes based on plot dimensions and data density"""
        # Get current figure dimensions in inches
        fig_width, fig_height = self.figure.get_size_inches()
        
        # Calculate base font size based on figure area
        area = fig_width * fig_height
        base_size = max(8, min(12, 8 * (area / 30)))  # Constrain between 6-12
        
        # Adjust for number of items
        if num_items > 0:
            density_factor = min(1.0, 30 / num_items)  # Reduce font for dense data
            base_size *= density_factor
        
        return {
            'title': base_size * 1.2,
            'axis': base_size,
            'ticks': base_size * 0.8,
            'annotations': base_size * 0.7
        }

    def clear_plot(self):
        self.figure.clear()
        self.canvas.draw()

    def plot_missing_heatmap(self, df):
        """Plot heatmap showing missing values distribution"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Create heatmap
        sns.heatmap(df.isnull(), cbar=False, cmap='binary_r', ax=ax)
        ax.set_title('Missing Values Distribution', fontsize=10)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.invert_yaxis()  # Invert Y-axis
        self.canvas.draw()

    def plot_data_types_distribution(self, summary):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Prepare data
        type_counts = summary['type'].value_counts()
        # Get dynamic font sizes
        num_categories = len(type_counts)
        font_sizes = self.calculate_font_sizes(num_categories)
        
        # Create bar plot
        sns.barplot(x=type_counts.index, y=type_counts.values, hue=type_counts.index, legend=False, palette='mako', ax=ax)
        ax.set_ylim(0, type_counts.max() * 1.1)
        ax.set_yticks([0, type_counts.max()])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_title('Data Types Distribution', fontsize=font_sizes['title'])
        ax.set_xlabel('Data Type', fontsize=font_sizes['axis'])
        ax.set_ylabel('Count', fontsize=font_sizes['axis'])

        ax.set_xticks(range(len(type_counts.index)))
        ax.set_xticklabels(type_counts.index, ha='right', fontsize=6)
        ax.set_xticklabels(
            type_counts.index, 
            ha='right', 
            fontsize=font_sizes['ticks'],
            rotation=45 if any(len(str(x)) > 10 for x in type_counts.index) else 0
        )
        self.canvas.draw()

    def plot_correlation_matrix(self, df):
        """Plot correlation matrix for numerical columns"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Get numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        num_cols = numerical_df.shape[1] if not numerical_df.empty else 0

        # Get dynamic font sizes
        font_sizes = self.calculate_font_sizes(num_cols)
        
        if not numerical_df.empty and numerical_df.shape[1] > 1:
            corr_matrix = numerical_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='viridis', 
                        fmt=".2f", linewidths=.5, ax=ax, annot_kws={"fontsize": font_sizes["ticks"]})
            ax.set_title('Correlation Matrix', fontsize=font_sizes["title"])
            ax.tick_params(axis='x', labelsize=font_sizes["axis"])
            ax.tick_params(axis='y', labelsize=font_sizes["axis"])
        else:
            ax.text(0.5, 0.5, "Not enough numerical columns", 
                    ha='center', va='center', fontsize=font_sizes["axis"])
            ax.set_title('Correlation Matrix', fontsize=font_sizes["axis"])
        self.canvas.draw()

    def plot_null_pie(self, df):
        """Plot pie chart showing null vs non-null values"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Calculate values
        total_elements = df.size
        total_nulls = df.isnull().sum().sum()
        font_sizes = self.calculate_font_sizes(2)
        
        if total_elements > 0:
            total_non_nulls = total_elements - total_nulls
            labels = ['Null Values', 'Non-Null Values']
            sizes = [total_nulls, total_non_nulls]
            colors = ['#ff9999', '#66b3ff']
            explode = (0.1, 0)  # Explode null slice
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90,
                   textprops={'fontsize': font_sizes["axis"]}, # Reduced label font size
                   pctdistance=0.8) # Distance of percentage from center
            ax.set_title('Percentaje of Null Values',fontsize=font_sizes["title"])
            ax.axis('equal')
        else:
            ax.text(0.5, 0.5, "DataFrame is empty", 
                    ha='center', va='center', fontsize=font_sizes["ticks"])
            ax.set_title('Null Distribution', fontsize=font_sizes["ticks"])
        self.canvas.draw()
    
    def plot_bar_chart(self, mean_tic_data, column_names, N):
        ax = self.figure.add_subplot(111)
        ax.clear()

        # Get dynamic font sizes
        font_sizes = self.calculate_font_sizes(N)

        ind = np.arange(N)
        plot_values = mean_tic_data[:N]
        plot_labels = column_names[:N]

        ax.bar(ind, plot_values, color="#3156A1")
        ax.set_ylabel("Mean log2 intensity", fontsize=font_sizes['axis'])
        ax.set_title(f"{N} samples", fontsize=font_sizes['title'])
        ax.set_xticks(ind)
        
        # Dynamic label rotation based on label length
        max_label_len = max(len(str(label)) for label in plot_labels)
        rotation = 90 if max_label_len > 15 or N > 20 else 45 if N > 10 else 0
        
        ax.set_xticklabels(
            plot_labels, 
            rotation=rotation, 
            ha='right' if rotation else 'center',
            fontsize=font_sizes['ticks']
        )
        self.canvas.draw()
    
    def plot_line_data(self, data):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(data, '*-')
        self.figure.tight_layout()
        self.canvas.draw()
