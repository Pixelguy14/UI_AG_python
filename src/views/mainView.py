from PyQt5.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
                            QTableView, QHeaderView, QToolBar, QAction, QDialog, QFormLayout, 
                            QLineEdit, QDateEdit, QComboBox, QMessageBox, QLabel, QSplitter,
                            QDialogButtonBox, QApplication, QFrame, QGridLayout, QSizePolicy, 
                            QStackedWidget, QSpinBox, QTabWidget, QTextEdit, QDoubleSpinBox, 
                            QAbstractScrollArea)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QDate, QSize
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor, QStandardItemModel
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel, QSqlQueryModel
import xdialog
from datetime import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import threading
from PyQt5 import QtCore
import pandas as pd

from src.functions.exploratory_data import *
from src.models.pandas_tablemodel import *

class mainView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        emptyDF = pd.DataFrame()
        
        self.colModel = PandasModel(emptyDF)
        self.summaryModel = PandasModel(emptyDF)
        self.DfModel = PandasModel(emptyDF)

        self.old_column = ""
    
    def initUI(self):
        self.setWindowTitle('Main Window')
        #self.setFixedSize(500, 600)
        self.showMaximized()
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #3156A1;
                font-family: Arial;
            }
            QTableView {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                alternate-background-color: #f9f9f9;
                selection-background-color: #94a7cb;
                selection-color: white;
            }
            QTableView::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
            QHeaderView::section {
                background-color: #94a7cb;
                color: white;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
            QPushButton {
                padding: 8px 16px;
                background-color: #94a7cb;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a68be;
            }
            QPushButton:pressed {
                background-color: #1f3868;
            }
            QPushButton#deleteBtn {
                background-color: #dc3545;
            }
            QPushButton#deleteBtn:hover {
                background-color: #9c1b28;
            }
            QToolBar {
                background-color: #94a7cb;
                spacing: 10px;
                padding: 5px;
                border: none;
            }
            QToolBar QToolButton {
                background-color: transparent;
                color: white;
                border: none;
                padding: 5px;
                font-weight: bold;
            }
            QToolBar QToolButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
                border-radius: 4px;
            }
            QLabel#titleLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
            QFrame#chartFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)

        # Top toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setIconSize(QSize(32, 32))
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        # Toolbar title
        title_label = QLabel("Genomics GUI")
        title_label.setObjectName("titleLabel")
        self.toolbar.addWidget(title_label)

        self.toolbar.addSeparator()

        # Action button for the toolbar
        self.btn_loadOpenms = QPushButton("Load OpenMS")
        self.btn_loadOpenms.setCursor(Qt.PointingHandCursor)
        self.btn_loadOpenms.clicked.connect(
            #lambda: self.setDFModel("/home/pixel/Documents/Cinvestav_2025/UI_Analisis_Genomico/Consensus/normalized.consensusXML")
            lambda: self.setDFModel("Openms")
        )
        self.toolbar.addWidget(self.btn_loadOpenms)

        self.btn_loadXcms = QPushButton("Load XCMS")
        self.btn_loadXcms.setCursor(Qt.PointingHandCursor)
        self.btn_loadXcms.clicked.connect(
            lambda: self.setDFModel("XCMS")
        )
        self.toolbar.addWidget(self.btn_loadXcms)

        self.toolbar.addSeparator()

        spacer = QWidget()

        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        spacer.setStyleSheet("background-color: transparent;")
        self.toolbar.addWidget(spacer)
        
        # button to exit
        self.logout_btn = QPushButton("Exit")
        self.logout_btn.setCursor(Qt.PointingHandCursor)
        self.logout_btn.clicked.connect(self.logout)
        self.toolbar.addWidget(self.logout_btn)

        # Main tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.West)
        
        self.summary_tab = QWidget()
        self.setupSummaryTab()
        self.tab_widget.addTab(self.summary_tab, "Summary view")
        
        self.df_tab = QWidget()
        self.setupDFTab()
        self.tab_widget.addTab(self.df_tab, "DataFrame View")
        
        self.setCentralWidget(self.tab_widget)

    def setupSummaryTab(self):
        main_widget = QStackedWidget()
        main_widget.setStyleSheet("""
            QStackedWidget::handle {
                background-color: #94a7cb;
                height: 4px;
                border-radius: 2px;
            }
        """)

    def setupDFTab(self):
        # Create main grid layout
        self.DFgrid = QGridLayout(self)
        
        # Create a vertical layout for the table section
        table_container = QVBoxLayout()
        
        # Title
        title = QLabel("DataFrame Table")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        table_container.addWidget(title)
        
        # Main DataFrame table
        self.DF_table = QTableView()
        self.DF_table.setAlternatingRowColors(True)
        self.DF_table.setSelectionBehavior(QTableView.SelectItems)  # Changed to SelectItems
        self.DF_table.setEditTriggers(QTableView.NoEditTriggers)
        
        # Configure header
        header = self.DF_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(70)
        header.setDefaultSectionSize(150)
        header.setMaximumSectionSize(300)
        header.sectionClicked.connect(self.onColumnDFClicked)  # Connect column header clicks
        
        # Configure table behavior
        self.DF_table.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        self.DF_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.DF_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.DF_table.verticalHeader().setVisible(False)
        self.DF_table.verticalHeader().setDefaultSectionSize(50)
        self.DF_table.setSortingEnabled(True)
        
        # Set the PandasModel
        self.DfModel = PandasModel()
        self.DF_table.setModel(self.DfModel)
        table_container.addWidget(self.DF_table)
        
        # Add instruction label
        self.instruction_label = QLabel("Click a column header to display column data")
        self.instruction_label.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        table_container.addWidget(self.instruction_label)
        
        #table_container.addWidget(self.column_table)
        
        # Add table container to grid
        self.DFgrid.addLayout(table_container, 0, 0, 5, 5)  # row, col, rowSpan, colSpan
        
        # Create and add plot widgets
        self.plot_widgets = []
        plot_definition = {'row': 0, 'col': 5, 'rowSpan': 2, 'colSpan': 2}
        
        self.DF_QC_bar_plot_widget = PlotWidgetQC(self.df_tab)
        self.DF_QC_bar_plot_widget.setMinimumHeight(300)
        self.DF_QC_bar_plot_widget.setMinimumWidth(300)
        self.plot_widgets.append(self.DF_QC_bar_plot_widget)
        
        self.DFgrid.addWidget(self.DF_QC_bar_plot_widget, 
                            plot_definition['row'],
                            plot_definition['col'],
                            plot_definition['rowSpan'],
                            plot_definition['colSpan'])
        
        # Create a vertical layout for the table section
        coltable_container = QVBoxLayout()
        
        # Title
        self.titleColinfo = QLabel("Column Information")
        self.titleColinfo.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        self.titleColinfo.setVisible(False)  # Initially hidden
        coltable_container.addWidget(self.titleColinfo)
        
        # Create invisible table for column data
        self.column_table = QTableView()
        self.column_table.setVisible(False)  # Initially hidden
        self.column_table.setAlternatingRowColors(True)
        self.column_table.setEditTriggers(QTableView.NoEditTriggers)
        self.column_table.verticalHeader().setVisible(False)
        self.column_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create model for column data
        self.colModel = QStandardItemModel()
        self.column_table.setModel(self.colModel)

        coltable_container.addWidget(self.column_table)

        self.DFgrid.addLayout(coltable_container, 2, 5, 3, 2) # row, col, rowSpan, colSpan
        
        self.DFgrid.setColumnStretch(0, 5)  # Table area fixed size
        self.DFgrid.setColumnStretch(5, 2)   # Plot area fixed size

        self.df_tab.setLayout(self.DFgrid)
    
    def setDFModel(self, interpreterType):
        self.DF_QC_bar_plot_widget.clear_plot()

        file_path = xdialog.open_file(
            title="Select a compatible file", 
            filetypes=[
                ("Consensus XML Files", "*.consensusXML"),
                ("Feature XML Files", "*.featureXML"),
                ("XML Files", "*.xml"),
                ("Tab-Separated Values", "*.tsv"),
                ("Comma-Separated Values", "*.csv"),
                ("Excel files", "*.xlsx;*.xls")
                ],
            multiple=False,
            #start_dir=current_directory,
            )
        if file_path == "" or file_path == None:
            print("no path file loaded")
            return None
        # Load the dataframe
        print(f"Loading data from: {file_path}")
        df = loadFile(file_path)

        if df.empty or df is None:
            print("Failed to load data or no compatible data could build the dataframe from the selected file.")
            return None
        
        # --- Start of Column Renaming Logic ---
        # Create a copy of the current columns to iterate safely
        original_columns = list(df.columns)
        rename_map_paths = {}

        for col_name in original_columns:
            if '/' in col_name or '\\' in col_name:
                # Use os.path.basename to get the last part of the path
                # This handles both forward and backward slashes correctly
                new_col_name = os.path.basename(col_name)
                if new_col_name != col_name: # Only add to map if a change is needed
                    rename_map_paths[col_name] = new_col_name
        
        # Apply path-based renames first, if any
        if rename_map_paths:
            df.rename(columns=rename_map_paths, inplace=True)
            print(f"Renamed path-like columns in the dataframe")

        # Create or update the PandasModel
        if not hasattr(self, 'DfModel'):
            self.DfModel = PandasModel(df)
        else:
            # If model exists, update its data
            self.DfModel._df = df.copy()
            self.DfModel.layoutChanged.emit()  # Notify views of data change
            #print(type(self.DfModel))
        self.DfModel.rename_columns({
            "rt": "RT",
            "mz": "m/z",
            "intensity": "Intensity"
        }, italic_cols=["m/z"])
        
        # Set the model to the table view
        self.DF_table.setModel(self.DfModel)

        # Reapply header settings (essential after model change)
        header = self.DF_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(100)
        header.setDefaultSectionSize(150)
        header.setMaximumSectionSize(300)        

        # Process data for plotting
        if interpreterType == "Openms":
            numeric_cols = df.iloc[:, 3:].select_dtypes(include=np.number).columns
        elif interpreterType == "XCMS":
            numeric_cols = df.iloc[:, 12:].select_dtypes(include=np.number).columns
        new_data = df[numeric_cols]
        mean_TIC = new_data.mean(axis=0) # Calculate mean along columns (axis=0)

        # --- Call the specific bar plot method on the correct PlotWidgetQC instance ---
        self.DF_QC_bar_plot_widget.plot_bar_chart(mean_TIC, new_data.columns, numeric_cols.shape[0])

        # Force layout update
        self.DFgrid.update()
        self.df_tab.updateGeometry()

        #self.colDF = preprocessing_summary_perVariable(df)
        #print(self.colDF)
        #summaryDF = preprocessing_general_dataset_statistics(df)
        ## eventually, turn summaryDF to summaryModel with pandasmodel to show in a table.

        print("Data successfully loaded and displayed")

    def onColumnDFClicked(self, column_index):
        # Get column name from index
        col_name = self.DfModel._df.columns[column_index]
        if self.old_column == col_name:
            return
        #print(f"Column at index {column_index}: {col_name}")
        
        # Create a DataFrame containing just this column
        cutDF = self.DfModel._df[[col_name]]
        # Create the preprocessing for just colDF
        colDF = preprocessing_summary_perVariable(cutDF)
        #print(colDF)
        #print(type(colDF))
        
        # Create a transposed view for vertical display
        transposed_df = colDF.T
        #print(transposed_df)
        
        # Update the column table model
        self.colModel = PandasModel(transposed_df)
        self.column_table.setModel(self.colModel)
        
        # Adjust column table settings
        self.column_table.verticalHeader().setVisible(True)
        self.column_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Hide instruction and show column table
        #self.instruction_label.setVisible(False)
        self.titleColinfo.setVisible(True)
        self.column_table.setVisible(True)

        self.old_column = col_name

    def logout(self):
        self.close()
