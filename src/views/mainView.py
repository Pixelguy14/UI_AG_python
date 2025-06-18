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
from src.models.metadata_dialogmodel import *
from src.models.pandas_tablemodel import *

class mainView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        emptyDF = pd.DataFrame()
        
        self.colModel = PandasModel(emptyDF) # Model with the DF data of statistical data for a column
        self.summaryModel = PandasModel(emptyDF) # Model with the DF data of statistical data summary 
        self.DfModel = PandasModel(emptyDF) # Main Model for the loaded DF

        self.DfMetadata = PandasModel(emptyDF) # Model of the metadata from the Main Df
        self.DfSample = PandasModel(emptyDF) # Model of the sample data from the Main Df

        self.DfReset = PandasModel(emptyDF) # Incase you'll want to reset the Df to before any changes.

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
        self.btn_loadOpenms = QPushButton("Load File")
        self.btn_loadOpenms.setCursor(Qt.PointingHandCursor)
        self.btn_loadOpenms.clicked.connect(
            #lambda: self.setDFModel("/home/pixel/Documents/Cinvestav_2025/UI_Analisis_Genomico/Consensus/normalized.consensusXML")
            lambda: self.setDFModel()
        )
        self.toolbar.addWidget(self.btn_loadOpenms)

        self.btn_defineMD = QPushButton("Define Metadata")
        self.btn_defineMD.setCursor(Qt.PointingHandCursor)
        #self.btn_defineMD.clicked.connect(lambda: self.setDFModel())
        self.btn_defineMD.clicked.connect(lambda: self.viewMetadataModel())
        self.toolbar.addWidget(self.btn_defineMD)

        self.toolbar.addSeparator()

        spacer = QWidget()

        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        spacer.setStyleSheet("background-color: transparent;")
        self.toolbar.addWidget(spacer)

        # button to reset the dataframe to it's original state
        self.logout_btn = QPushButton("Reset Dataframe")
        self.logout_btn.setCursor(Qt.PointingHandCursor)
        self.logout_btn.clicked.connect(self.resetOriginal)
        self.toolbar.addWidget(self.logout_btn)
        
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

        self.imputation_tab = QWidget()
        self.setupImputationTab()
        self.tab_widget.addTab(self.imputation_tab, "Imputation View")
        
        self.setCentralWidget(self.tab_widget)

    def setupSummaryTab(self):
        self.Plotgrid = QGridLayout()

        # Create a vertical layout for the plot section
        container1 = QVBoxLayout()
        self.Log2_plotWidget = PlotWidgetQC(self.summary_tab)
        container1.addWidget(self.Log2_plotWidget)
        # Add container to grid
        self.Plotgrid.addLayout(container1, 0, 0, 3, 2) # row, col, rowSpan, colSpan

        # Create a vertical layout for the plot section
        container2 = QVBoxLayout()
        self.missing_heatmap_widget = PlotWidgetQC(self.summary_tab)
        container2.addWidget(self.missing_heatmap_widget)
        # Add container to grid
        self.Plotgrid.addLayout(container2, 3, 0, 4, 2) # row, col, rowSpan, colSpan

        # Create a vertical layout for the plot section
        container3 = QVBoxLayout()
        # Create invisible table for general data
        self.general_data_table = QTableView()
        self.general_data_table.setVisible(False)  # Initially hidden
        self.general_data_table.setAlternatingRowColors(True)
        self.general_data_table.setEditTriggers(QTableView.NoEditTriggers)
        self.general_data_table.verticalHeader().setVisible(False)
        #self.general_data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create model for general data
        self.colGeneral = QStandardItemModel()
        self.general_data_table.setModel(self.colGeneral)

        container3.addWidget(self.general_data_table)
        # Add container to grid
        self.Plotgrid.addLayout(container3, 7, 0, 2, 2) # row, col, rowSpan, colSpan

        # Create a vertical layout for the plot section
        container4 = QVBoxLayout()
        self.correlation_widget = PlotWidgetQC(self.summary_tab)
        container4.addWidget(self.correlation_widget)
        # Add container to grid
        self.Plotgrid.addLayout(container4, 0, 2, 6, 4) # row, col, rowSpan, colSpan

        # Create a vertical layout for the plot section
        container4 = QVBoxLayout()
        self.data_types_widget = PlotWidgetQC(self.summary_tab)
        container4.addWidget(self.data_types_widget)
        # Add container to grid
        self.Plotgrid.addLayout(container4, 6, 2, 3, 2) # row, col, rowSpan, colSpan
        
        # Create a vertical layout for the plot section
        container5 = QVBoxLayout()
        # Title
        title = QLabel("leftover space")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        container5.addWidget(title)
        # Add container to grid
        self.Plotgrid.addLayout(container5, 6, 4, 3, 2) # row, col, rowSpan, colSpan
        
        self.Plotgrid.setColumnStretch(0, 3) # col, stretch
        self.Plotgrid.setColumnStretch(1, 3)
        self.Plotgrid.setColumnStretch(2, 2)
        self.Plotgrid.setColumnStretch(3, 2)
        self.Plotgrid.setColumnStretch(4, 2)
        self.Plotgrid.setColumnStretch(5, 2)
        # 3+3+2+2+2+2=14
        # first 2 cols: 6/14, rest: 8/14

        self.Plotgrid.setRowStretch(0, 3) # row, stretch
        self.Plotgrid.setRowStretch(1, 3) 
        self.Plotgrid.setRowStretch(2, 3) 
        self.Plotgrid.setRowStretch(3, 4) 
        self.Plotgrid.setRowStretch(4, 4) 
        self.Plotgrid.setRowStretch(5, 4) 
        self.Plotgrid.setRowStretch(6, 3) 
        self.Plotgrid.setRowStretch(7, 1)
        #self.Plotgrid.setRowStretch(8, 2)
        # 3+3+3+4+4+4+3+1=25
        # first 3: 9/25 second 3: 12/25 rest: 3/25 rest: 1/25

        self.summary_tab.setLayout(self.Plotgrid)

    def setupDFTab(self):
        # Create main grid layout
        self.DFgrid = QGridLayout()
        
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
        
        # Add table container to grid
        self.DFgrid.addLayout(table_container, 0, 0, 5, 5)  # row, col, rowSpan, colSpan
        
        null_distribution_container = QVBoxLayout()
        self.null_distribution_widget = PlotWidgetQC(self.df_tab)
        #self.null_distribution_widget.setMinimumHeight(300)
        #self.null_distribution_widget.setMinimumWidth(300)
        null_distribution_container.addWidget(self.null_distribution_widget)
        self.DFgrid.addLayout(null_distribution_container, 0, 5, 2, 2)

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
    
    def setupImputationTab(self):
        self.ImputationGrid = QGridLayout()
        self.imputation_tab.setLayout(self.ImputationGrid)

    def setDFModel(self):

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
    
        self.Log2_plotWidget.clear_plot()
        self.missing_heatmap_widget.clear_plot()
        self.correlation_widget.clear_plot()
        self.null_distribution_widget.clear_plot()
        self.titleColinfo.setVisible(False)
        self.column_table.setVisible(False)
        self.old_column = ""
        
        # Start of Column Renaming Logic
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
            #print(f"Renamed path-like columns in the dataframe")

        # Create or update the PandasModel
        if not hasattr(self, 'DfModel'):
            self.DfModel = PandasModel(df)
        else:
            # If model exists, update its data and notify views of data change
            self.DfModel.reset_model(df)
            #self.DfModel._df = df.copy()
            self.DfModel.layoutChanged.emit()
            #print(type(self.DfModel))
        # We assign the initial model to a copy for resetting in this state
        self.DfReset._df = self.DfModel._df
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

        #self.loadLogSamplePlot()
        self.loadGeneralDataTable()

        self.data_types_widget.plot_data_types_distribution(preprocessing_summary_perVariable(self.DfModel._df))

        print("Data successfully loaded and displayed")

    def loadGeneralDataTable(self):
        summaryDF = preprocessing_general_dataset_statistics(self.DfModel._df)
        # Update the column table model
        self.colGeneral = PandasModel(summaryDF.T)
        self.general_data_table.setModel(self.colGeneral)

        header = self.general_data_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(300)
        header.setDefaultSectionSize(300)
        header.setMaximumSectionSize(500) 
        
        # Adjust column table settings
        self.general_data_table.verticalHeader().setVisible(True)
        self.general_data_table.horizontalHeader().setVisible(False)
        self.general_data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Hide instruction and show column table
        self.general_data_table.setVisible(True)

    def loadLogSamplePlot(self):
        self.Log2_plotWidget.clear_plot()
        self.missing_heatmap_widget.clear_plot()
        self.data_types_widget.clear_plot()
        self.correlation_widget.clear_plot()
        
        # Process data for plottin for sample data
        numeric_cols = self.DfSample._df.iloc[:, :].select_dtypes(include=np.number).columns
        new_data = self.DfSample._df[numeric_cols]
        mean_TIC = new_data.mean(axis=0) # Calculate mean along columns (axis=0)

        # Call the specific bar plot method on the correct PlotWidgetQC instance
        self.Log2_plotWidget.plot_bar_chart(mean_TIC, new_data.columns, numeric_cols.shape[0])

        self.missing_heatmap_widget.plot_missing_heatmap(self.DfSample._df)
        self.data_types_widget.plot_data_types_distribution(preprocessing_summary_perVariable(self.DfModel._df))
        self.correlation_widget.plot_correlation_matrix(self.DfSample._df)

        # Force layout update
        self.DFgrid.update()
        self.df_tab.updateGeometry()

    def onColumnDFClicked(self, column_index):
        # Get column name from index
        col_name = self.DfModel._df.columns[column_index]
        if self.old_column == col_name:
            return
        #print(f"Column at index {column_index}: {col_name}")

        # Clear previous distribution plot
        self.null_distribution_widget.clear_plot()
        
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
        self.titleColinfo.setVisible(True)
        self.column_table.setVisible(True)

        self.old_column = col_name

        self.null_distribution_widget.plot_null_pie(cutDF)

    def viewMetadataModel(self):
        if self.DfModel._df is None or self.DfModel._df.empty:
            QMessageBox.warning(self, "No Data", "Please load a dataset first!")
            return
        
        dialog = DialogMetadataModel(self.DfModel._df, self.DfMetadata._df.columns.tolist(), self.DfSample._df.columns.tolist(), self)
        if dialog.exec_() == QDialog.Accepted:
            DfMetadata, DfSample, DfModel = dialog.getResults()
            
            # Store the results
            self.DfMetadata._df = DfMetadata
            self.DfSample._df = DfSample
            #self.DfModel._df = DfModel

            # Update the main model safely
            self.DfModel.beginResetModel()
            self.DfModel._df = DfModel
            self.DfModel.endResetModel()

            self.loadGeneralDataTable()
            
            # Update UI plots
            self.loadLogSamplePlot()

            QMessageBox.information(self, "Success", 
                                "Columns assigned successfully!\n"
                                f"Metadata columns: {len(DfMetadata.columns)}\n"
                                f"Sample columns: {len(DfSample.columns)}\n"
                                f"Original columns: {len(DfModel.columns)}")

    def resetOriginal(self):
        if self.DfModel._df.empty:
            QMessageBox.warning(self, "No Data", "Please load a dataset first!")
            return
        emptyDF = pd.DataFrame()
        # We assign the copy to the main model
        self.DfModel.beginResetModel()
        self.DfModel._df = self.DfReset._df
        self.DfModel.endResetModel()

        self.DfMetadata = PandasModel(emptyDF)
        self.DfSample = PandasModel(emptyDF)

        self.Log2_plotWidget.clear_plot()
        self.missing_heatmap_widget.clear_plot()
        self.correlation_widget.clear_plot()
        self.null_distribution_widget.clear_plot()
        self.titleColinfo.setVisible(False)
        self.column_table.setVisible(False)
        self.data_types_widget.plot_data_types_distribution(preprocessing_summary_perVariable(self.DfModel._df))
        self.loadGeneralDataTable()

        self.old_column = ""

        QMessageBox.information(self, "Success", 
                                "Model restored to it's original data!")


    def logout(self):
        self.close()
