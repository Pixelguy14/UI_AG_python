from PyQt5.QtWidgets import (QMainWindow, QPushButton, QWidget, QHeaderView, QToolBar, QDialog, QMessageBox, QLabel, QSizePolicy, QTabWidget)
from PyQt5.QtCore import Qt, QSize
import xdialog
import pandas as pd
import numpy as np
import os

from src.functions.exploratory_data import preprocessing_general_dataset_statistics, preprocessing_summary_perVariable
from src.models.metadata_dialogmodel import DialogMetadataModel
from src.models.imputation_dialogmodel import DialogImputationModel
from src.models.threshold_dialogmodel import DialogThresholdingModel
from src.models.pandas_tablemodel import PandasModel
from src.models.loadfile_dialogmodel import DialogLoadFileModel
from src.views.summaryTabView import setupSummaryTab
from src.views.dataframeTabView import setupDFTab
from src.views.imputationTabView import setupImputationTab

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
        self.DfSampleThd = PandasModel(emptyDF) # Model of the thresholded sample data
        self.DfSampleImp = PandasModel(emptyDF) # Model of the imputed sample data

        self.DfReset = PandasModel(emptyDF) # Incase you'll want to reset the Df to before any changes.

        self.old_column = ""
        self.df_orientation = 'cols' # Default orientation

    def initUI(self):
        self.setWindowTitle('Main Window')
        #self.setFixedSize(500, 600)
        self.showMaximized()
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #3156A1;
                font-family: Arial;
            }
            QGridLayout {
                background-color: #F2F2F2;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QTableView {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #F2F2F2;
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
                background-color: #F2F2F2;
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
        title_label = QLabel("Omics GUI")
        title_label.setObjectName("titleLabel")
        self.toolbar.addWidget(title_label)

        self.toolbar.addSeparator()

        # Action button for the toolbar
        self.btnDefineFile = QPushButton("Load File")
        self.btnDefineFile.setCursor(Qt.PointingHandCursor)
        #self.btnDefineFile.clicked.connect(lambda: self.setDFModel())
        self.btnDefineFile.clicked.connect(lambda: self.viewLoadFileModel())
        self.toolbar.addWidget(self.btnDefineFile)

        self.btnDefineMD = QPushButton("Define Metadata")
        self.btnDefineMD.setCursor(Qt.PointingHandCursor)
        self.btnDefineMD.clicked.connect(lambda: self.viewMetadataModel())
        self.toolbar.addWidget(self.btnDefineMD)

        self.btnDefineC = QPushButton("Define Classes")
        self.btnDefineC.setCursor(Qt.PointingHandCursor)
        self.btnDefineC.clicked.connect(lambda: self.viewMetadataModel())
        self.toolbar.addWidget(self.btnDefineC)

        self.toolbar.addSeparator()

        spacer = QWidget()

        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        spacer.setStyleSheet("background-color: transparent;")
        self.toolbar.addWidget(spacer)

        # button to reset the dataframe to it's original state
        self.btnLogout = QPushButton("Reset Dataframe")
        self.btnLogout.setCursor(Qt.PointingHandCursor)
        self.btnLogout.clicked.connect(self.resetOriginal)
        self.toolbar.addWidget(self.btnLogout)
        
        # button to exit
        self.btnLogout = QPushButton("Exit")
        self.btnLogout.setCursor(Qt.PointingHandCursor)
        self.btnLogout.clicked.connect(self.logout)
        self.toolbar.addWidget(self.btnLogout)

        # Main tab widget
        self.tabWidget = QTabWidget()
        self.tabWidget.setTabPosition(QTabWidget.West)
        
        self.tabSummary = QWidget()
        #self.setupSummaryTab()
        setupSummaryTab(self)
        self.tabWidget.addTab(self.tabSummary, "Summary")
        
        self.tabDf = QWidget()
        #self.setupDFTab()
        setupDFTab(self)
        self.tabWidget.addTab(self.tabDf, "DataFrame")

        self.tabImputation = QWidget()
        #self.setupImputationTab()
        setupImputationTab(self)
        self.tabWidget.addTab(self.tabImputation, "Imputation")

        self.tabTargetDMiner = QWidget()
        self.tabWidget.addTab(self.tabTargetDMiner, "TDM")
        
        self.setCentralWidget(self.tabWidget)

    def viewLoadFileModel(self):
        # Create and show the dialog
        dialog = DialogLoadFileModel(self)
        if dialog.exec_() != QDialog.Accepted:
            return
            
        # Get results from dialog
        self.df = dialog.getResults()
        
        # Clear UI elements
        self.plotWidgetLog2.clear_plot()
        self.plotWidgetMissingHeatmap.clear_plot()
        self.plotWidgetCorrelation.clear_plot()
        self.plotWidgetNullDistribution.clear_plot()
        self.titleColinfo.setVisible(False)
        self.tableColumn.setVisible(False)
        self.old_column = ""
        
        self.DfModel = PandasModel(self.df)
        self.DfReset._df = self.df.copy()
        
        # Update the table view
        self.dfTable.setModel(self.DfModel)
        
        # Configure header
        header = self.dfTable.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(90)
        header.setDefaultSectionSize(120)
        header.setMaximumSectionSize(150)
        
        # Update other UI elements
        self.loadGeneralDataTable()
        self.plotWidgetDataTypes.plot_data_types_distribution(preprocessing_summary_perVariable(self.DfModel._df))
        
        print("Data successfully loaded and displayed.")

    def loadGeneralDataTable(self):
        summaryDF = preprocessing_general_dataset_statistics(self.DfModel._df)
        # Update the column table model
        self.colGeneral = PandasModel(summaryDF.T)
        self.tableGeneralData.setModel(self.colGeneral)

        header = self.tableGeneralData.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(300)
        header.setDefaultSectionSize(300)
        header.setMaximumSectionSize(500) 
        
        # Adjust column table settings
        self.tableGeneralData.verticalHeader().setVisible(True)
        self.tableGeneralData.horizontalHeader().setVisible(False)
        self.tableGeneralData.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Hide instruction and show column table
        self.tableGeneralData.setVisible(True)

    def loadSummaryPlots(self):
        self.plotWidgetLog2.clear_plot()
        self.plotWidgetMissingHeatmap.clear_plot()
        self.plotWidgetDataTypes.clear_plot()
        self.plotWidgetCorrelation.clear_plot()
        if self.DfModel._df.empty:
            return
        self.plotWidgetDataTypes.plot_data_types_distribution(preprocessing_summary_perVariable(self.DfModel._df))
        if self.DfSampleThd._df.empty:
            return
        # Process data for plottin for sample data
        numeric_cols = self.DfSampleThd._df.iloc[:, :].select_dtypes(include=np.number).columns
        new_data = self.DfSampleThd._df[numeric_cols]
        mean_TIC = new_data.mean(axis=0) # Calculate mean along columns (axis=0)
        # Call the specific bar plot method on the correct PlotWidgetQC instance
        self.plotWidgetLog2.plot_bar_chart(mean_TIC, new_data.columns, numeric_cols.shape[0])

        self.plotWidgetMissingHeatmap.plot_missing_heatmap(self.DfSampleThd._df)
        self.plotWidgetCorrelation.plot_correlation_matrix(self.DfSampleThd._df)
        
        # Force layout update
        self.DFgrid.update()
        self.tabDf.updateGeometry()

    def onColumnDFClicked(self, column_index):
        if not hasattr(self, 'DfModel') or self.DfModel._df.empty:
            return

        col_name = self.DfModel._df.columns[column_index]

        self.plotWidgetNullDistribution.clear_plot()
        if self.old_column == col_name:
            return

        cutDF = self.DfModel._df[[col_name]]

        # Get stats for the selected data
        colDF = preprocessing_summary_perVariable(cutDF)
        transposed_df = colDF.T
        
        # Display stats in the side table
        self.colModel = PandasModel(transposed_df) # This table is always column-oriented
        self.tableColumn.setModel(self.colModel)
        
        self.tableColumn.verticalHeader().setVisible(True)
        self.tableColumn.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.titleColinfo.setVisible(True)
        self.tableColumn.setVisible(True)

        # Plot pie chart of nulls
        self.plotWidgetNullDistribution.plot_null_pie(cutDF)

    def onColumnAnalysisClicked(self, column_index):
        # Get column name from index
        col_name = self.DfSampleThd._df.columns[column_index]
        if self.old_column == col_name:
            return

        # Clear previous distribution plot
        self.plotWidgetNullDistributionAnalysis.clear_plot()
        
        # Create a DataFrame containing just this column
        cutDF = self.DfSampleThd._df[[col_name]]
        # Create the preprocessing for just colDF
        colDF = preprocessing_summary_perVariable(cutDF)
        
        # Create a transposed view for vertical display
        transposed_df = colDF.T
        
        # Update the column table model
        self.colModelAnalysis = PandasModel(transposed_df)
        self.tableColumnAnalysis.setModel(self.colModelAnalysis)
        
        # Adjust column table settings
        self.tableColumnAnalysis.verticalHeader().setVisible(True)
        self.tableColumnAnalysis.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Hide instruction and show column table
        self.titleColinfoAnalysis.setVisible(True)
        self.tableColumnAnalysis.setVisible(True)

        self.old_column = col_name

        self.plotWidgetNullDistributionAnalysis.plot_null_pie(cutDF)

    def onColumnDistributionClicked(self, column_index):
        # Get column name from index
        col_name = self.DfSampleThd._df.columns[column_index]
        
        # Clear previous distribution plot
        self.plotWidgetDistribution.clear_plot()

        # Get the data series for the selected column
        data_series = self.DfSampleThd._df[col_name]

        # Plot the distribution
        self.plotWidgetDistribution.plot_distribution(data_series)

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

            self.DfSampleThd._df = DfSample

            # Update Samples Table view:
            self.dfSampleTable.setModel(self.DfSampleThd)
            self.dfAnalysisTable.setModel(self.DfSampleThd)
            self.dfSampleTableOrig.setModel(self.DfSample)
            self.dfSampleTableProc.setModel(self.DfSampleThd)

            # Reapply header settings (essential after model change)
            header = self.dfSampleTable.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.Interactive)
            header.setMinimumSectionSize(90)
            header.setDefaultSectionSize(120)
            header.setMaximumSectionSize(150)

            print("Sample data successfully loaded and displayed")

            # Update the main model safely
            self.DfModel.beginResetModel()
            self.DfModel._df = DfModel
            self.DfModel.endResetModel()

            self.loadGeneralDataTable()

            # Update Samples Table view:
            self.dfSampleTable.setModel(self.DfSampleThd)

            # Reapply header settings (essential after model change)
            header = self.dfSampleTable.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.Interactive)
            header.setMinimumSectionSize(90)
            header.setDefaultSectionSize(120)
            header.setMaximumSectionSize(150)
            
            # Update UI plots
            self.loadSummaryPlots()

            QMessageBox.information(self, "Success", 
                                "Columns assigned successfully!\n"
                                f"Metadata columns: {len(DfMetadata.columns)}\n"
                                f"Sample columns: {len(DfSample.columns)}\n"
                                f"Original columns: {len(DfModel.columns)}")
        
    def viewThresholdingModel(self):
        if self.DfSample._df.empty:
            QMessageBox.warning(self, "No Data", "Please select the samples in the dataset!")
            return
        dialog = DialogThresholdingModel(self.DfSample._df, self.DfSampleThd._df, self)
        if dialog.exec_() == QDialog.Accepted:
            DfSample, DfSampleThd = dialog.getResults()

            # Update the thresholded model safely
            self.DfSampleThd.beginResetModel()
            self.DfSampleThd._df = DfSampleThd
            self.DfSampleThd.endResetModel()

            self.DfSample._df = DfSample

            # Update UI plots
            self.loadSummaryPlots()

            # Update table views
            self.dfAnalysisTable.setModel(self.DfSampleThd)
            self.dfSampleTableProc.setModel(self.DfSampleThd)
            """
            QMessageBox.information(self, "Success", 
                                "Sample Data was thresholded")
            """
    
    def viewImputationModel(self):
        if self.DfSample._df.empty:
            QMessageBox.warning(self, "No Data", "Please select the samples in the dataset!")
            return
        # Create and execute the dialog
        dialog = DialogImputationModel(self.DfSample._df, self.DfSampleThd._df, self.DfSampleImp._df, self)
        # Check if the user clicked "OK"
        if dialog.exec_() == QDialog.Accepted:
            # Retrieve the imputed DataFrame from the dialog
            imputed_df, samplethd_df = dialog.getResults()
            # Ensure the imputation was successful before updating
            if imputed_df is not None:
                # Update the model for the thresholded/imputed data
                self.DfSampleThd.beginResetModel()
                self.DfSampleThd._df = samplethd_df
                self.DfSampleImp._df = imputed_df
                self.DfSampleThd.endResetModel()

                # Refresh the plots with the new data
                self.loadSummaryPlots()
                print("Data imputation applied successfully")

    def resetOriginal(self):
        if self.DfReset._df.empty:
            QMessageBox.warning(self, "No Data", "Please load a dataset first or no changes to reset!")
            return
        
        emptyDF = pd.DataFrame()
        
        # Reset main model to its original state
        self.df_orientation = 'cols' # Reset orientation
        self.DfModel = PandasModel(self.DfReset._df.copy())
        self.dfTable.setModel(self.DfModel)

        # Reset other models
        self.DfMetadata = PandasModel(emptyDF)
        self.DfSample = PandasModel(emptyDF)
        self.DfSampleThd = PandasModel(emptyDF)

        # Clear UI components
        self.plotWidgetLog2.clear_plot()
        self.plotWidgetMissingHeatmap.clear_plot()
        self.plotWidgetCorrelation.clear_plot()
        self.plotWidgetNullDistribution.clear_plot()
        self.titleColinfo.setVisible(False)
        self.tableColumn.setVisible(False)
        self.plotWidgetDataTypes.plot_data_types_distribution(preprocessing_summary_perVariable(self.DfModel._df))
        self.loadGeneralDataTable()

        self.old_column = ""

        QMessageBox.information(self, "Success", "Model restored to its original data!")

    def logout(self):
        self.close()
