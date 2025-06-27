from PyQt5.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
                            QTableView, QHeaderView, QToolBar, QDialog, 
                            QLineEdit, QDateEdit, QComboBox, QMessageBox, QLabel,
                            QDialogButtonBox, QApplication, QFrame, QGridLayout, QSizePolicy, 
                            QStackedWidget, QSpinBox, QTabWidget, QTextEdit, QDoubleSpinBox, 
                            QAbstractScrollArea)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QDate, QSize
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor, QStandardItemModel
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel, QSqlQueryModel
import xdialog
import pandas as pd

from src.functions.exploratory_data import *
from src.functions.imputation_methods import *
from src.models.metadata_dialogmodel import *
from src.models.imputation_dialogmodel import *
from src.models.umbral_dialogmodel import *
from src.models.pandas_tablemodel import *
from src.models.plot_widgetmodel import *

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
        self.DfSampleUmb = PandasModel(emptyDF) # Model of the umbralized sample data

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
        self.btnLoadFile = QPushButton("Load File")
        self.btnLoadFile.setCursor(Qt.PointingHandCursor)
        self.btnLoadFile.clicked.connect(lambda: self.setDFModel())
        self.toolbar.addWidget(self.btnLoadFile)

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
        self.setupSummaryTab()
        self.tabWidget.addTab(self.tabSummary, "Summary")
        
        self.tabDf = QWidget()
        self.setupDFTab()
        self.tabWidget.addTab(self.tabDf, "DataFrame")

        self.tabImputation = QWidget()
        self.setupImputationTab()
        self.tabWidget.addTab(self.tabImputation, "Imputation")
        
        self.setCentralWidget(self.tabWidget)

    def setupSummaryTab(self):
        self.Plotgrid = QGridLayout()

        # Create a vertical layout for the plot section
        container1 = QVBoxLayout()
        self.plotWidgetLog2 = PlotWidgetQC(self.tabSummary)
        container1.addWidget(self.plotWidgetLog2)
        # Add container to grid
        self.Plotgrid.addLayout(container1, 0, 0, 3, 2) # row, col, rowSpan, colSpan

        # Create a vertical layout for the plot section
        container2 = QVBoxLayout()
        self.plotWidgetMissingHeatmap = PlotWidgetQC(self.tabSummary)
        container2.addWidget(self.plotWidgetMissingHeatmap)
        # Add container to grid
        self.Plotgrid.addLayout(container2, 3, 0, 4, 2) # row, col, rowSpan, colSpan

        # Create a vertical layout for the plot section
        container3 = QVBoxLayout()
        # Create invisible table for general data
        self.tableGeneralData = QTableView()
        self.tableGeneralData.setVisible(False)  # Initially hidden
        self.tableGeneralData.setAlternatingRowColors(True)
        self.tableGeneralData.setEditTriggers(QTableView.NoEditTriggers)
        self.tableGeneralData.verticalHeader().setVisible(False)
        #self.tableGeneralData.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create model for general data
        self.colGeneral = QStandardItemModel()
        self.tableGeneralData.setModel(self.colGeneral)

        container3.addWidget(self.tableGeneralData)
        # Add container to grid
        self.Plotgrid.addLayout(container3, 7, 0, 2, 2) # row, col, rowSpan, colSpan

        # Create a vertical layout for the plot section
        container4 = QVBoxLayout()
        self.plotWidgetCorrelation = PlotWidgetQC(self.tabSummary)
        container4.addWidget(self.plotWidgetCorrelation)
        # Add container to grid
        self.Plotgrid.addLayout(container4, 0, 2, 6, 4) # row, col, rowSpan, colSpan

        # Create a vertical layout for the plot section
        container4 = QVBoxLayout()
        self.plotWidgetDataTypes = PlotWidgetQC(self.tabSummary)
        container4.addWidget(self.plotWidgetDataTypes)
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

        self.tabSummary.setLayout(self.Plotgrid)

    def setupDFTab(self):
        # Create main grid layout
        self.DFgrid = QGridLayout()
        self.DFgrid.setContentsMargins(0, 0, 0, 0)
        self.DFgrid.setSpacing(10)
        
        # Create a vertical layout for the table section
        verticalBoxLayout = QVBoxLayout()
        
        # Title
        title = QLabel("DataFrame Table")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        verticalBoxLayout.addWidget(title)
        
        # Main DataFrame table
        self.dfTable = QTableView()
        self.dfTable.setAlternatingRowColors(True)
        self.dfTable.setSelectionBehavior(QTableView.SelectItems)  # Changed to SelectItems
        self.dfTable.setEditTriggers(QTableView.NoEditTriggers)
        
        # Configure header
        header = self.dfTable.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(70)
        header.setDefaultSectionSize(150)
        header.setMaximumSectionSize(300)
        header.sectionClicked.connect(self.onColumnDFClicked)  # Connect column header clicks
        
        # Configure table behavior
        self.dfTable.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        self.dfTable.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.dfTable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dfTable.verticalHeader().setVisible(False)
        self.dfTable.verticalHeader().setDefaultSectionSize(50)
        #self.dfTable.setSortingEnabled(True)
        
        # Set the PandasModel
        self.DfModel = PandasModel()
        self.dfTable.setModel(self.DfModel)
        verticalBoxLayout.addWidget(self.dfTable)
        
        # Add instruction label
        self.labelInstruction = QLabel("Click a column header to display column data")
        self.labelInstruction.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        self.labelInstruction.setAlignment(Qt.AlignCenter)
        verticalBoxLayout.addWidget(self.labelInstruction)
        
        # Add table container to grid
        self.DFgrid.addLayout(verticalBoxLayout, 0, 0, 5, 5)  # row, col, rowSpan, colSpan
        
        verticalBoxLayoutNullDistribution = QVBoxLayout()
        self.plotWidgetNullDistribution = PlotWidgetQC(self.tabDf)
        #self.plotWidgetNullDistribution.setMinimumHeight(300)
        #self.plotWidgetNullDistribution.setMinimumWidth(300)
        verticalBoxLayoutNullDistribution.addWidget(self.plotWidgetNullDistribution)
        self.DFgrid.addLayout(verticalBoxLayoutNullDistribution, 0, 5, 2, 2)

        # Create a vertical layout for the table section
        colverticalBoxLayout = QVBoxLayout()
        
        # Title
        self.titleColinfo = QLabel("Column Information")
        self.titleColinfo.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        self.titleColinfo.setVisible(False)  # Initially hidden
        colverticalBoxLayout.addWidget(self.titleColinfo)
        
        # Create invisible table for column data
        self.tableColumn = QTableView()
        self.tableColumn.setVisible(False)  # Initially hidden
        self.tableColumn.setAlternatingRowColors(True)
        self.tableColumn.setEditTriggers(QTableView.NoEditTriggers)
        self.tableColumn.verticalHeader().setVisible(False)
        self.tableColumn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create model for column data
        self.colModel = QStandardItemModel()
        self.tableColumn.setModel(self.colModel)

        colverticalBoxLayout.addWidget(self.tableColumn)

        self.DFgrid.addLayout(colverticalBoxLayout, 2, 5, 3, 2) # row, col, rowSpan, colSpan
        
        self.DFgrid.setColumnStretch(0, 5)  # Table area fixed size
        self.DFgrid.setColumnStretch(5, 2)   # Plot area fixed size

        self.tabDf.setLayout(self.DFgrid)
    
    def setupImputationTab(self):
        self.ImputationGrid = QGridLayout()

        # Create an horizontal layout above the grid that contains imputation related operations
        horizontalBoxLayoutImputation = QHBoxLayout()

        btnUmbralization = QPushButton("Umbralize")
        btnUmbralization.setCursor(Qt.PointingHandCursor)
        btnUmbralization.clicked.connect(lambda: self.viewUmbralizationModel())
        btnUmbralization.setStyleSheet("padding: 4px 8px;")
        horizontalBoxLayoutImputation.addWidget(btnUmbralization)

        btnImputation = QPushButton("Imputation Methods")
        btnImputation.setCursor(Qt.PointingHandCursor)
        btnImputation.clicked.connect(lambda: self.viewImputationModel())
        btnImputation.setStyleSheet("padding: 4px 8px;")
        horizontalBoxLayoutImputation.addWidget(btnImputation)

        btnTargetDecoy = QPushButton("Target Decoy Miner")
        btnTargetDecoy.setCursor(Qt.PointingHandCursor)
        #self.btnTargetDecoy.clicked.connect()
        btnTargetDecoy.setStyleSheet("padding: 4px 8px;")
        horizontalBoxLayoutImputation.addWidget(btnTargetDecoy)

        btnNormalize = QPushButton("Normalize")
        btnNormalize.setCursor(Qt.PointingHandCursor)
        #self.btnNormalize.clicked.connect()
        btnNormalize.setStyleSheet("padding: 4px 8px;")
        horizontalBoxLayoutImputation.addWidget(btnNormalize)

        btnEvalDist = QPushButton("Evaluate Sample Distribution")
        btnEvalDist.setCursor(Qt.PointingHandCursor)
        #self.btnEvalDist.clicked.connect()
        btnEvalDist.setStyleSheet("padding: 4px 8px;")
        horizontalBoxLayoutImputation.addWidget(btnEvalDist)

        self.ImputationGrid.addLayout(horizontalBoxLayoutImputation, 0, 0, 1, 8)

        # Create the tab widget
        self.imputationTabWidget = QTabWidget()
        self.ImputationGrid.addWidget(self.imputationTabWidget, 1, 0, 7, 8)

        # -- Analysis Tab --
        self.analysisTab = QWidget()
        self.imputationTabWidget.addTab(self.analysisTab, "Analysis")
        self.setupAnalysisTab()

        # -- Comparison Tab --
        self.comparisonTab = QWidget()
        self.imputationTabWidget.addTab(self.comparisonTab, "Comparison")
        self.setupComparisonTab()

        # -- View Distribution Tab --
        self.viewDistributionTab = QWidget()
        self.imputationTabWidget.addTab(self.viewDistributionTab, "View Distribution")
        self.setupViewDistributionTab()

        self.tabImputation.setLayout(self.ImputationGrid)

    def setupAnalysisTab(self):
        # Create main grid layout
        self.analysisGrid = QGridLayout()
        self.analysisGrid.setContentsMargins(0, 0, 0, 0)
        self.analysisGrid.setSpacing(10)
        
        # Create a vertical layout for the table section
        verticalBoxLayout = QVBoxLayout()
        
        # Title
        title = QLabel("Analysis DataFrame Table")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        verticalBoxLayout.addWidget(title)
        
        # Main DataFrame table
        self.dfAnalysisTable = QTableView()
        self.dfAnalysisTable.setAlternatingRowColors(True)
        self.dfAnalysisTable.setSelectionBehavior(QTableView.SelectItems)
        self.dfAnalysisTable.setEditTriggers(QTableView.NoEditTriggers)
        
        # Configure header
        header = self.dfAnalysisTable.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(70)
        header.setDefaultSectionSize(150)
        header.setMaximumSectionSize(300)
        header.sectionClicked.connect(self.onColumnAnalysisClicked)
        
        # Configure table behavior
        self.dfAnalysisTable.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        self.dfAnalysisTable.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.dfAnalysisTable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dfAnalysisTable.verticalHeader().setVisible(False)
        self.dfAnalysisTable.verticalHeader().setDefaultSectionSize(50)
        
        # Set the PandasModel
        self.DfSampleUmb = PandasModel()
        self.dfAnalysisTable.setModel(self.DfSampleUmb)
        verticalBoxLayout.addWidget(self.dfAnalysisTable)
        
        # Add instruction label
        self.labelInstructionAnalysis = QLabel("Click a column header to display column data")
        self.labelInstructionAnalysis.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        self.labelInstructionAnalysis.setAlignment(Qt.AlignCenter)
        verticalBoxLayout.addWidget(self.labelInstructionAnalysis)
        
        # Add table container to grid
        self.analysisGrid.addLayout(verticalBoxLayout, 0, 0, 5, 5)
        
        verticalBoxLayoutNullDistribution = QVBoxLayout()
        self.plotWidgetNullDistributionAnalysis = PlotWidgetQC(self.analysisTab)
        verticalBoxLayoutNullDistribution.addWidget(self.plotWidgetNullDistributionAnalysis)
        self.analysisGrid.addLayout(verticalBoxLayoutNullDistribution, 0, 5, 2, 2)

        # Create a vertical layout for the table section
        colverticalBoxLayout = QVBoxLayout()
        
        # Title
        self.titleColinfoAnalysis = QLabel("Column Information")
        self.titleColinfoAnalysis.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        self.titleColinfoAnalysis.setVisible(False)
        colverticalBoxLayout.addWidget(self.titleColinfoAnalysis)
        
        # Create invisible table for column data
        self.tableColumnAnalysis = QTableView()
        self.tableColumnAnalysis.setVisible(False)
        self.tableColumnAnalysis.setAlternatingRowColors(True)
        self.tableColumnAnalysis.setEditTriggers(QTableView.NoEditTriggers)
        self.tableColumnAnalysis.verticalHeader().setVisible(False)
        self.tableColumnAnalysis.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create model for column data
        self.colModelAnalysis = QStandardItemModel()
        self.tableColumnAnalysis.setModel(self.colModelAnalysis)

        colverticalBoxLayout.addWidget(self.tableColumnAnalysis)

        self.analysisGrid.addLayout(colverticalBoxLayout, 2, 5, 3, 2)
        
        self.analysisGrid.setColumnStretch(0, 5)
        self.analysisGrid.setColumnStretch(5, 2)

        self.analysisTab.setLayout(self.analysisGrid)

    def setupComparisonTab(self):
        self.comparisonGrid = QGridLayout()

        # Create a vertical layout for the first table section
        verticalBoxLayout1 = QVBoxLayout()
        
        # Title
        title1 = QLabel("Original Sample DataFrame")
        title1.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        verticalBoxLayout1.addWidget(title1)
        
        # Original Sample DataFrame table
        self.DfSample = PandasModel()
        self.dfSampleTableOrig = QTableView()
        self.dfSampleTableOrig.setAlternatingRowColors(True)
        self.dfSampleTableOrig.setEditTriggers(QTableView.NoEditTriggers)
        self.dfSampleTableOrig.setModel(self.DfSample)
        verticalBoxLayout1.addWidget(self.dfSampleTableOrig)

        self.comparisonGrid.addLayout(verticalBoxLayout1, 0, 0, 1, 1)

        # Create a vertical layout for the second table section
        verticalBoxLayout2 = QVBoxLayout()
        
        # Title
        title2 = QLabel("Processed Sample DataFrame")
        title2.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        verticalBoxLayout2.addWidget(title2)
        
        # Processed Sample DataFrame table
        self.DfSampleUmb = PandasModel()
        self.dfSampleTableProc = QTableView()
        self.dfSampleTableProc.setAlternatingRowColors(True)
        self.dfSampleTableProc.setEditTriggers(QTableView.NoEditTriggers)
        self.dfSampleTableProc.setModel(self.DfSampleUmb)
        verticalBoxLayout2.addWidget(self.dfSampleTableProc)

        self.comparisonGrid.addLayout(verticalBoxLayout2, 0, 1, 1, 1)

        self.comparisonTab.setLayout(self.comparisonGrid)

    def setupViewDistributionTab(self):
        self.viewDistributionGrid = QGridLayout()

        # Create a vertical layout for the table section
        verticalBoxLayout = QVBoxLayout()
        
        # Title
        title = QLabel("Sample DataFrame Table")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        verticalBoxLayout.addWidget(title)
        
        # Sample DataFrame table 
        self.dfSampleTable = QTableView()
        self.dfSampleTable.setAlternatingRowColors(True)
        self.dfSampleTable.setSelectionBehavior(QTableView.SelectItems)
        self.dfSampleTable.setEditTriggers(QTableView.NoEditTriggers)
        
        # Configure header
        header = self.dfSampleTable.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(70)
        header.setDefaultSectionSize(150)
        header.setMaximumSectionSize(300)
        
        # Configure table behavior
        self.dfSampleTable.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        self.dfSampleTable.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.dfSampleTable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dfSampleTable.verticalHeader().setVisible(False)
        self.dfSampleTable.verticalHeader().setDefaultSectionSize(50)
        
        # Set the PandasModel
        self.DfSampleUmb = PandasModel()
        self.dfSampleTable.setModel(self.DfSampleUmb)
        verticalBoxLayout.addWidget(self.dfSampleTable)
        
        # Add instruction label
        self.labelInstructionImp = QLabel("Click a column header to display column distribution")
        self.labelInstructionImp.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        self.labelInstructionImp.setAlignment(Qt.AlignCenter)
        verticalBoxLayout.addWidget(self.labelInstructionImp)
        
        # Add table container to grid
        self.viewDistributionGrid.addLayout(verticalBoxLayout, 0, 0, 4, 4)

        # Create a vertical layout for the plot section
        plotVerticalBoxLayout = QVBoxLayout()
        self.plotWidgetDistribution = PlotWidgetQC(self.viewDistributionTab)
        plotVerticalBoxLayout.addWidget(self.plotWidgetDistribution)
        self.viewDistributionGrid.addLayout(plotVerticalBoxLayout, 0, 4, 4, 4)

        self.viewDistributionGrid.setColumnStretch(0, 4)
        self.viewDistributionGrid.setColumnStretch(4, 4)
        self.viewDistributionTab.setLayout(self.viewDistributionGrid)

        header.sectionClicked.connect(self.onColumnDistributionClicked)

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
    
        self.plotWidgetLog2.clear_plot()
        self.plotWidgetMissingHeatmap.clear_plot()
        self.plotWidgetCorrelation.clear_plot()
        self.plotWidgetNullDistribution.clear_plot()
        self.titleColinfo.setVisible(False)
        self.tableColumn.setVisible(False)
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
        """
        self.DfModel.rename_columns({
            "rt": "RT",
            "mz": "m/z",
            "intensity": "Intensity"
        }, italic_cols=["m/z"])
        """

        # Set the model to the table view
        self.dfTable.setModel(self.DfModel)

        # Reapply header settings (essential after model change)
        header = self.dfTable.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(90)
        header.setDefaultSectionSize(120)
        header.setMaximumSectionSize(150)        

        #self.loadLogSamplePlot()
        self.loadGeneralDataTable()

        self.plotWidgetDataTypes.plot_data_types_distribution(preprocessing_summary_perVariable(self.DfModel._df))

        print("Data successfully loaded and displayed")

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

    def loadLogSamplePlot(self):
        self.plotWidgetLog2.clear_plot()
        self.plotWidgetMissingHeatmap.clear_plot()
        self.plotWidgetDataTypes.clear_plot()
        self.plotWidgetCorrelation.clear_plot()
        if self.DfModel._df.empty:
            return
        self.plotWidgetDataTypes.plot_data_types_distribution(preprocessing_summary_perVariable(self.DfModel._df))
        if self.DfSampleUmb._df.empty:
            return
        # Process data for plottin for sample data
        numeric_cols = self.DfSampleUmb._df.iloc[:, :].select_dtypes(include=np.number).columns
        new_data = self.DfSampleUmb._df[numeric_cols]
        mean_TIC = new_data.mean(axis=0) # Calculate mean along columns (axis=0)
        # Call the specific bar plot method on the correct PlotWidgetQC instance
        self.plotWidgetLog2.plot_bar_chart(mean_TIC, new_data.columns, numeric_cols.shape[0])

        self.plotWidgetMissingHeatmap.plot_missing_heatmap(self.DfSampleUmb._df)
        self.plotWidgetCorrelation.plot_correlation_matrix(self.DfSampleUmb._df)
        
        # Force layout update
        self.DFgrid.update()
        self.tabDf.updateGeometry()

    def onColumnDFClicked(self, column_index):
        # Get column name from index
        col_name = self.DfModel._df.columns[column_index]
        if self.old_column == col_name:
            return
        #print(f"Column at index {column_index}: {col_name}")

        # Clear previous distribution plot
        self.plotWidgetNullDistribution.clear_plot()
        
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
        self.tableColumn.setModel(self.colModel)
        
        # Adjust column table settings
        self.tableColumn.verticalHeader().setVisible(True)
        self.tableColumn.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Hide instruction and show column table
        self.titleColinfo.setVisible(True)
        self.tableColumn.setVisible(True)

        self.old_column = col_name

        self.plotWidgetNullDistribution.plot_null_pie(cutDF)

    def onColumnAnalysisClicked(self, column_index):
        # Get column name from index
        col_name = self.DfSampleUmb._df.columns[column_index]
        if self.old_column == col_name:
            return

        # Clear previous distribution plot
        self.plotWidgetNullDistributionAnalysis.clear_plot()
        
        # Create a DataFrame containing just this column
        cutDF = self.DfSampleUmb._df[[col_name]]
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
        col_name = self.DfSampleUmb._df.columns[column_index]
        
        # Clear previous distribution plot
        self.plotWidgetDistribution.clear_plot()

        # Get the data series for the selected column
        data_series = self.DfSampleUmb._df[col_name]

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

            self.DfSampleUmb._df = DfSample

            # Update Samples Table view:
            self.dfSampleTable.setModel(self.DfSampleUmb)
            self.dfAnalysisTable.setModel(self.DfSampleUmb)
            self.dfSampleTableOrig.setModel(self.DfSample)
            self.dfSampleTableProc.setModel(self.DfSampleUmb)

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
            self.dfSampleTable.setModel(self.DfSampleUmb)

            # Reapply header settings (essential after model change)
            header = self.dfSampleTable.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.Interactive)
            header.setMinimumSectionSize(90)
            header.setDefaultSectionSize(120)
            header.setMaximumSectionSize(150)
            
            # Update UI plots
            self.loadLogSamplePlot()

            QMessageBox.information(self, "Success", 
                                "Columns assigned successfully!\n"
                                f"Metadata columns: {len(DfMetadata.columns)}\n"
                                f"Sample columns: {len(DfSample.columns)}\n"
                                f"Original columns: {len(DfModel.columns)}")

    def viewUmbralizationModel(self):
        if self.DfSample._df.empty:
            QMessageBox.warning(self, "No Data", "Please select the samples in the dataset!")
            return
        dialog = DialogUmbralModel(self.DfSample._df, self.DfSampleUmb._df, self)
        if dialog.exec_() == QDialog.Accepted:
            DfSample, DfSampleUmb = dialog.getResults()

            # Update the umbralized model safely
            self.DfSampleUmb.beginResetModel()
            self.DfSampleUmb._df = DfSampleUmb
            self.DfSampleUmb.endResetModel()

            self.DfSample._df = DfSample

            # Update UI plots
            self.loadLogSamplePlot()

            # Update table views
            self.dfAnalysisTable.setModel(self.DfSampleUmb)
            self.dfSampleTableProc.setModel(self.DfSampleUmb)
            """
            QMessageBox.information(self, "Success", 
                                "Sample Data was umbralized")
            """
    
    def viewImputationModel(self):
        if self.DfSample._df.empty:
            QMessageBox.warning(self, "No Data", "Please select the samples in the dataset!")
            return
        # Create and execute the dialog
        dialog = DialogImputationModel(self.DfSample._df, self.DfSampleUmb._df, self)
        # Check if the user clicked "OK"
        if dialog.exec_() == QDialog.Accepted:
            # Retrieve the imputed DataFrame from the dialog
            imputed_df = dialog.getResults()
            # Ensure the imputation was successful before updating
            if imputed_df is not None:
                # Update the model for the umbralized/imputed data
                self.DfSampleUmb.beginResetModel()
                self.DfSampleUmb._df = imputed_df
                self.DfSampleUmb.endResetModel()

                # Refresh the plots with the new data
                self.loadLogSamplePlot()
                print("Data imputation applied successfully")

    def resetOriginal(self):
        if self.DfModel._df.empty and self.DfModel._df.equals(self.DfReset._df):
            QMessageBox.warning(self, "No Data", "Please load a dataset first!")
            return
        emptyDF = pd.DataFrame()
        # We assign the copy to the main model
        self.DfModel.beginResetModel()
        self.DfModel._df = self.DfReset._df
        self.DfModel.endResetModel()

        self.DfMetadata = PandasModel(emptyDF)
        self.DfSample = PandasModel(emptyDF)
        self.DfSampleUmb = PandasModel(emptyDF)

        self.plotWidgetLog2.clear_plot()
        self.plotWidgetMissingHeatmap.clear_plot()
        self.plotWidgetCorrelation.clear_plot()
        self.plotWidgetNullDistribution.clear_plot()
        self.titleColinfo.setVisible(False)
        self.tableColumn.setVisible(False)
        self.plotWidgetDataTypes.plot_data_types_distribution(preprocessing_summary_perVariable(self.DfModel._df))
        self.loadGeneralDataTable()

        self.old_column = ""

        QMessageBox.information(self, "Success", 
                                "Model restored to it's original data!")

    def logout(self):
        self.close()
