from PyQt5.QtWidgets import (QLabel, QTableView, QHeaderView,
                             QAbstractScrollArea, QSizePolicy, QVBoxLayout,
                             QGridLayout)
from PyQt5.QtGui import QStandardItemModel
from src.models.pandas_tablemodel import PandasModel
from src.models.plot_widgetmodel import PlotWidgetQC
# from src.models.plot_widgetmodel_pyqtgraph import PlotWidgetQC
from PyQt5.QtCore import Qt


def setupAnalysisTab(self):
    # Create main grid layout
    self.analysisGrid = QGridLayout()
    self.analysisGrid.setContentsMargins(0, 0, 0, 0)
    self.analysisGrid.setSpacing(10)

    # Create a vertical layout for the table section
    verticalBoxLayout = QVBoxLayout()

    # Title
    title = QLabel("Analysis DataFrame Table")
    title.setStyleSheet("font-size: 16px; font-weight: bold; "
                        "color: #333; padding: 5px;")
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
    self.dfAnalysisTable.setSizeAdjustPolicy(
        QAbstractScrollArea.AdjustToContents)
    self.dfAnalysisTable.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)
    self.dfAnalysisTable.verticalHeader().setVisible(False)
    self.dfAnalysisTable.verticalHeader().setDefaultSectionSize(50)

    # Set the PandasModel
    self.DfSampleThd = PandasModel()
    self.dfAnalysisTable.setModel(self.DfSampleThd)
    verticalBoxLayout.addWidget(self.dfAnalysisTable)

    # Add instruction label
    self.labelInstructionAnalysis = QLabel("Click a column header "
                                           "to display column data")
    self.labelInstructionAnalysis.setStyleSheet("font-style: italic; color: "
                                                "#666; padding: 5px;")
    self.labelInstructionAnalysis.setAlignment(Qt.AlignCenter)
    verticalBoxLayout.addWidget(self.labelInstructionAnalysis)

    # Add table container to grid
    self.analysisGrid.addLayout(verticalBoxLayout, 0, 0, 5, 5)

    verticalBoxLayoutNullDistribution = QVBoxLayout()
    self.plotWidgetNullDistributionAnalysis = PlotWidgetQC(self.analysisTab)
    verticalBoxLayoutNullDistribution.addWidget(
        self.plotWidgetNullDistributionAnalysis)
    self.analysisGrid.addLayout(verticalBoxLayoutNullDistribution, 0, 5, 2, 2)

    # Create a vertical layout for the table section
    colverticalBoxLayout = QVBoxLayout()

    # Title
    self.titleColinfoAnalysis = QLabel("Column Information")
    self.titleColinfoAnalysis.setStyleSheet("font-size: 16px; font-weight: "
                                            "bold; color: #333; padding: 5px;")
    self.titleColinfoAnalysis.setVisible(False)
    colverticalBoxLayout.addWidget(self.titleColinfoAnalysis)

    # Create invisible table for column data
    self.tableColumnAnalysis = QTableView()
    self.tableColumnAnalysis.setVisible(False)
    self.tableColumnAnalysis.setAlternatingRowColors(True)
    self.tableColumnAnalysis.setEditTriggers(QTableView.NoEditTriggers)
    self.tableColumnAnalysis.verticalHeader().setVisible(False)
    self.tableColumnAnalysis.setSizePolicy(QSizePolicy.Expanding,
                                           QSizePolicy.Expanding)

    # Create model for column data
    self.colModelAnalysis = QStandardItemModel()
    self.tableColumnAnalysis.setModel(self.colModelAnalysis)

    colverticalBoxLayout.addWidget(self.tableColumnAnalysis)

    self.analysisGrid.addLayout(colverticalBoxLayout, 2, 5, 3, 2)

    self.analysisGrid.setColumnStretch(0, 5)
    self.analysisGrid.setColumnStretch(5, 2)

    self.analysisTab.setLayout(self.analysisGrid)
