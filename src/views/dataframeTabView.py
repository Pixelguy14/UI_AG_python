from PyQt5.QtWidgets import (QVBoxLayout, QTableView, QHeaderView, QLabel, QGridLayout, QSizePolicy, QAbstractScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel

from src.models.plot_widgetmodel import PlotWidgetQC
from src.models.pandas_tablemodel import PandasModel

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
