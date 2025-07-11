from PyQt5.QtWidgets import (QVBoxLayout, QTableView, QLabel, QGridLayout)
from PyQt5.QtGui import QStandardItemModel
from src.models.plot_widgetmodel import PlotWidgetQC
# from src.models.plot_widgetmodel_pyqtgraph import PlotWidgetQC

def setupSummaryTab(self):
    self.Plotgrid = QGridLayout()
    
    # Create a vertical layout for the plot section
    containerTableGeneralData = QVBoxLayout()
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

    containerTableGeneralData.addWidget(self.tableGeneralData)
    # Add container to grid
    self.Plotgrid.addLayout(containerTableGeneralData, 0, 0, 9, 2) # row, col, rowSpan, colSpan
    
    # Create a vertical layout for the plot section
    containerDataTypes = QVBoxLayout()
    self.plotWidgetDataTypes = PlotWidgetQC(self.tabSummary)
    containerDataTypes.addWidget(self.plotWidgetDataTypes)
    # Add container to grid
    self.Plotgrid.addLayout(containerDataTypes, 0, 3, 4, 3) # row, col, rowSpan, colSpan

    # Create a vertical layout for the plot section
    containerWidgetLog = QVBoxLayout()
    self.plotWidgetLog2 = PlotWidgetQC(self.tabSummary)
    containerWidgetLog.addWidget(self.plotWidgetLog2)
    # Add container to grid
    self.Plotgrid.addLayout(containerWidgetLog, 4, 3, 5, 3) # row, col, rowSpan, colSpan
    
    # Create a vertical layout for the plot section
    containerCorrelation = QVBoxLayout()
    self.plotWidgetCorrelation = PlotWidgetQC(self.tabSummary)
    containerCorrelation.addWidget(self.plotWidgetCorrelation)
    # Add container to grid
    self.Plotgrid.addLayout(containerCorrelation, 10, 0, 9, 3) # row, col, rowSpan, colSpan

    # Create a vertical layout for the plot section
    containerMissingHeatmap = QVBoxLayout()
    self.plotWidgetMissingHeatmap = PlotWidgetQC(self.tabSummary)
    containerMissingHeatmap.addWidget(self.plotWidgetMissingHeatmap)
    # Add container to grid
    self.Plotgrid.addLayout(containerMissingHeatmap, 10, 4, 9, 3) # row, col, rowSpan, colSpan




    """
    # Create a vertical layout for the plot section
    container5 = QVBoxLayout()
    # Title
    title = QLabel("leftover space")
    title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
    container5.addWidget(title)
    # Add container to grid
    self.Plotgrid.addLayout(container5, 6, 4, 3, 2) # row, col, rowSpan, colSpan
    """
    """
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
    """
    self.tabSummary.setLayout(self.Plotgrid)
