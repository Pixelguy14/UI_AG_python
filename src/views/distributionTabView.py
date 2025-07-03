from PyQt5.QtWidgets import QLabel, QTableView, QHeaderView, QAbstractScrollArea, QSizePolicy, QVBoxLayout, QGridLayout
from src.models.pandas_tablemodel import PandasModel
from src.models.plot_widgetmodel import PlotWidgetQC
from PyQt5.QtCore import Qt

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
    self.DfSampleThd = PandasModel()
    self.dfSampleTable.setModel(self.DfSampleThd)
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