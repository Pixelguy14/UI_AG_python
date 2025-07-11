from PyQt5.QtWidgets import QLabel, QTableView, QVBoxLayout, QGridLayout
from src.models.pandas_tablemodel import PandasModel

def setupComparisonTab(self):
    self.comparisonGrid = QGridLayout()

    # Create a vertical layout for the first table section
    verticalBoxLayout1 = QVBoxLayout()

    # Title
    title1 = QLabel("Original Sample DataFrame")
    title1.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; " \
                         "padding: 5px;")
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
    self.DfSampleThd = PandasModel()
    self.dfSampleTableProc = QTableView()
    self.dfSampleTableProc.setAlternatingRowColors(True)
    self.dfSampleTableProc.setEditTriggers(QTableView.NoEditTriggers)
    self.dfSampleTableProc.setModel(self.DfSampleThd)
    verticalBoxLayout2.addWidget(self.dfSampleTableProc)

    self.comparisonGrid.addLayout(verticalBoxLayout2, 0, 1, 1, 1)

    self.comparisonTab.setLayout(self.comparisonGrid)
