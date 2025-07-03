from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QTabWidget, QWidget, QGridLayout
from PyQt5.QtCore import Qt
from src.views.analysisTabView import setupAnalysisTab
from src.views.comparisonTabView import setupComparisonTab
from src.views.distributionTabView import setupViewDistributionTab

def setupImputationTab(self):
    self.ImputationGrid = QGridLayout()

    # Create an horizontal layout above the grid that contains imputation related operations
    horizontalBoxLayoutImputation = QHBoxLayout()

    btnThresholding = QPushButton("Threshold")
    btnThresholding.setCursor(Qt.PointingHandCursor)
    btnThresholding.clicked.connect(lambda: self.viewThresholdingModel())
    btnThresholding.setStyleSheet("padding: 4px 8px;")
    horizontalBoxLayoutImputation.addWidget(btnThresholding)

    btnImputation = QPushButton("Imputation Methods")
    btnImputation.setCursor(Qt.PointingHandCursor)
    btnImputation.clicked.connect(lambda: self.viewImputationModel())
    btnImputation.setStyleSheet("padding: 4px 8px;")
    horizontalBoxLayoutImputation.addWidget(btnImputation)

    btnNormalization = QPushButton("Normalization Methods")
    btnNormalization.setCursor(Qt.PointingHandCursor)
    #self.btnNormalization.clicked.connect()
    btnNormalization.setStyleSheet("padding: 4px 8px;")
    horizontalBoxLayoutImputation.addWidget(btnNormalization)

    btnTransformation = QPushButton("Data Transformation Methods")
    btnTransformation.setCursor(Qt.PointingHandCursor)
    #self.btnTransformation.clicked.connect()
    btnTransformation.setStyleSheet("padding: 4px 8px;")
    horizontalBoxLayoutImputation.addWidget(btnTransformation)

    btnScaling = QPushButton("Data Scaling Methods")
    btnScaling.setCursor(Qt.PointingHandCursor)
    #self.btnScaling.clicked.connect()
    btnScaling.setStyleSheet("padding: 4px 8px;")
    horizontalBoxLayoutImputation.addWidget(btnScaling)

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
    #self.setupAnalysisTab()
    setupAnalysisTab(self)

    # -- Comparison Tab --
    self.comparisonTab = QWidget()
    self.imputationTabWidget.addTab(self.comparisonTab, "Comparison")
    #self.setupComparisonTab()
    setupComparisonTab(self)

    # -- View Distribution Tab --
    self.viewDistributionTab = QWidget()
    self.imputationTabWidget.addTab(self.viewDistributionTab, "View Distribution")
    #self.setupViewDistributionTab()
    setupViewDistributionTab(self)

    self.tabImputation.setLayout(self.ImputationGrid)
