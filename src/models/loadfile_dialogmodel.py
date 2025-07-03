from PyQt5.QtWidgets import (QPushButton, QVBoxLayout, QMessageBox, QDialog, QLabel, 
                             QRadioButton, QButtonGroup, QGroupBox, QTableView, QHBoxLayout,
                             QDialogButtonBox, QHeaderView, QFrame)
from PyQt5.QtCore import Qt
import xdialog
import os
from src.functions.exploratory_data import loadFile
from src.models.pandas_tablemodel import PandasModel

class DialogLoadFileModel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = None
        self.mainIdentifier = False  # false = cols, true = rows
        self.mainHandler = False    # false = vertical, true = horizontal
        self.setWindowTitle("Load and handle Dataset")
        self.setMinimumSize(1200, 600)
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("Load Dataset and select handling")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        main_layout.addWidget(title_label)

        # Load file button
        self.btnLoadFile = QPushButton("Load File")
        self.btnLoadFile.setCursor(Qt.PointingHandCursor)
        self.btnLoadFile.clicked.connect(self.loadFileDF)
        main_layout.addWidget(self.btnLoadFile)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Identifier selection
        identifier_group = QGroupBox("Select where are your Identifiers")
        identifier_layout = QVBoxLayout(identifier_group)
        
        instruction_label = QLabel("Are the main identifiers (e.g., sample names) in columns or rows?")
        instruction_label.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        instruction_label.setAlignment(Qt.AlignCenter)
        identifier_layout.addWidget(instruction_label)
        
        # Radio buttons for identifier position
        self.identifier_group = QButtonGroup(self)
        self.btnCols = QRadioButton("Columns (identifiers as column headers)")
        self.btnRows = QRadioButton("Rows (identifiers as row names)")
        self.btnCols.setChecked(True)
        
        self.identifier_group.addButton(self.btnCols, 0)
        self.identifier_group.addButton(self.btnRows, 1)
        
        identifier_layout.addWidget(self.btnCols)
        identifier_layout.addWidget(self.btnRows)
        main_layout.addWidget(identifier_group)

        # Handler selection
        handler_group = QGroupBox("Dataset Handling Orientation")
        handler_layout = QVBoxLayout(handler_group)
        
        instruction_label2 = QLabel("Do you want to handle your dataset Vertically or Horizontally? (This will transpose your dataset if the identifiers have the oposite orientation)")
        instruction_label2.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        instruction_label2.setAlignment(Qt.AlignCenter)
        handler_layout.addWidget(instruction_label2)
        
        # Radio buttons for dataset handling
        self.handler_group = QButtonGroup(self)
        self.btnVertical = QRadioButton("Vertical (samples as rows, features as columns)")
        self.btnHorizontal = QRadioButton("Horizontal (samples as columns, features as rows)")
        self.btnVertical.setChecked(True)
        
        self.handler_group.addButton(self.btnVertical, 0)
        self.handler_group.addButton(self.btnHorizontal, 1)
        
        handler_layout.addWidget(self.btnVertical)
        handler_layout.addWidget(self.btnHorizontal)
        main_layout.addWidget(handler_group)

        # Preview table
        preview_label = QLabel("Data Preview:")
        preview_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(preview_label)
        
        self.preview_table = QTableView()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setEditTriggers(QTableView.NoEditTriggers)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setMaximumHeight(200)
        main_layout.addWidget(self.preview_table)

        # Button box
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        main_layout.addWidget(self.buttonBox)

        self.setLayout(main_layout)

    def loadFileDF(self):
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
        if not file_path:
            print("No file selected.")
            return

        print(f"Loading data from: {file_path}")
        self.df = loadFile(file_path)

        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "Data Load Error", "Failed to load data or the file is empty.")
            return
            
        # Path-like column renaming
        rename_map_paths = {col: os.path.basename(col) for col in self.df.columns if '/' in col or '\\' in col}
        if rename_map_paths:
            self.df.rename(columns=rename_map_paths, inplace=True)
            
        # Update preview
        preview_model = PandasModel(self.df.head(10))
        self.preview_table.setModel(preview_model)
        
        # Enable OK button
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

    def accept(self):
        # Get selected options
        self.mainIdentifier = bool(self.identifier_group.checkedId())  # 0 = cols, 1 = rows
        self.mainHandler = bool(self.handler_group.checkedId())       # 0 = vertical, 1 = horizontal
        
        # Apply orientation changes if needed
        if self.mainIdentifier != self.mainHandler:
            # If identifiers are in rows but we want vertical handling, transpose
            # Or if identifiers are in columns but we want horizontal handling, transpose
            self.df = self.df.T
            print("Data transposed to match orientation")
            
        super().accept()

    def getResults(self):
        return self.df, self.mainIdentifier, self.mainHandler
