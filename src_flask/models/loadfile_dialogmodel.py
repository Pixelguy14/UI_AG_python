from PyQt5.QtWidgets import (QPushButton, QVBoxLayout, QMessageBox, QDialog, QLabel, 
                             QRadioButton, QButtonGroup, QGroupBox, QTableView,
                             QDialogButtonBox, QFrame)
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
        identifier_group = QGroupBox("Select your Samples location")
        identifier_layout = QVBoxLayout(identifier_group)
        
        instruction_label = QLabel("Are the Samples located in columns or rows?")
        instruction_label.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        instruction_label.setAlignment(Qt.AlignCenter)
        identifier_layout.addWidget(instruction_label)
        
        # Radio buttons for identifier position
        self.identifier_group = QButtonGroup(self)
        self.btnCols = QRadioButton("Columns (samples as columns, features as rows)")
        self.btnRows = QRadioButton("Rows (samples as rows, features as columns)")
        self.btnCols.setChecked(True)
        
        self.identifier_group.addButton(self.btnCols, 0)
        self.identifier_group.addButton(self.btnRows, 1)
        
        identifier_layout.addWidget(self.btnCols)
        identifier_layout.addWidget(self.btnRows)

        instruction_label2 = QLabel("(This will transpose your dataset if the samples are set as rows)")
        instruction_label2.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        instruction_label2.setAlignment(Qt.AlignCenter)
        identifier_layout.addWidget(instruction_label2)

        main_layout.addWidget(identifier_group)

        # Preview table
        preview_label = QLabel("Data Preview:")
        preview_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(preview_label)
        
        # Preview Table
        self.preview_table = QTableView()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setEditTriggers(QTableView.NoEditTriggers)

        # Table behavior
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
                ("Comma-Separated Values", "*.csv")
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
        
        # Apply orientation changes if needed
        if self.mainIdentifier != 0:
            # we want samples as columns, features as rows
            self.df = self.df.T
            print("Data transposed to match the interface orientation")
        super().accept()
        #print(self.df.head(10))

    def getResults(self):
        return self.df
