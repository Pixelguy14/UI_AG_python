from PyQt5.QtWidgets import (QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                            QHeaderView, QDialog, QComboBox, QLabel, 
                            QDialogButtonBox, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt
import pandas as pd

class DialogMetadataModel(QDialog):
    def __init__(self, df, lM, lS, parent=None):
        super().__init__(parent)
        self.df = df
        self.listMeta = lM # list of the headers in metadata dataframe
        self.listSamp = lS # list of the headers in samples dataframe
        self.assignments = {col: "undefined" for col in df.columns}
        self.setWindowTitle("Assign Column Types")
        self.setMinimumSize(600, 400)
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("Assign Column Types")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        main_layout.addWidget(title_label)

        # Create table widget
        self.table = QTableWidget(len(self.df.columns), 2)
        self.table.setHorizontalHeaderLabels(["Column Header", "Type Assignment"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                gridline-color: #eee;
            }
            QHeaderView::section {
                background-color: #94a7cb;
                color: white;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
        """)

        # Populate table
        set_listMeta = set(self.listMeta)
        set_listSamp = set(self.listSamp)
        for row, col_name in enumerate(self.df.columns):
            # Column headers
            col_item = QTableWidgetItem(col_name)
            col_item.setFlags(col_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, col_item)
            
            # Type assignment widget
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(5, 2, 5, 2)
            layout.setSpacing(5)
            
            # Type combo box
            type_combo = QComboBox()
            type_combo.addItems(["undefined", "metadata", "sample"])
            detected_type = "undefined"
            if col_name in set_listMeta:
                detected_type = "metadata"
            elif col_name in set_listSamp:
                detected_type = "sample"
            self.assignments[col_name] = detected_type
            type_combo.setCurrentText(detected_type)
            #type_combo.setCurrentText(self.assignments[col_name])
            type_combo.currentTextChanged.connect(
                lambda text, c=col_name: self.updateAssignment(c, text)
            )
            type_combo.setStyleSheet("""
                QComboBox {
                    padding: 5px;
                    border: 1px solid #ddd;
                    border-radius: 3px;
                    background-color: white;
                }
            """)
            layout.addWidget(type_combo)
            
            # Remove button
            remove_btn = QPushButton("✕")
            remove_btn.setFixedSize(25, 25)
            remove_btn.setProperty("column", col_name)
            remove_btn.clicked.connect(self.removeColumn)
            remove_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border-radius: 3px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #9c1b28;
                }
            """)
            layout.addWidget(remove_btn)
            
            self.table.setCellWidget(row, 1, widget)
            self.table.setRowHeight(row, 45)

        main_layout.addWidget(self.table)

        # Add instruction label
        self.instruction_label = QLabel("If left Undefined the column will default to Sample")
        self.instruction_label.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.instruction_label)

        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.setStyleSheet("""
            QDialogButtonBox {
                background-color: transparent;
            }
        """)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    def updateAssignment(self, column, assignment):
        self.assignments[column] = assignment

    def removeColumn(self):
        button = self.sender()
        column = button.property("column")
        
        # Remove the row from the table widget
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == column:
                self.table.removeRow(row)
                self.assignments[column] = "removed"
                break

    def getResults(self):
        # Create metadata and sample dataframes
        metadata_cols = [col for col, assign in self.assignments.items() 
                        if assign == "metadata" and col in self.df.columns]
        sample_cols = [col for col, assign in self.assignments.items() 
                    if assign == "sample" or assign == "undefined" and col in self.df.columns]
        
        # Create original dataframe with removed columns
        removed_cols = [col for col, assign in self.assignments.items() 
                        if assign == "removed" and col in self.df.columns]
        
        # Create the dataframes
        df_metadata = self.df[metadata_cols].copy() if metadata_cols else pd.DataFrame()
        df_sample = self.df[sample_cols].copy() if sample_cols else pd.DataFrame()
        df_original = self.df.drop(columns=removed_cols).copy() if removed_cols else self.df.copy()
        
        return df_metadata, df_sample, df_original
    