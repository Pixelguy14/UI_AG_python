from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QDialog, QMessageBox, QLabel, QDialogButtonBox, QSlider
from PyQt5.QtCore import Qt
import math

## QDialog with a slider to select % of threshold
class DialogThresholdingModel(QDialog):
    def __init__(self, dfS, dfST, parent=None):
        super().__init__(parent)
        self.dfS = dfS # Dataframe containing sample data
        self.dfST = dfST # Dataframe to store the thresholded sample data

        self.setWindowTitle("Sample Thresholding")
        self.setMinimumSize(400, 250)
        self.initUI()
        
        self.curr_thresh_perc = 80

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("Select Thresholding Percentage")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 5px;")
        title_label.setAlignment(Qt.AlignCenter) # Center the title
        main_layout.addWidget(title_label)

        # Add instruction label
        self.instruction_label = QLabel("Move the slider to set the percentage threshold for non-null values.")
        self.instruction_label.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.instruction_label)
        
        # Slider and current percentage display
        slider_layout_widget = QWidget()
        slider_layout = QHBoxLayout(slider_layout_widget)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(10)

        self.slider = QSlider(Qt.Horizontal) # Specify orientation
        self.slider.setRange(0, 100) # Percentage from 0 to 100
        self.slider.setSingleStep(10) # 10% steps
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBothSides) # Show ticks
        self.slider.setTickInterval(10) # Tick every 10%
        self.slider.setValue(80) # Initial value
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px; /* the groove expands to the size of the slider by default. */
                background: #555;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #1b9c31; /* Blue handle */
                border: 1px solid #777;
                width: 18px;
                margin: -5px 0; /* handle is 16px wide, so -2px to make it grow out of the groove */
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #2ecc71; /* Green fill for the progress */
                border: 1px solid #777;
                height: 8px;
                border-radius: 4px;
            }
        """)
        
        self.percentage_label = QLabel("80%")
        self.percentage_label.setFixedWidth(50) # Give it a fixed width
        self.percentage_label.setStyleSheet("font-weight: bold; color: #333;")
        self.percentage_label.setAlignment(Qt.AlignCenter)

        self.slider.valueChanged.connect(self.updatePercentageLabel)
        
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.percentage_label)
        main_layout.addWidget(slider_layout_widget) # Add the widget containing slider and label

        # Action buttons (Confirm and Revert)
        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.setContentsMargins(0, 0, 0, 0)
        action_buttons_layout.setSpacing(10)
        action_buttons_layout.addStretch(1) # Push buttons to the right

        # Confirm Thresholding button
        btnConfirm = QPushButton("Apply Thresholding")
        btnConfirm.clicked.connect(self.confirmThresholding)
        btnConfirm.setStyleSheet("""
            QPushButton {
                background-color: #35dc59;
                color: white;
                border-radius: 5px;
                font-weight: bold;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #1b9c31;
            }
        """)
        action_buttons_layout.addWidget(btnConfirm)

        # Revert Thresholding button
        btnRevertThresholding = QPushButton("Revert to Original")
        btnRevertThresholding.clicked.connect(self.revertThresholding)
        btnRevertThresholding.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c; /* Red for revert */
                color: white;
                border-radius: 5px;
                font-weight: bold;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        action_buttons_layout.addWidget(btnRevertThresholding)
        action_buttons_layout.addStretch(1) # Push buttons to the left

        main_layout.addLayout(action_buttons_layout)

        # Standard OK/Cancel buttons at the bottom
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.setStyleSheet("""
            QDialogButtonBox {
                background-color: transparent;
            }
            QPushButton {
                background-color: #3498db; /* Blue for Ok/Cancel */
                color: white;
                border-radius: 5px;
                font-weight: bold;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1a527f;
            }
        """)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    def updatePercentageLabel(self, value):
        self.curr_thresh_perc = value
        self.percentage_label.setText(f"{value}%")

    def confirmThresholding(self):
        reply = QMessageBox.warning(
            self,
            "Confirm Thresholding",
            "Applying thresholding will modify the sample data.\nDo you want to proceed?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            
            num_columns = len(self.dfS.columns)
            # Ensure at least 1 column is considered if percentage is > 0 and num_columns > 0
            if num_columns > 0:
                threshold = math.ceil((self.curr_thresh_perc / 100.0) * num_columns)
                if self.curr_thresh_perc > 0 and threshold == 0:
                    threshold = 1
            else:
                threshold = 0 # No columns to threshold

            self.dfST = self.dfS.dropna(thresh=threshold)

            QMessageBox.information(self, "Thresholding Applied", "Sample data has been thresholded successfully.")
            print(f"Sample Data thresholded with {self.curr_thresh_perc}% threshold.")
            print(f"Original shape: {self.dfS.shape}, thresholded shape: {self.dfST.shape}")
            
            self.accept() # Close the dialog with accepted status

        else:
            QMessageBox.information(self, "Thresholding Cancelled", "Thresholding was not applied.")

    def revertThresholding(self):
        reply = QMessageBox.warning(
            self,
            "Confirm Revert",
            "Are you sure you want to revert to the original sample data?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.dfST = self.dfS.copy()
            QMessageBox.information(self, "Reverted", "Sample data has been reverted to its original state.")
            print("Sample Data Reverted to Original.")
        else:
            QMessageBox.information(self, "Revert Cancelled", "Revert operation cancelled.")


    def getResults(self):
        return self.dfS, self.dfST
