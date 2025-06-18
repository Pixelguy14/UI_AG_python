# source /home/pixel/Documents/Cinvestav_2025/UI_Analisis_Genomico/UI_AG_python/bin/activate
import sys
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
                            QTableView, QHeaderView, QToolBar, QAction, QDialog, QFormLayout, 
                            QLineEdit, QDateEdit, QComboBox, QMessageBox, QLabel, QSplitter,
                            QDialogButtonBox, QApplication, QFrame, QGridLayout, QSizePolicy, 
                            QStackedWidget, QSpinBox)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QDate, QSize
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel, QSqlQueryModel
from datetime import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyopenms as oms
import pandas as pd
import math
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.impute import KNNImputer
if sys.platform.startswith('linux'):
    if os.environ.get('QT_QPA_PLATFORM') != 'xcb':
        os.environ['QT_QPA_PLATFORM'] = 'xcb'

from src.views.mainView import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = mainView()
    main_window.show()
    sys.exit(app.exec_())
