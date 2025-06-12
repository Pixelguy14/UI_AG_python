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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.impute import KNNImputer

from src.views.mainView import *
"""
df = loadDF_Consensus("/home/pixel/Documents/Cinvestav_2025/UI_Analisis_Genomico/Consensus/normalized.consensusXML")
print(df)

new_header = [x for x in df]

new_data = df[new_header[3:]]

mean_TIC = new_data.mean(0)
# plot bars for first N columns, useful if large number of samples
N = 12
ind = np.arange(N)
plt.figure(figsize=(15,3))
plt.bar(ind, mean_TIC[:N], width=0.7, color="purple", alpha=0.4)
plt.ylabel("Average log2 (intensity)")
plt.title("Ave log2 Intensity, up to %d samples" %N)
plt.xticks(ind, new_data.columns[:N], rotation=-90)
plt.show()
"""
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = mainView()
    main_window.show()
    sys.exit(app.exec_())
