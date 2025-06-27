# source /home/pixel/Documents/Cinvestav_2025/UI_Analisis_Genomico/UI_AG_python/bin/activate
import sys
from PyQt5.QtWidgets import (QApplication)
import os
import sys
if sys.platform.startswith('linux'):
    if os.environ.get('QT_QPA_PLATFORM') != 'xcb':
        os.environ['QT_QPA_PLATFORM'] = 'xcb'

from src.views.mainView import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = mainView()
    main_window.show()
    sys.exit(app.exec_())
