from main_window import MainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
from qt_material import apply_stylesheet
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MainWindow()
    apply_stylesheet(app, theme = "dark_teal.xml")
    screen = app.primaryScreen()
    ui.showFullScreen()  
    ui._init_all_tabs()
    sys.exit(app.exec_())


    
