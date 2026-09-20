from main_window import MainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
from ui.theme import apply as apply_theme
from ui.loading_splash import LoadingSplashScreen
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_theme(app)

    # 显示加载进度条
    splash = LoadingSplashScreen()
    splash.show()
    screen = app.primaryScreen().geometry()
    x = (screen.width() - splash.width()) // 2
    y = (screen.height() - splash.height()) // 2
    splash.move(x, y)
    QApplication.processEvents()

    def on_progress(percent, message):
        splash.update_progress(percent, message)

    ui = MainWindow(progress_callback=on_progress)
    
    QApplication.processEvents()
    
    ui.showFullScreen()
    ui._init_all_tabs()
    splash.update_progress(100, "启动完成")
    splash.close()
    QApplication.processEvents()
    sys.exit(app.exec_())


    