from PyQt5.QtWidgets import QMainWindow, QToolBar, QPushButton, QVBoxLayout, QWidget
from tube_view_widget import TubeViewWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Bend Wizard')
        self.resize(1000, 800)
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.view = TubeViewWidget()
        main_layout.addWidget(self.view)
        toolbar = QToolBar("View Controls")
        self.addToolBar(toolbar)
        fit_btn = QPushButton("Fit Inner Tube")
        fit_btn.clicked.connect(self.view.fit_inner_tube)
        toolbar.addWidget(fit_btn)
        self.setCentralWidget(central_widget)
