import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication
from PyQt5.QtCore import QCoreApplication


class Example(QWidget): # 继承

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        qbtn = QPushButton('Quit', self) # 创建一个button



        qbtn.clicked.connect(QCoreApplication.instance().quit)


        qbtn.resize(qbtn.sizeHint()) # 默认大小
        qbtn.move(50, 50)

        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Quit button')
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example() # 创建一个实例
    sys.exit(app.exec_())