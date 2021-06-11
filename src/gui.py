# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets, QtChart
import os
import re
    

class Ui_MainWindow(QtWidgets.QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(950, 614)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.chartWidget = QtChart.QChartView()
        self.chartWidget.setMinimumSize(QtCore.QSize(500, 500))
        self.chartWidget.setObjectName("chartWidget")
        self.gridLayout.addWidget(self.chartWidget, 0, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.pointList = QtWidgets.QListWidget(self.centralwidget)
        self.pointList.setObjectName("pointList")
        self.verticalLayout.addWidget(self.pointList)
        self.loadBtn = QtWidgets.QPushButton(self.centralwidget)
        self.loadBtn.setObjectName("loadBtn")
        self.verticalLayout.addWidget(self.loadBtn)
        self.gridLayout.addLayout(self.verticalLayout, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 950, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.loadBtn.clicked.connect(self.handleInput)
        self.data = []
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PointPrediction"))
        self.loadBtn.setText(_translate("MainWindow", "Załaduj dane"))

    def updatePointList(self):
        self.pointList.clear()
        for point in self.data:
            self.pointList.addItem(f"X: {point[0]}; Y: {point[1]}")

    def updateGraph(self):
        series = QtChart.QLineSeries()
        for point in self.data:
            series.append(point[0], point[1])
        chart = QtChart.QChart()
        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)
        chart.setTheme(QtChart.QChart.ChartThemeBlueCerulean)

        self.chartWidget.setChart(chart)
        



    def handleInput(self):
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, "Wybierz dane", os.getcwd(), "*.txt")
        with open(file_path[0]) as file:
            self.data = []
            for line in file.readlines():
                line = line.strip()
                if re.match(r"-?[0-9]+.[0-9]+\t-?[0-9]+.[0-9]+", line) == None:
                    message_box = QtWidgets.QMessageBox(self)
                    message_box.setText(u"Błędne dane wejściowe")
                    message_box.setWindowTitle(u"Błąd")
                    message_box.show()
                    return
                vec = line.split('\t')
                point = (float(vec[0]), float(vec[1]))
                self.data.append(point)
        self.updatePointList()
        self.normalizeInput()
        self.updateGraph()

    def normalizeInput(self):
        min_x = min(self.data, key= lambda p: p[0])[0]
        min_y = min(self.data, key= lambda p: p[1])[1]
        max_x = max(self.data, key= lambda p: p[0])[0]
        max_y = max(self.data, key= lambda p: p[1])[1]
        result = map(lambda p: (-1.0 + (p[0] - min_x) * 2.0 / (max_x - min_x), 
        -1.0 + (p[1] - min_y) * 2.0 / (max_y - min_y)), self.data)
        self.data = list(result)
        




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
