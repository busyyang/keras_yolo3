# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 575)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 781, 461))
        self.label.setText("")
        self.label.setObjectName("label")
        self.btnCamera = QtWidgets.QPushButton(self.centralwidget)
        self.btnCamera.setGeometry(QtCore.QRect(10, 490, 81, 31))
        self.btnCamera.setObjectName("btnCamera")
        self.textCamera = QtWidgets.QTextEdit(self.centralwidget)
        self.textCamera.setGeometry(QtCore.QRect(100, 490, 311, 31))
        self.textCamera.setInputMethodHints(QtCore.Qt.ImhNone)
        self.textCamera.setObjectName("textCamera")
        self.textFile = QtWidgets.QTextEdit(self.centralwidget)
        self.textFile.setGeometry(QtCore.QRect(520, 490, 271, 31))
        self.textFile.setInputMethodHints(QtCore.Qt.ImhNone)
        self.textFile.setObjectName("textFile")
        self.btnFile = QtWidgets.QPushButton(self.centralwidget)
        self.btnFile.setGeometry(QtCore.QRect(430, 490, 81, 31))
        self.btnFile.setObjectName("btnFile")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnCamera.setText(_translate("MainWindow", "打开摄像头"))
        self.btnFile.setText(_translate("MainWindow", "打开文件"))
