# encoding: utf-8
from GUI import Ui_MainWindow
from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QMainWindow, QGridLayout, QFileDialog
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys, cv2, threading
from yolo import YOLO
from PIL import Image, ImageQt
from timeit import default_timer as timer


class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-clear-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()


class AppWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(AppWindow, self).__init__(parent)
        self.setupUi(self)
        self.btnCamera.clicked.connect(self.show_camera)
        self.btnFile.clicked.connect(self.show_file)
        self.yolo = YOLO()

    def show_file(self):
        self.isCamera = False
        self.open()

    def show_camera(self):
        self.isCamera = True
        self.open()

    def open(self):
        if not self.isCamera:
            self.cap = cv2.VideoCapture(self.textFile.toPlainText())
            self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.cap = cv2.VideoCapture(self.textCamera.toPlainText())
            self.cap = CameraBufferCleanerThread(self.cap)
        th = threading.Thread(target=self.display)
        th.start()

    def display(self):
        if self.isCamera:
            while True:
                if self.cap.last_frame is not None:
                    frame = cv2.cvtColor(self.cap.last_frame, cv2.COLOR_RGB2BGR)
                    img = Image.fromarray(frame)
                    d = self.yolo.detect_image(img)
                    d = ImageQt.ImageQt(d)  # 转化为Qt对象
                    self.label.setPixmap(QPixmap.fromImage(d))
                cv2.waitKey(1)

        else:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                # RGB转BGR
                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    img = Image.fromarray(frame)
                    d = self.yolo.detect_image(img)
                    d = ImageQt.ImageQt(d)  # 转化为Qt对象
                    self.label.setPixmap(QPixmap.fromImage(d))
                if self.isCamera:
                    cv2.waitKey(1)
                else:
                    cv2.waitKey(int(1000 / self.frameRate))
            self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AppWindow()
    win.show()
    sys.exit(app.exec_())
