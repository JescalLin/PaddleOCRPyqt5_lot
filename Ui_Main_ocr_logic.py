import sys
import threading
import cv2
from PyQt5.QtCore import QThread
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton, QMessageBox 
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow
from Ui_Main_ocr import Ui_MainWindow

from paddleocr import PaddleOCR
import numpy as np
from collections import Counter
import imutils

cls_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_mobile_v2.0_cls_infer'
det_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_det_infer' 
rec_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_rec_infer'
# ocr = PaddleOCR(cls_model_dir=cls_model_dirs, det_model_dir=det_model_dirs ,rec_model_dir = rec_model_dirs,det_db_thresh=0.1,det_db_box_thresh=0.1,det_db_unclip_ratio=1.6)
ocr = PaddleOCR(
    cls_model_dir = cls_model_dirs, 
    det_model_dir = det_model_dirs,
    rec_model_dir = rec_model_dirs,
    )


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

    def slot_box(self):
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        height, width = frame.shape[:2]
        img = frame
        result = ocr.ocr(img)
        for i in range(len(result)):
            pts = np.array([result[i][0]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(0,255,0))
            if (result[i][1][0].isalnum()==True and len(result[i][1][0])>=10 and 'EXP' not in result[i][1][0]):
                cv2.putText(img, result[i][1][0][-10:], (int(result[i][0][3][0]), int(result[i][0][3][1])+40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                self.lineEdit_box.setText(result[i][1][0][-10:])
            self.plainTextEdit_log.appendPlainText(result[i][1][0])




        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = QtGui.QImage(img_rgb, width, height, QtGui.QImage.Format_RGB888)
        qimg = QtGui.QPixmap.fromImage(img_rgb)
        self.label_box.setPixmap(qimg)
        cap.release()
        cv2.destroyAllWindows()

    def slot_pic(self):
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        height, width = frame.shape[:2]
        img = frame
        result = ocr.ocr(img)
        for i in range(len(result)):
            pts = np.array([result[i][0]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(0,255,0))
            if (result[i][1][0].isalnum()==True and len(result[i][1][0])>=10 and 'EXP' not in result[i][1][0]):
                cv2.putText(img, result[i][1][0][-10:], (int(result[i][0][3][0]), int(result[i][0][3][1])+40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                self.lineEdit_pic.setText(result[i][1][0][-10:])
            self.plainTextEdit_log.appendPlainText(result[i][1][0])




        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = QtGui.QImage(img_rgb, width, height, QtGui.QImage.Format_RGB888)
        qimg = QtGui.QPixmap.fromImage(img_rgb)
        self.label_pic.setPixmap(qimg)
        cap.release()
        cv2.destroyAllWindows()

            
    def slot_paper(self):
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        height, width = frame.shape[:2]
        img = frame
        result = ocr.ocr(img)
        for i in range(len(result)):
            pts = np.array([result[i][0]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(0,255,0))
            if (result[i][1][0].isalnum()==True and len(result[i][1][0])>=10 and 'EXP' not in result[i][1][0]):
                cv2.putText(img, result[i][1][0][-10:], (int(result[i][0][3][0]), int(result[i][0][3][1])+40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                self.lineEdit_paper.setText(result[i][1][0][-10:])
            self.plainTextEdit_log.appendPlainText(result[i][1][0])




        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = QtGui.QImage(img_rgb, width, height, QtGui.QImage.Format_RGB888)
        qimg = QtGui.QPixmap.fromImage(img_rgb)
        self.label_paper.setPixmap(qimg)
        cap.release()
        cv2.destroyAllWindows()

    def slot_lot(self):
        pass


    
    def closeEvent(self, event):
        global running
        running = False
        print("stoped..")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())