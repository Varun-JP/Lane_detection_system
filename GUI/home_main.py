# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'home.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(503, 382)
        Dialog.setStyleSheet("background-color:#2C3E50;")
        self.title_label = QtWidgets.QLabel(Dialog)
        self.title_label.setGeometry(QtCore.QRect(50, 30, 411, 71))
        self.title_label.setStyleSheet("color: white;\n"
"font: 20pt \"Times New Roman\";\n"
"border: 2px solid black;\n"
"border-radius: 5px;\n"
"padding: 2px;")
        self.title_label.setScaledContents(True)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setObjectName("title_label")
        self.try_2 = QtWidgets.QLabel(Dialog)
        self.try_2.setGeometry(QtCore.QRect(190, 140, 121, 61))
        self.try_2.setStyleSheet("color:white;\n"
"font: 14pt \"Times New Roman\";\n"
"border: 2px solid black;\n"
"border-radius: 5px;\n"
"padding: 2px;")
        self.try_2.setText("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">Try it on:</span></p></body></html>")
        self.try_2.setAlignment(QtCore.Qt.AlignCenter)
        self.try_2.setObjectName("try_2")
        self.image_button = QtWidgets.QPushButton(Dialog)
        self.image_button.setGeometry(QtCore.QRect(70, 250, 111, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.image_button.setFont(font)
        self.image_button.setMouseTracking(False)
        self.image_button.setStyleSheet("color:#3498DB;\n"
"font: bold 11pt \"Times New Roman\";\n"
"border: 2px solid #3498DB;\n"
"border-radius: 5px;\n"
"padding: 2px;\n"
"")
        self.image_button.setObjectName("image_button")
        self.video_button = QtWidgets.QPushButton(Dialog)
        self.video_button.setGeometry(QtCore.QRect(310, 250, 111, 51))
        self.video_button.setStyleSheet("color:#3498DB;\n"
"font: bold 11pt \"Times New Roman\";\n"
"border: 2px solid #3498DB;\n"
"border-radius: 5px;\n"
"padding: 2px;\n"
"")
        self.video_button.setObjectName("video_button")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Lane Detection System"))
        self.title_label.setText(_translate("Dialog", "Road Lane Detection System"))
        self.image_button.setText(_translate("Dialog", "Image"))
        self.video_button.setText(_translate("Dialog", "Video"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
