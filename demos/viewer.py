import sys
import ds2sim.viewer
import numpy as np

import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets


# For convenience.
QPen, QColor, QRectF = QtGui.QPen, QtGui.QColor, QtCore.QRectF
DS2Text = ds2sim.viewer.DS2Text


class MyClassifier(ds2sim.viewer.ClassifierCamera):
    def classifyImage(self, img):
        # `img` is always a <height, width, 3> NumPy image.
        assert img.dtype == np.uint8

        # Pass the image to your ML model.
        # myAwesomeClassifier(img)

        # Define a red bounding box.
        x, y, width, height = 0.3, 0.4, 0.3, 0.3
        bbox = [QPen(QColor(255, 0, 0)), QRectF(x, y, width, height)]

        # Define a green text label.
        x, y = 0.3, 0.4
        text = [QPen(QColor(100, 200, 0)), DS2Text(x, y, 'Found Something')]

        # Install the overlays.
        self.setMLOverlays([bbox, text])


# Qt boilerplate to start the application.
app = QtWidgets.QApplication(sys.argv)
widget = MyClassifier('Camera', host='127.0.0.1', port=9095)
widget.show()
app.exec_()
