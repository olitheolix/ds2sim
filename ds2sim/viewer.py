""" View Cameras in Qt.

This script shows the rendered simulation in a Qt application.
The Qt interface uses the REST API to control the cameras and request images.
"""
import io
import time
import json
import requests
import numpy as np
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import ds2sim.camera
import ds2sim.logger
from PIL import Image


class DS2Text:
    def __init__(self, x, y, text):
        assert isinstance(x, (int, float)) and 0 <= x <= 1
        assert isinstance(y, (int, float)) and 0 <= y <= 1
        assert isinstance(text, str)
        self.x, self.y = x, y
        self.text = text


class ClassifiedImageLabel(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit = ds2sim.logger.getLogger('Viewer')
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.setScaledContents(True)
        self.ml_overlays = []

    def setMLOverlays(self, overlays):
        self.ml_overlays.clear()
        if not isinstance(overlays, (tuple, list)):
            return False

        ok = True
        for el in overlays:
            try:
                pen, data = el
                assert isinstance(pen, QtGui.QPen)
                assert isinstance(data, (QtCore.QRectF, DS2Text))
                self.ml_overlays.append((pen, data))
            except (ValueError, TypeError, AssertionError):
                ok = False
        return ok

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QtGui.QPainter(self)
        width, height = self.rect().width(), self.rect().height()

        # Draw each element specified by the user.
        for wpen, wtype in self.ml_overlays:
            painter.setPen(wpen)
            if isinstance(wtype, QtCore.QRectF):
                x, y = wtype.x() * width, wtype.y() * height
                w, h = wtype.width() * width, wtype.height() * height
                painter.drawRect(x, y, w, h)
            elif isinstance(wtype, DS2Text):
                painter.drawText(wtype.x * width, wtype.y * height, wtype.text)
            else:
                print('Unknown', wtype)


class ViewerWidget(QtWidgets.QWidget):
    """Show one camera. This widget is usually embedded in a parent widget."""
    def __init__(self, parent, camera, host, port):
        super().__init__(parent)
        self.logit = ds2sim.logger.getLogger('Viewer')

        assert isinstance(camera, str)
        self.camera_name = camera
        self.host = f'http://{host}:{port}'

        # Labels to display the scene image.
        self.label_img = ClassifiedImageLabel()
        self.label_name = QtWidgets.QLabel(camera)

        # Add the just created display elements into a layout.
        layout = QtWidgets.QVBoxLayout()
        layout_bot = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label_name)
        layout.addWidget(self.label_img)
        layout.addLayout(layout_bot)

        # Ensure the labels do not grow vertically.
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.label_name.setSizePolicy(sizePolicy)

        self.setLayout(layout)
        self.width, self.height = 512, 512

        # Create camera.
        self.camera = ds2sim.camera.FreeFlight(init_pos=[2, 0, 15])

        # Camera movement flags.
        self.movement = {'strafe': 0, 'forward': 0, 'slow': False}

        # If True, the mouse will control the camera instead of the cursor on
        # the desktop GUI.
        self.pos_before_grab = None
        self.mouse_grab = False

        # Start the timer.
        self.drawTimer = self.startTimer(500)
        self.last_ts = time.time()
        self.frame_cnt = 0

    def setMLOverlays(self, overlays):
        return self.label_img.setMLOverlays(overlays)

    def centerCursor(self):
        """Place the cursor in the pre-defined center position. """
        if not self.pos_before_grab:
            return
        c = self.cursor()
        c.setPos(self.pos_before_grab)
        c.setShape(QtCore.Qt.BlankCursor)
        self.setCursor(c)

    def cameraTranslateStartEvent(self, event):
        """ Set the movement flags associated with `key`."""
        # Toggle the slow flag.
        if event.modifiers() == QtCore.Qt.ShiftModifier:
            self.movement['slow'] = not self.movement['slow']

        key = event.text()
        forward = strafe = 0
        if key == 'e':
            forward += 1
        elif key == 'd':
            forward -= 1
        elif key == 'f':
            strafe += 1
        elif key == 's':
            strafe -= 1
        elif key == 'r':
            self.camera.reset()
        elif key == 'b':
            f = 1 * np.random.uniform(0, 1, size=4)
            f = {ii + 1: val for ii, val in enumerate(f.tolist())}
            self.rclient.setThrusters({1: f})
        elif key == 'p':
            r, u, f, p = self.camera.getCameraVectors()
            print('Camera:')
            print(' Right   : {:.2f} {:.2f} {:.2f}'.format(*r))
            print(' Up      : {:.2f} {:.2f} {:.2f}'.format(*u))
            print(' Forward : {:.2f} {:.2f} {:.2f}'.format(*f))
            print(' Position: {:.2f} {:.2f} {:.2f}'.format(*p))
        else:
            return

        self.movement['forward'] = forward
        self.movement['strafe'] = strafe

    def cameraTranslateStopEvent(self, event):
        """ Clear the movement flags associated with `key`."""
        key = event.text()
        if key in ['e', 'd']:
            self.movement['forward'] = 0
        elif key in ['f', 's']:
            self.movement['strafe'] = 0
        else:
            pass

    def mousePressEvent(self, event):
        self.mouse_grab = not self.mouse_grab
        c = self.cursor()
        if self.mouse_grab:
            self.pos_before_grab = c.pos()
            self.centerCursor()
            c.setShape(QtCore.Qt.BlankCursor)
        else:
            c.setPos(self.pos_before_grab)
            c.setShape(QtCore.Qt.ArrowCursor)
        self.setCursor(c)

    def updateLocalCamera(self):
        if not self.mouse_grab or self.pos_before_grab is None:
            return

        # Get current cursor position.
        xpos, ypos = self.cursor().pos().x(), self.cursor().pos().y()

        # Convert mouse offset from default position to left/up rotation, then
        # reset the cursor to its default position.
        sensitivity = 0.003
        self.centerCursor()
        phi = sensitivity * (self.pos_before_grab.x() - xpos)
        theta = -sensitivity * (self.pos_before_grab.y() - ypos)
        dz, dx = self.movement['forward'], self.movement['strafe']
        if self.movement['slow']:
            dz, dx = 0.02 * dz, 0.02 * dx

        # Send the new camera position to the Horde host.
        self.camera.update(phi, theta, dx, 0, dz)

    def updateServerCamera(self):
        right, up, _, pos = self.camera.getCameraVectors()
        payload = {'right': right.tolist(), 'up': up.tolist(), 'pos': pos.tolist()}
        data = {'data': json.dumps({self.camera_name: payload})}
        try:
            ret = requests.post(self.host + '/set-camera', data=data)
        except (TypeError, requests.exceptions.ConnectionError):
            self.logit.warn('Connection Error')
            return False

        if ret.status_code != 200:
            self.warn('Invalid request')
            return False
        return True

    def fetchNextFrame(self):
        payload = {'camera': self.camera_name, 'width': 512, 'height': 512}
        data = {'data': json.dumps(payload)}
        try:
            ret = requests.post(self.host + '/get-render', data=data)
        except (TypeError, requests.exceptions.ConnectionError):
            self.logit.warn('Connection Error')
            return

        try:
            img = Image.open(io.BytesIO(ret.content))
        except OSError:
            self.logit.error('Server returned invalid JPG image')
            return None
        return np.array(img.convert('RGB'), np.uint8)

    def classifyImage(self, img):
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def drawBBoxes(self, img, boxes):
        pass

    def displayScene(self, img):
        # Sanity check: must be an 8Bit RGB image.
        try:
            assert isinstance(img, np.ndarray)
            assert img.dtype == np.uint8
            assert len(img.shape) == 3
            assert img.shape[2] == 3
        except AssertionError:
            self.logit.warn('Image data for Pixmap is invalid')
            return

        # Convert the Image to QImage.
        qimg = QtGui.QImage(
            img.data, img.shape[1], img.shape[0], img.strides[0],
            QtGui.QImage.Format_RGB888)

        # Display the scene on the label.
        self.label_img.setPixmap(QtGui.QPixmap(qimg))

    def timerEvent(self, event):
        # Acknowledge timer.
        self.killTimer(event.timerId())
        self.frame_cnt += 1

        # Record the screen.
        # self.grab().save(f'/tmp/delme_{self.frame_cnt:04d}.jpg')

        # Update the camera position.
        self.updateLocalCamera()
        if not self.updateServerCamera():
            self.drawTimer = self.startTimer(5000)
            return

        # Fetch the next frame. If we could not get one (possibly because the
        # server has not been start, wait 5s until retry).
        img = self.fetchNextFrame()
        if img is None:
            self.drawTimer = self.startTimer(5000)
            return

        # Pass it to the user-overloaded classifier method.
        ml_img = self.classifyImage(img)

        # Display the image.
        self.displayScene(img if ml_img is None else ml_img)

        # Reset the timer.
        self.drawTimer = self.startTimer(50)


class MainWindow(QtWidgets.QWidget):
    """Arrange the camera widgets."""
    def __init__(self, cameras: dict, host='127.0.0.1', port=9095):
        super().__init__(parent=None)
        self.logit = ds2sim.logger.getLogger('Viewer')
        self.setWindowTitle('DS2 Demo')
        self.move(100, 100)

        # Points to widget that has mouse grab.
        self.active_camera = None

        # Put one ViewerWidget per camera into layout.
        self.viewers = []
        layout = QtWidgets.QHBoxLayout()
        for cname, widget in cameras.items():
            assert isinstance(cname, str)
            viewer = widget(self, cname, host, port)
            viewer.installEventFilter(self)
            layout.addWidget(viewer)
            self.viewers.append(viewer)
        self.setLayout(layout)

        # Start the timer.
        self.drawTimer = self.startTimer(0)
        self.last_ts = time.time()
        self.init = True

    def keyPressEvent(self, event):
        """Propagate key presses to the active ViewerWidget"""
        char = event.text()
        char = char[0] if len(char) > 1 else char
        if char == 'q':
            self.close()
            return

        if self.active_camera:
            self.active_camera.cameraTranslateStartEvent(event)

    def keyReleaseEvent(self, event):
        """Propagate key releases to the active ViewerWidget"""
        char = event.text()
        char = char[0] if len(char) > 1 else char
        if self.active_camera:
            self.active_camera.cameraTranslateStopEvent(event)

    def eventFilter(self, obj, event):
        """Intercept mouse clicks to (de)activate the active ViewerWidget"""
        if event.type() == QtCore.QEvent.MouseButtonPress:
            # The click was most likely on QLabel object (the one with the
            # image). From there we need to move up until we find the
            # ViewerWidget instance that harbours it. If there is one, then the
            # key{press,release}Event methods can deliver the camera movement
            # events to that widget.
            self.active_camera = None
            while obj is not self:
                if obj in self.viewers:
                    self.active_camera = obj
                    break
                obj = obj.parent()

        # Propagate the event down the chain.
        return False

    def timerEvent(self, event):
        self.killTimer(event.timerId())

        # Move the cursor into the middle of the widget, but not before the
        # camera widget has actually drawn itself. The check with a minimum
        # width of 400 is hacky, but suffices for now.
        if self.init and self.geometry().width() > 400:
            self.init = False
            # Move cursor into the widget (improves usability).
            self.cursor().setPos(self.geometry().center())
        else:
            self.drawTimer = self.startTimer(50)
