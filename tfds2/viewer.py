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
import tfds2.logging
from PIL import Image


class Camera:
    """ Free flight camera.

    Args:
        init_pos (Vec3): Initial camera position
    """
    def __init__(self, init_pos):
        self.init_pos = np.array(init_pos, np.float32)
        assert self.init_pos.shape == (3, )
        self.pos = self.init_pos
        self.reset()

    def reset(self):
        self.pos = self.init_pos
        self.Q = np.array([1, 0, 0, 0], np.float64)

        # Camera vector: camera points in -z direction initially.
        self.c_r = np.array([1, 0, 0], np.float64)
        self.c_u = np.array([0, 1, 0], np.float64)
        self.c_f = np.array([0, 0, 1], np.float64)

    def prodQuatVec(self, q, v):
        """Return vector that corresponds to `v` rotated by `q`"""
        t = 2 * np.cross(q[1:], v)
        return v + q[0] * t + np.cross(q[1:], t)

    def prodQuatQuat(self, q0, q1):
        """Return product of two Quaternions (q0 * q1)"""
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        m = [
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            +x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            +x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
        ]
        return np.array(m, dtype=np.float64)

    def update(self, phi: float, theta: float, dx, dy, dz) -> None:
        """Update camera vectors."""
        # Compute the Quaternions to rotate around x (theta) and y (phi).a
        sp, cp = np.sin(phi / 2), np.cos(phi / 2)
        st, ct = np.sin(theta / 2), np.cos(theta / 2)
        q_phi = np.array([ct, st, 0, 0], np.float64)
        q_theta = np.array([cp, 0, sp, 0], np.float64)

        # Combine the phi/theta update, then update our camera Quaternion.
        q_rot = self.prodQuatQuat(q_phi, q_theta)
        self.Q = self.prodQuatQuat(q_rot, self.Q)

        # Create the latest camera vectors.
        self.c_r = self.prodQuatVec(self.Q, np.array([1, 0, 0], np.float64))
        self.c_u = self.prodQuatVec(self.Q, np.array([0, 1, 0], np.float64))
        self.c_f = self.prodQuatVec(self.Q, np.array([0, 0, 1], np.float64))

        # Compute new position based on latest camera vectors.
        self.pos += -dz * self.c_f + dx * self.c_r

    def getCameraVectors(self):
        """Return the right, up, forward, and position vector"""
        return self.c_r, self.c_u, self.c_f, self.pos


class ClassifiedImageLabel(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit = tfds2.logging.getLogger('Viewer')
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.setScaledContents(True)
        self.ml_regions = []

    def setMLRegions(self, regions):
        try:
            assert isinstance(regions, (tuple, list, np.ndarray))
            if len(regions) == 0:
                self.ml_regions.clear()
                return True
            regions = np.array(regions)
            assert regions.ndim == 2
            rect = np.array(regions[:, :4], np.float32)
            rect = np.clip(rect, 0, 1)
            rgba = 255 * np.array(regions[:, 4:8], np.float32)
            rgba = np.clip(rgba, 0, 255)
            rgba = rgba.astype(np.int32)
            txt = [str(_) for _ in regions[:, 8]]
        except (IndexError, AssertionError, ValueError):
            self.logit.warn('Regions are invalid')
            return False

        rect, rgba = rect.tolist(), rgba.tolist()
        self.ml_regions = [_ for _ in zip(rect, rgba, txt)]
        return True

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QtGui.QPainter(self)

        width, height = self.rect().width(), self.rect().height()
        for rect, rgba, txt in self.ml_regions:
            painter.setPen(QtGui.QPen(QtGui.QColor(*rgba)))

            x, w = rect[0] * width, rect[2] * width
            y, h = rect[1] * height, rect[3] * height
            painter.drawRect(x, y, w, h)
            painter.drawText(x, y, txt)


class ViewerWidget(QtWidgets.QWidget):
    """Show one camera. This widget is usually embedded in a parent widget."""
    def __init__(self, parent, camera, host, port):
        super().__init__(parent)
        self.logit = tfds2.logging.getLogger('Viewer')

        assert isinstance(camera, str)
        self.camera_name = camera
        self.host = f'http://{host}:{port}'

        # Labels to display the scene image.
        self.label_img = ClassifiedImageLabel()
        self.label_fetch = QtWidgets.QLabel('Classify: 0 ms')
        self.label_classify = QtWidgets.QLabel('Fetch: 0 ms')

        # Add the just created display elements into a layout.
        layout = QtWidgets.QVBoxLayout()
        layout_bar = QtWidgets.QHBoxLayout()
        layout_bar.addWidget(self.label_fetch)
        layout_bar.addWidget(self.label_classify)
        layout.addWidget(self.label_img)
        layout.addLayout(layout_bar)
        self.setLayout(layout)
        self.width, self.height = 512, 512

        # Create camera.
        self.camera = Camera(init_pos=[2, 0, 15])

        # Camera movement flags.
        self.movement = {'strafe': 0, 'forward': 0, 'slow': False}

        # If True, the mouse will control the camera instead of the cursor on
        # the desktop GUI.
        self.pos_before_grab = None
        self.mouseGrab = False

        # Start the timer.
        self.drawTimer = self.startTimer(500)
        self.last_ts = time.time()

    def setMLRegions(self, regions):
        self.label_img.setMLRegions(regions)

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
        self.mouseGrab = not self.mouseGrab
        c = self.cursor()
        if self.mouseGrab:
            self.pos_before_grab = c.pos()
            self.centerCursor()
            c.setShape(QtCore.Qt.BlankCursor)
        else:
            c.setPos(self.pos_before_grab)
            c.setShape(QtCore.Qt.ArrowCursor)
        self.setCursor(c)

    def updateLocalCamera(self):
        if not self.mouseGrab or self.pos_before_grab is None:
            return

        # Get current cursor position.
        xpos, ypos = self.cursor().pos().x(), self.cursor().pos().y()

        # Convert mouse offset from default position to left/up rotation, then
        # reset the cursor to its default position.
        sensitivity = 0.003
        self.centerCursor()
        phi = sensitivity * (self.pos_before_grab.x() - xpos)
        theta = sensitivity * (self.pos_before_grab.y() - ypos)
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

        # Install the image as the new pixmap for the label.
        self.label_img.setPixmap(QtGui.QPixmap(qimg))

    def timerEvent(self, event):
        self.killTimer(event.timerId())

        # Update the camera position.
        self.updateLocalCamera()
        if not self.updateServerCamera():
            self.drawTimer = self.startTimer(5000)
            return

        # Fetch the next frame.
        t0 = time.time()
        img = self.fetchNextFrame()
        etime = int(1000 * (time.time() - t0))
        self.label_fetch.setText(f'Fetch: {etime:,} ms')

        if img is None:
            self.drawTimer = self.startTimer(5000)
            return

        # Pass it to the (overloaded) classifier method.
        t0 = time.time()
        ml_img = self.classifyImage(img)
        etime = int(1000 * (time.time() - t0))
        self.label_classify.setText(f'Classify: {etime:,} ms')

        # Display the image.
        self.displayScene(img if ml_img is None else ml_img)

        # Reset the timer.
        self.drawTimer = self.startTimer(100)


class MainWindow(QtWidgets.QWidget):
    """Arrange the camera widgets."""
    def __init__(self, cameras: dict, host, port):
        super().__init__(parent=None)
        self.logit = tfds2.logging.getLogger('Viewer')

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
