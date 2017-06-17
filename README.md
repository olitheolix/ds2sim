# DS2Server

A simple Space Sim to experiment with machine learning.

It wraps [Horde3D](http://www.horde3d.org/) with Cython and uses it to render
scenes like these.

<img src="docs/img/example_scene.jpg" width="400">


You can use the engine to create your own training data, but is probably easier
to use the one from [DS2Data](https://github.com/olitheolix/ds2data). It has
a representative set of the objects in the scene, as well as a pre-rendered
flight path to test your model.


Once you have your model you can plug it into the Viewer application (see
[View The Space Simulation](#View-The-Space-Simulation) section for details).
Then fly through the scene and find out how well (and quickly) it identifies
the objects while flying through the scene. You may also replace the manual
controls with another AI to fly through the scene on its own.


## Installation
First, this will only work an NVidia GPU, because the project uses
[headless rendering](https://devblogs.nvidia.com/parallelforall/egl-eye-opengl-visualization-without-x-server/).
If you do not have one, you may still train and test ML models with the data
set at [DS2Data](https://github.com/olitheolix/ds2data).

Next, since this is a Cython wrapper for Horde3D, you will need Cython and
Horde3D - who would have guessed. To compile and install Horde3D, activate the
virtual environment of your choice, install Cython, and then clone/compile/install
Horde3D like so:
```bash
git clone https://github.com/olitheolix/Horde3D
mkdir -p Horde3D/build
cd Horde3D/build
git checkout ds2
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=`python -c 'import sys; print(sys.prefix)'`
make install
```

Afterwards, install *DS2Sim* with PIP:
```bash
pip install ds2sim
```

## View the Space Simulation
This consists of two parts: a web server to supply the rendered images, and a
Qt application to display them.

Start the server with:
```bash
ds2server
```

Then put the following code into a file and run it. Note that almost the entire
file is boilerplate for Qt.
```python
import sys
import ds2server.viewer
import numpy as np

import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets


# Ignore for now.
class MyViewer(ds2server.viewer.ViewerWidget):
    def classifyImage(self, img):
        pass


# Define a camera name and assign our custom widget to manage it.
cameras = {'Cam1': MyViewer}

# Qt boilerplate to start the application.
app = QtWidgets.QApplication(sys.argv)
widget = ds2server.viewer.MainWindow(cameras, host='127.0.0.1', port=9095)
widget.show()
app.exec_()
```

This should produce a Qt application that "flies" you through a pre-rendered
scene.

The only lines that are not boilerplate are the `MyViewer` widget (which does
nothing right now), and the definition of a `camera` dictionary. That dictionary
specifies which camera will be controlled by which widget. In this particular
example, we just create a simple camera and use the default widget, which will
merely display the scene.


## Plug Your ML Model Into The Simulation
The real fun is, of course, to use ML to find and identify all the cubes. To do
so, overload the `classifyImage` method in the previous demo like so:

```python
# For convenience.
QPen, QColor, QRectF = QtGui.QPen, QtGui.QColor, QtCore.QRectF
DS2Text = ds2server.viewer.DS2Text

class MyViewer(ds2server.viewer.ViewerWidget):
    def classifyImage(self, img):
        # `img` is always a <height, width, 3> NumPy image.
        assert img.dtype == np.uint8

        # Pass the image to your ML model.
        # myAwesomeClassifier(img)

        # Define a red bounding box.
        x, y, width, height = 0.4, 0.2, 0.4, 0.4
        bbox = [QPen(QColor(255, 0, 0)), QRectF(x, y, width, height)]

        # Define a green text label.
        x, y = 0.2, 0.7
        text = [QPen(QColor(0, 255, 0)), DS2Text(x, y, 'Found Something')]

        # Install the overlays.
        self.setMLOverlays([bbox, text])
```

The `classifyImage` method will be called for each frame, and always receives
one RGB image as a NumPy array with type Uint8. Pass that image to your
classifier any way you like.

Then, when you have found out what is where in the scene, you can add overlays
via the `setMLOverlays` method to highlight it. The arguments to that method is
a list of 2-tuples. The first element in each tuple must be a `QPen` instance,
and it defines the colour and transparency. The second argument specifies what
to draw. Currently, only `QRect` and `DS2Textures` elements are supported.

When you start the application again, you should see an output like this.

Single Frame | Spaceflight
:-------------------------:|:-------------------------:
<img src="docs/img/viewer_box.jpg" width="400">|<img src="docs/img/animated.gif" width="400">
