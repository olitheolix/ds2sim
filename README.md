# DS2Server

A simple Space Sim inspired sandbox to experiment with machine learning.

The module provides a set of rendered images like the ones below. Furthermore,
it provides a Qt application that mimics a flight through a 3D scene. The scene
itself consists of pre-rendered images only to avoid a dependency on OpenGL.


<img src="docs/img/cubes.jpg">

## Installation
```bash
pip install ds2server
```

## Generate Training Data
To generate the data set in the current directory, run
```bash
ds2generate
```
This will produce JPG encoded training images. This is all you need to train
your ML models.


## View the Space Simulation
This consists of two parts: a webserver that produces the images on request,
and a Qt client application to display it.

First, start the server with:
```bash
ds2server
```

Then, put the following code (mostly Qt boilerplate to create an application)
into a file and run it.
```python
import sys
import PyQt5.QtWidgets
import ds2server.viewer

# Default widget that will merely show the scene.
class MyViewer(ds2server.viewer.ViewerWidget):
    pass

# Define a camera name and assign it our widget.
cameras = {'Cam1': MyViewer}

# Qt boilerplate to start the application.
app = PyQt5.QtWidgets.QApplication(sys.argv)
widget = ds2server.viewer.MainWindow(cameras)
widget.show()
app.exec_()
```

This should create a Qt application like this:

<img src="docs/img/viewer.jpg" width="256">

The only line of interest here is the definition of `camera`. This dictionary
specifies how many cameras widgets want to use, and which widget to instantiate
for it. In this example, we just create a simple camera and use the default
widget, which will merely display the scene.


## Plug Your ML Model Into The Simulation
The real fun is, of course, to use your ML model to find and identify all the
obstacles in the scene. To do so, overload the `classifyImage` method
in the previous demo like so:

```python
class MyViewer(ds2server.viewer.ViewerWidget):
    def classifyImage(self, img):
        # `img` is always a <height, width, 3> NumPy image.
        assert img.dtype == np.uint8

        # Pass the image to your ML model.
        # myAwesomeClassifier(img)

        # Draw bounding boxes to highlight where you found what in the scene.
        # Here, as an example, we specify 3 random boxes.
        for i in range(3):
            self.showMLRect(
                x=random.random(),
                y=random.random(),
                width=random.random(),
                height=random.random(),
                rgba=[random.random() for _ in range(4)],
                text=f'Region {i}',
                text_pos=(0, 0),
            )
```

Here is an example of how this looks.

<div style="text-align:center"><img src="docs/img/viewer.gif" width="256"></div>
