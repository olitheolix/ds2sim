import io
import os
import numpy as np
from PIL import Image

try:
    import ds2server.horde
    HORDE_INSTALLED = True
except ImportError:
    HORDE_INSTALLED = False


def getEngine(disable_horde=False):
    if disable_horde or not HORDE_INSTALLED:
        return DummyHorde()

    h = ds2server.horde.Engine(512, 512)
    h.loadCubes()
    h.addLights()
    return h


class DummyHorde:
    def __init__(self):
        src_path = os.path.dirname(os.path.abspath(__file__))
        self.src_path = os.path.join(src_path, '..', 'dataset')

    def renderScene(self, cmat, width, height):
        img = np.zeros((width, height, 3), np.uint8)
        x0, x1 = sorted(np.random.randint(0, width, 2).tolist())
        y0, y1 = sorted(np.random.randint(0, height, 2).tolist())
        for i in range(3):
            img[y0:y1, x0:x1, i] = np.random.randint(0, 255)
        img = Image.fromarray(img)

        f = io.BytesIO()
        img.save(f, format='jpeg')
        f.seek(0)
        return f.read()
