import io
import os
import glob
import random
import ds2server.horde.horde_simple
from PIL import Image


class Engine:
    def __init__(self):
        src_path = os.path.dirname(os.path.abspath(__file__))
        self.src_path = os.path.join(src_path, '..', 'dataset')

        self.horde = ds2server.horde.horde_simple.HordeSimple(512, 512)
        self.horde.loadCubes()
        self.horde.addLights()

    def renderScene(self, cmat, width, height):
        return self.horde.step(cmat)

        fnames = glob.glob(os.path.join(self.src_path, 'scene', '*.jpg'))
        assert len(fnames) > 0
        fname = random.sample(fnames, 1)[0]
        img = Image.open(fname)
        img = img.resize((width, height))

        f = io.BytesIO()
        img.save(f, format='jpeg')
        f.seek(0)
        return f.read()
