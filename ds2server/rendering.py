import os
import glob

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
        img_path = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(img_path, 'dataset', 'flightpath')
        fnames = glob.glob(os.path.join(img_path, '*.jpg'))
        fnames.sort()
        self.fnames = fnames
        self.frame_cnt = 0

    def renderScene(self, cmat, width, height):
        fname = self.fnames[self.frame_cnt]
        self.frame_cnt = (self.frame_cnt + 1) % len(self.fnames)
        return open(fname, 'rb').read()
