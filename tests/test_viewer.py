import PyQt5.QtWidgets as QtWidgets
import tfds2.viewer
import unittest.mock as mock
import numpy as np


class TestBBox:
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def isCorrect(self, ret, ref):
        ref = 255 * np.array(ref, np.uint8)

        assert isinstance(ret, np.ndarray)
        assert ret.dtype == np.uint8
        assert ret.ndim == 3
        assert ret.shape == ref.shape + (3, )

        for i in range(3):
            assert np.array_equal(ret[:, :, i], ref)
        return True

    def empty(self):
        return np.zeros((3, 3, 3), np.uint8)

    def test_addBBoxToImage_bogus_data(self):
        addBBox = tfds2.viewer.addBBoxToImage

        assert addBBox(None, [(0, 0, 2, 2, 0.5)]) is None
        assert addBBox('foo', [(0, 0, 2, 2, 0.5)]) is None
        assert addBBox(np.zeros((3, 3, 3), np.uint8), 'foo') is None
        assert addBBox(np.zeros((3,), np.uint8), [(0, 0, 2, 2)]) is None
        assert addBBox(np.zeros((3, 3), np.uint8), [(0, 0, 2, 2)]) is None
        assert addBBox(np.zeros((3, 3, 3), np.uint8), [(0, 0, 2, 2)]) is None

    def test_addBBoxToImage_completely_outside(self):
        addBBox = tfds2.viewer.addBBoxToImage

        # Rectangle is completely outside.
        ref = np.zeros((3, 3))
        ret = addBBox(self.empty(), [(2, 2, 0, 0, 1)])
        assert self.isCorrect(ret, ref)

        ret = addBBox(self.empty(), [(-2, -2, 1, 1, 1)])
        assert self.isCorrect(ret, ref)
        ret = addBBox(self.empty(), [(3, 3, 1, 1, 1)])
        assert self.isCorrect(ret, ref)

    def test_addBBoxToImage_invalid_width_height(self):
        addBBox = tfds2.viewer.addBBoxToImage

        # Width or height is negative
        for width, height in [(-1, 2), (1, -2), (-1, -2)]:
            ret = addBBox(self.empty(), [(1, 1, width, height, 1)])
            assert self.isCorrect(ret, np.zeros((3, 3)))

    def test_addBBoxToImage_valid(self):
        addBBox = tfds2.viewer.addBBoxToImage

        ref = [[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]]
        ret = addBBox(self.empty(), [(0, 0, 3, 3, 1)])
        assert self.isCorrect(ret, ref)

        ref = [[1, 1, 1],
               [0, 0, 1],
               [1, 1, 1]]
        ret = addBBox(self.empty(), [(-5, 0, 8, 3, 1)])
        assert self.isCorrect(ret, ref)

        ref = [[0, 0, 0],
               [0, 1, 1],
               [0, 1, 0]]
        ret = addBBox(self.empty(), [(1, 1, 3, 4, 1)])
        assert self.isCorrect(ret, ref)

        ref = [[0, 0, 0],
               [0, 1, 1],
               [0, 1, 1]]
        ret = addBBox(self.empty(), [(1, 1, 2, 2, 1)])
        assert self.isCorrect(ret, ref)

        ref = [[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]]
        ret = addBBox(self.empty(), [(1, 1, 1, 1, 1)])
        assert self.isCorrect(ret, ref)


class TestClassifiedImageLabel:
    @classmethod
    def setup_class(cls):
        cls.app = QtWidgets.QApplication([])

    @classmethod
    def teardown_class(cls):
        del cls.app

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_setMLRegions_bogus_data(self):
        widget = tfds2.viewer.ClassifiedImageLabel()

        # Construct several invalid arguments, that is, any argument that does
        # is not a list of the 9 elements [x, y, w, h, r, g, b, a, txt].
        invalid = [
            None, [None], [(1, 2, 3)], [list(range(9)), list(range(8))]
        ]

        for arg in invalid:
            assert widget.setMLRegions(arg) is False
            assert widget.ml_regions == []

    def test_setMLRegions_valid(self):
        widget = tfds2.viewer.ClassifiedImageLabel()

        assert widget.ml_regions == []
        assert widget.setMLRegions([]) is True
        assert widget.ml_regions == []

        # Construct several invalid arguments, that is, any argument that does
        # is not a list of the 9 elements [x, y, w, h, r, g, b, a, txt].
        r = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        assert widget.setMLRegions(r) is True
        assert len(widget.ml_regions) == 1
        rect, rgba, txt = widget.ml_regions[0]
        assert rect == [0, 1, 1, 1]
        assert rgba == [255, 255, 255, 255]
        assert txt == '8'

        r = [[0, .1, .2, .3, .4, .5, .6, .7, 'foo']]
        assert widget.setMLRegions(r) is True
        assert len(widget.ml_regions) == 1
        rect, rgba, txt = widget.ml_regions[0]
        assert np.allclose(rect, [0, .1, .2, .3])
        assert np.allclose(255 * np.array([.4, .5, .6, .7]), rgba, atol=1)
        assert txt == 'foo'
