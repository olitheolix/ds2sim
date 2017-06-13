import PyQt5.QtWidgets as QtWidgets
import ds2server.viewer
import numpy as np


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
        widget = ds2server.viewer.ClassifiedImageLabel()

        # Construct several invalid arguments, that is, any argument that
        # is not a list of the 9 elements [x, y, w, h, r, g, b, a, txt].
        invalid = [
            None, [None], [(1, 2, 3)], [list(range(9)), list(range(8))]
        ]

        for arg in invalid:
            assert widget.setMLRegions(arg) is False
            assert widget.ml_regions == []

    def test_setMLRegions_valid(self):
        widget = ds2server.viewer.ClassifiedImageLabel()

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
