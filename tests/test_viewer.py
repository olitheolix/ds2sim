import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import ds2sim.viewer

# Convenience.
QPen = QtGui.QPen
QColor = QtGui.QColor
QRectF = QtCore.QRectF
DS2Text = ds2sim.viewer.DS2Text


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

    def test_setMLOverlays_bogus_data(self):
        widget = ds2sim.viewer.ClassifiedImageLabel()

        # Construct several invalid arguments, that is, any argument that
        # is not a list of the 9 elements [x, y, w, h, r, g, b, a, txt].
        invalid = [
            None, [None],
            [QPen(QColor())],
            [QPen(QColor()), QPen(QColor())],
            [DS2Text(.1, .2, 'foo'), QPen(QColor())],
        ]
        for arg in invalid:
            assert widget.setMLOverlays(arg) is False
            assert widget.ml_overlays == []

        # Convenience: define both a valid and invalid entry.
        valid = [QPen(QColor()), DS2Text(.1, .2, 'foo')]
        invalid = [DS2Text(.1, .2, 'foo'), QPen(QColor())]

        # The valid entry must be accepted.
        assert widget.setMLOverlays([valid]) is True
        assert widget.ml_overlays[0][0] is valid[0]
        assert widget.ml_overlays[0][1] is valid[1]

        # The invalid entry must be rejected. Also, the internal list of
        # overlays must be cleared.
        assert widget.setMLOverlays([invalid]) is False
        assert widget.ml_overlays == []

        # When passing a valid and invalid element then the return value must
        # still be False. However, the valid element must have made it into the
        # overlay list.
        assert widget.setMLOverlays([valid, invalid]) is False
        assert widget.ml_overlays[0][0] is valid[0]
        assert widget.ml_overlays[0][1] is valid[1]

    def test_setMLOverlays_valid(self):
        widget = ds2sim.viewer.ClassifiedImageLabel()

        # Nothing must happen when we supply an empty list.
        assert widget.ml_overlays == []
        assert widget.setMLOverlays([]) is True
        assert widget.ml_overlays == []

        overlays = [
            [QPen(QColor(1, 2, 3, 4)), QRectF(0, 0, 0.5, 0.7)],
            [QPen(QColor(1, 2, 3, 4)), DS2Text(x=0.5, y=0.6, text='foo')],
        ]
        assert widget.setMLOverlays(overlays) is True
        assert len(widget.ml_overlays) == 2
