import json
import urllib
import ds2sim.camera
import ds2sim.webserver
import unittest.mock as mock
import tornado.web
import tornado.testing
import tornado.websocket

import numpy as np


class TestRestAPI(tornado.testing.AsyncHTTPTestCase):
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

    def get_app(self):
        handlers = [
            (r'/get-camera', ds2sim.webserver.RestGetCamera),
            (r'/set-camera', ds2sim.webserver.RestSetCamera),
            (r'/get-render', ds2sim.webserver.RestRenderScene),
        ]
        self.m_renderer = mock.MagicMock()
        settings = {'cameras': {}, 'renderer': self.m_renderer}
        return tornado.web.Application(handlers, **settings)

    def test_getCameras_empty(self):
        # No cameras exist, because we have not defined any. Therefore, any
        # fetch request must succeed, but return None for the respective cameras.
        for cnames in [['foo'], ['foo', 'bar']]:
            body = urllib.parse.urlencode({'data': json.dumps(cnames)})
            ret = self.fetch('/get-camera', method='POST', body=body)
            assert ret.code == 200
            expected = {name: None for name in cnames}
            assert json.loads(ret.body.decode('utf8')) == expected

        # When we query all cameras, then we must receive an empty dictionary.
        body = urllib.parse.urlencode({'data': json.dumps(None)})
        ret = self.fetch('/get-camera', method='POST', body=body)
        assert ret.code == 200
        assert json.loads(ret.body.decode('utf8')) == {}

    def test_getSetCameras_invalid(self):
        # Missing 'position' field.
        invalid_args = [
            {'foo': {'right': [1, 0, 0], 'up': [0, 1, 0]}},
            ['foo', None],
            {'foo': [1, 0, 0]},
            {'foo': {'right': [1, 0], 'up': [0, 1, 0]}},
            {'foo': {'right': [1, 0, 0], 'up': [0, 1, 0], 'pos': None}},
            {'foo': {'right': [1, 0, 0], 'up': [0, 1, 0], 'pos': [None, 0, 0]}},
        ]

        for arg in invalid_args:
            body = urllib.parse.urlencode({'data': json.dumps(arg)})
            ret = self.fetch('/set-camera', method='POST', body=body)
            assert ret.code == 400

            body = urllib.parse.urlencode({'data': json.dumps(None)})
            ret = self.fetch('/get-camera', method='POST', body=body)
            assert ret.code == 200 and json.loads(ret.body.decode('utf8')) == {}

    def test_getSetCameras(self):
        cameras = {
            'foo': {'right': [1, 0, 0], 'up': [0, 1, 0], 'pos': [0, 0, 0]},
            'bar': {'right': [0, 1, 0], 'up': [0, 0, 1], 'pos': [1, 2, 3]},
        }

        # Update/create the two cameras defined above.
        body = urllib.parse.urlencode({'data': json.dumps(cameras)})
        ret = self.fetch('/set-camera', method='POST', body=body)
        assert ret.code == 200

        # Fetch both cameras (ie. supply None, instead of a list of strings).
        body = urllib.parse.urlencode({'data': json.dumps(None)})
        ret = self.fetch('/get-camera', method='POST', body=body)
        assert ret.code == 200
        assert cameras == json.loads(ret.body.decode('utf8'))

        # Fetch the cameras individually.
        for cname, cdata in cameras.items():
            body = urllib.parse.urlencode({'data': json.dumps([cname])})
            ret = self.fetch('/get-camera', method='POST', body=body)
            assert ret.code == 200
            assert {cname: cdata} == json.loads(ret.body.decode('utf8'))

        # Fetch both cameras in a single request.
        body = urllib.parse.urlencode({'data': json.dumps(['foo', 'bar'])})
        ret = self.fetch('/get-camera', method='POST', body=body)
        assert ret.code == 200
        assert cameras == json.loads(ret.body.decode('utf8'))

        # Fetch two cameras, only one of which exists.
        body = urllib.parse.urlencode({'data': json.dumps(['foo', 'error'])})
        ret = self.fetch('/get-camera', method='POST', body=body)
        assert ret.code == 200
        ret = json.loads(ret.body.decode('utf8'))
        assert ret['error'] is None
        assert ret['foo'] == cameras['foo']

    @mock.patch.object(ds2sim.camera, 'compileCameraMatrix')
    @mock.patch.object(ds2sim.webserver, 'rawToJpeg')
    def test_getRenderedImage(self, m_rtjpeg, m_ccm):
        m_ccm.return_value = b'mock-cmat'
        m_rtjpeg.return_value = b'encoded-image'
        width, height = 100, 200

        payload = {'camera': 'foo', 'width': width, 'height': height}
        request = {'data': json.dumps(payload)}

        # Request rendered image from non-existing camera.
        m_rs = self.m_renderer.renderScene
        body = urllib.parse.urlencode(request)
        assert self.fetch('/get-render', method='POST', body=body).code == 400
        assert not m_rs.called

        # Define a camera.
        cameras = {'foo': {'right': [0, 0, 1], 'up': [0, 0, 1], 'pos': [0, 0, 0]}}
        body = urllib.parse.urlencode({'data': json.dumps(cameras)})
        assert self.fetch('/set-camera', method='POST', body=body).code == 200

        # Pretend that Horde returned a 2x2x3 image.
        m_rs.return_value = np.ones((2, 2, 3), np.uint8)

        # Request rendered image from existing camera.
        body = urllib.parse.urlencode(request)
        ret = self.fetch('/get-render', method='POST', body=body)
        assert ret.code == 200
        assert ret.body == b'encoded-image'
        m_rs.assert_called_with(cmat=b'mock-cmat', width=width, height=height)
        m_rtjpeg.assert_called_with(m_rs.return_value, quality=90)


class TestUtilityFunctions:
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

    def test_rawToJpeg_valid(self):
        """ Use a random NumPy matrix to check that all encoding options."""
        img = np.array((16, 16, 3), np.uint8)

        fun = ds2sim.webserver.rawToJpeg

        img = np.ones((16, 16, 3), np.uint8)
        ret = fun(img, quality=50)
        assert isinstance(ret, bytes)

        # Not uint8
        img = np.array((16, 16, 3), np.uint16)
        assert fun(img, quality=50) is None

        # Degenerate input.
        img = np.array((0, 16, 3), np.uint16)
        assert fun(img, quality=50) is None
