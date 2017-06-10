import json
import urllib
import tfds2.webserver
import tfds2.rendering
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
            (r'/get-camera', tfds2.webserver.RestGetCamera),
            (r'/set-camera', tfds2.webserver.RestSetCamera),
            (r'/get-render', tfds2.webserver.RestRenderScene),
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

    @mock.patch.object(tfds2.webserver, 'compileCameraMatrix')
    def test_getRenderedImage(self, m_ccm):
        m_ccm.return_value = b'mock-cmat'
        m_ri = self.m_renderer.renderScene
        width, height = 100, 200

        payload = {'camera': 'foo', 'width': width, 'height': height}
        request = {'data': json.dumps(payload)}

        # Request rendered image from non-existing camera.
        body = urllib.parse.urlencode(request)
        assert self.fetch('/get-render', method='POST', body=body).code == 400
        assert not m_ri.called

        # Define a camera.
        cameras = {'foo': {'right': [0, 0, 1], 'up': [0, 0, 1], 'pos': [0, 0, 0]}}
        body = urllib.parse.urlencode({'data': json.dumps(cameras)})
        assert self.fetch('/set-camera', method='POST', body=body).code == 200

        # Request rendered image from existing camera.
        m_ri.return_value = b'foobar'
        body = urllib.parse.urlencode(request)
        ret = self.fetch('/get-render', method='POST', body=body)
        assert ret.code == 200
        assert ret.body == b'foobar'
        m_ri.assert_called_with(cmat=b'mock-cmat', width=width, height=height)

    def test_compileCameraMatrix_valid(self):
        """Serialise a test matrix."""
        # Create random position.
        pos = np.random.normal(0, 1, 3)

        # Create a random orthonormal matrix.
        R = np.linalg.svd(np.random.normal(size=(3, 3)))[0]
        assert np.allclose(np.eye(3), R @ R.T)

        # The rotation matrix must span a right handed coordinate system.
        right, up = R[0, :], R[1, :]
        forward = np.cross(right, up)
        R[2, :] = forward
        assert np.allclose(np.eye(3), R @ R.T)

        # Compile the camera matrix with the position in the last _row_.
        # This is also how compileCameraMatrix must construct and serialise
        # it internally.
        ref_cmat = np.eye(4)
        ref_cmat[:3, :3] = R
        ref_cmat[3, :3] = pos
        ref_cmat = ref_cmat.astype(np.float64)
        ref_cmat = ref_cmat.flatten('C')

        # Camera vectors must be serialised into 4x4 float32 matrix.
        fun = tfds2.webserver.compileCameraMatrix
        ret_cmat = fun(right, up, pos)
        assert isinstance(ret_cmat, bytes)
        assert len(ret_cmat) == 16 * 4

        # The serialisation must have added the missing forward vector
        # and stored the matrix in column major format. Construct this
        # transformation here as well, then compare the arrays.
        assert np.allclose(np.fromstring(ret_cmat, np.float32), ref_cmat)

    def test_compileCameraMatrix_invalid(self):
        """Must return None if the matrix is invalid."""
        fun = tfds2.webserver.compileCameraMatrix

        # Wrong data types.
        assert fun(None, None, [1, 2, 3]) is None

        # Wrong dimensions: must be 3 vectors with 3 elements.
        assert fun([1, 2, 3, 4], [5, 6, 7], [8, 9]) is None

        # Right and Up are not orthogonal
        right, pos = [1, 0, 0], [0, 0, 0]
        assert fun(right, right, pos) is None

        # Rotation matrix is not orthonormal.
        assert fun(right, [0, 2, 0], pos) is None
