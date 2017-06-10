import json
import urllib
import tfds2.webserver
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
        settings = {'cameras': {}, 'renderer': tfds2.rendering.Engine()}
        return tornado.web.Application(handlers, **settings)

    def test_getSetCameras(self):
        for cnames in [None, ['foo']]:
            body = urllib.parse.urlencode({'data': json.dumps(cnames)})
            ret = self.fetch('/get-camera', method='POST', body=body)
            assert ret.code == 200
            ret = json.loads(ret.body.decode('utf8'))
            expected = {} if cnames is None else {_: None for _ in cnames}
            assert ret == expected

        # Update/create a new camera.
        cam_vecs = np.eye(3).flatten().tolist()
        body = urllib.parse.urlencode({'data': json.dumps({'foo': cam_vecs})})
        self.fetch('/set-camera', method='POST', body=body)

        for cnames in [None, ['foo']]:
            body = urllib.parse.urlencode({'data': json.dumps(cnames)})
            ret = self.fetch('/get-camera', method='POST', body=body)
            assert ret.code == 200
            ret = json.loads(ret.body.decode('utf8'))
            assert ret == {'foo': cam_vecs}

    @mock.patch.object(tfds2.rendering.Engine, 'renderScene')
    def test_getRenderedImage(self, m_ri):
        width, height = 100, 200

        request = {
            'data': json.dumps({'camera': 'foo', 'width': width, 'height': height})}

        # Request rendered image from non-existing camera.
        body = urllib.parse.urlencode(request)
        m_ri.return_value = None
        ret = self.fetch('/get-render', method='POST', body=body)
        assert ret.code == 400
        m_ri.assert_called_with(cmat=None, width=width, height=height)

        # Camera data: 3x3 matrix. The columns are right/up/position.
        cam_vecs = np.eye(3).flatten().tolist()
        body = urllib.parse.urlencode({'data': json.dumps({'foo': cam_vecs})})
        self.fetch('/set-camera', method='POST', body=body)

        body = urllib.parse.urlencode(request)
        m_ri.return_value = b'foobar'
        ret = self.fetch('/get-render', method='POST', body=body)
        assert ret.code == 200
        m_ri.assert_called_with(cmat=cam_vecs, width=width, height=height)
        assert ret.body == b'foobar'

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

        # This 1x9 vector must be serialised into 4x4 float32 values.
        src = np.hstack([right, up, pos])
        fun = tfds2.webserver.compileCameraMatrix
        ret_cmat = fun(src.tolist())
        assert isinstance(ret_cmat, bytes)
        assert len(ret_cmat) == 16 * 4

        # The serialisation must have added the missing forward vector
        # and stored the matrix in column major format. Construct this
        # transformation here as well, then compare the arrays.
        assert np.allclose(np.fromstring(ret_cmat, np.float32), ref_cmat)

    def test_compileCameraMatrix_invalid(self):
        """Must return None if the matrix is invalid."""
        # Create random position and orthonormal matrix.
        pos = np.random.normal(0, 1, size=3)
        R = np.linalg.svd(np.random.normal(size=(3, 3)))[0]
        assert np.allclose(np.eye(3), R @ R.T)

        fun = tfds2.webserver.compileCameraMatrix

        # Wrong data types (must be equivalent to list of lists).
        assert fun(None) is None
        assert fun('foo') is None
        assert fun(['foo']) is None
        assert fun([['foo'], [1, 2]]) is None

        # Wrong dimensions (must be 9x1)
        assert fun(np.eye(3).tolist()) is None
        assert fun([[1, 2, 3, 4], [5, 6, 7], [8, 9], [10]]) is None

        # Compile a valid set of camera vectors.
        right, up = R[:2]
        cam_vecs = np.hstack([right, up, pos])
        assert fun(cam_vecs) is not None

        # Rotation matrix is not orthonormal.
        cam_vecs = np.hstack([right, right, pos])
        assert fun(cam_vecs) is None
